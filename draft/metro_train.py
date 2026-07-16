import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import math
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SegformerFeatureExtractor,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    get_scheduler
)
from metro_dataset import MetroDataset


# === Ultility functions === #
# measures dissimilarity between the model prediction's map and the ground truth mask
def dice_loss_logits(logits, labels, eps=1e-6, ignore_index=None):
    # logits: (B, C, H, W), labels: (B, H, W)
    probs = torch.softmax(logits, dim=1)
    B, C, H, W = probs.shape
    labels_onehot = torch.zeros_like(probs)
    labels_exp = labels.unsqueeze(1)  # (B,1,H,W)
    labels_onehot.scatter_(1, labels_exp, 1.0)
    if ignore_index is not None:
        ignore_mask = (labels == ignore_index).unsqueeze(1)
        labels_onehot = labels_onehot.masked_fill(ignore_mask, 0.0)
        probs = probs.masked_fill(ignore_mask, 0.0)
    # compute dice per class
    dice_per_class = []
    for c in range(C):
        p = probs[:, c].reshape(B, -1)
        g = labels_onehot[:, c].reshape(B, -1)
        inter = (p * g).sum(dim=1)
        union = p.sum(dim=1) + g.sum(dim=1)
        dice = 1.0 - (2.0 * inter + eps) / (union + eps)
        dice_per_class.append(dice)
    dice_per_class = torch.stack(dice_per_class, dim=1)  # (B, C)
    return dice_per_class.mean()

# Calsulate mIoU (mean intersection over union) and pixel accuracy
def compute_inter_union_and_acc(preds_flat, gts_flat, num_classes, ignore_index=None):
    # Flattened numpy arrays
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    valid_mask = np.ones_like(gts_flat, dtype=bool)
    if ignore_index is not None:
        valid_mask = gts_flat != ignore_index
    total_correct = (preds_flat[valid_mask] == gts_flat[valid_mask]).sum()
    total = valid_mask.sum()
    for c in range(num_classes):
        pred_c = (preds_flat == c)
        gt_c = (gts_flat == c)
        inter[c] = np.logical_and(pred_c, gt_c & valid_mask).sum()
        union[c] = np.logical_or(pred_c & valid_mask, gt_c & valid_mask).sum()
    return inter, union, total_correct, total

# take the tensor format and convert into int8 for human visualization
def unnormalize_and_to_uint8(img_tensor, mean, std):
    # img_tensor: (H,W,3) float in normalized space; mean/std lists are same order as HF (RGB)
    img = img_tensor.copy()
    for i in range(3):
        img[..., i] = img[..., i] * std[i] + mean[i]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def colorize_mask(mask_np, id2color):
    H, W = mask_np.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for k, col in id2color.items():
        out[mask_np == k] = col
    return out

# === Argparse configuration === #
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str,
                        default=r'', help="train folder (images/, masks/)")
    parser.add_argument("--val_dir", type=str,
                        default=r'', help="val folder (images/, masks/)")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="runs/segformer")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--colors",
        type=str,
        default='{"0": [128,128,128], "1": [255,255,0], "2": [255,0,0]}',
        help="JSON: {class_id: [B,G,R]}"
    )

    # early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Stop training if validation metric doesn't improve for this many epochs")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4,
                        help="Minimum absolute improvement in mIoU to reset patience")
    parser.add_argument("--early_stopping_min_epochs", type=int, default=5,
                        help="Don't apply early stopping before this many epochs have run")
    return parser.parse_args()


# === Training Script === #
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    # Feature extractor
    feat = SegformerImageProcessor.from_pretrained(model_name)
    feat.reduce_labels = False
    feat.size = {"height": args.image_size, "width": args.image_size}

    # dataset
    train_ds = MetroDataset(args.train_dir, feature_extractor=feat, augment=True)
    val_ds = MetroDataset(args.val_dir, feature_extractor=feat, augment=False)

    id2label = train_ds.id2label
    label2id = train_ds.label2id
    num_classes = len(id2label)
    print("Num classes:", num_classes, "Labels:", id2label)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    # ==== Optimizer ====
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if len(p.shape) == 1 or n.endswith(".bias") or "norm" in n.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=args.learning_rate, eps=1e-8)  # keep eps for numerical stability

    # ==== Scheduler ====
    total_steps = math.ceil(len(train_loader) * args.num_epochs / args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ==== Loss and mixed precision ====
    if args.ignore_index >= 0:
        ce_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    else:
        ce_loss = nn.CrossEntropyLoss()  # No ignore_index
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # Color map for overlay: ensure args.colors is a dict (if passed from CLI, parse JSON first)
    # Here we convert values to tuples of ints (B,G,R) for cv2
    colors_dict = json.loads(args.colors)  # ← from CLI JSON string
    id2color = {
        int(k): tuple(int(v[i]) for i in (2, 1, 0))  # Convert RGB → BGR
        for k, v in colors_dict.items()
    }

    # checkpointing/resume variables
    best_miou = -1.0
    global_step = 0
    start_epoch = 0
    epochs_since_improve = 0

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_miou = float(ckpt.get("miou", -1.0))
        epochs_since_improve = int(ckpt.get("epochs_since_improve", 0))
        print(
            f"Resumed from {args.resume_from} -> start_epoch={start_epoch}, best_miou={best_miou:.4f}, epochs_since_improve={epochs_since_improve}")

    # Train loop
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.num_epochs}", leave=False, colour='cyan')
        running_loss = 0.0
        optimizer.zero_grad()
        for it, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)  # (B,3,H,W)
            labels = batch["labels"].to(device)  # (B,H,W)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(pixel_values)
                logits = outputs.logits
                # resize logits if needed
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                       align_corners=False)
                loss_ce = ce_loss(logits, labels)
                loss_d = dice_loss_logits(logits, labels, ignore_index=args.ignore_index if args.ignore_index >= 0 else None)
                loss = loss_ce + args.dice_weight * loss_d
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (it + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            running_loss += loss.item() * args.gradient_accumulation_steps
            if global_step % args.log_step == 0:
                writer.add_scalar("Train/Loss_step", loss.item() * args.gradient_accumulation_steps, global_step)
            pbar.set_postfix({"Loss": f"{(running_loss / (it + 1)):.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Train/Loss_epoch", avg_train_loss, epoch)

        # Validation loop
        model.eval()
        total_inter = np.zeros(num_classes, dtype=np.float64)
        total_union = np.zeros(num_classes, dtype=np.float64)
        total_correct = 0
        total_pixels = 0
        val_losses = []

        with torch.no_grad():
            for vb, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False, colour='yellow')):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(pixel_values)
                    logits = outputs.logits
                    if logits.shape[-2:] != labels.shape[-2:]:
                        logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                           align_corners=False)
                    loss_ce = ce_loss(logits, labels)
                    loss_d = dice_loss_logits(logits, labels, ignore_index=args.ignore_index if args.ignore_index >= 0 else None)
                    val_loss = loss_ce + args.dice_weight * loss_d
                val_losses.append(val_loss.item())

                preds = logits.argmax(dim=1).cpu().numpy()  # (B,H,W)
                gts = labels.cpu().numpy()
                for b in range(preds.shape[0]):
                    p_flat = preds[b].ravel()
                    g_flat = gts[b].ravel()
                    inter, uni, corr, tot = compute_inter_union_and_acc(p_flat, g_flat, num_classes,
                                                                        ignore_index=args.ignore_index)
                    total_inter += inter
                    total_union += uni
                    total_correct += corr
                    total_pixels += tot

                # log sample overlay once per epoch (first batch)
                if vb == 0:
                    pv = pixel_values[0].cpu().permute(1, 2, 0).numpy()  # (H,W,3)
                    # unnormalize using feature extractor mean/std
                    mean = feat.image_mean if hasattr(feat, "image_mean") else [0.0, 0.0, 0.0]
                    std = feat.image_std if hasattr(feat, "image_std") else [1.0, 1.0, 1.0]
                    img_disp = unnormalize_and_to_uint8(pv, mean, std)  # RGB
                    img_disp_bgr = img_disp[:, :, ::-1]  # BGR for cv2 ops

                    pred_mask = preds[0].astype(np.uint8)
                    gt_mask = gts[0].astype(np.uint8)
                    pred_color = colorize_mask(pred_mask, id2color)  # BGR
                    gt_color = colorize_mask(gt_mask, id2color)
                    overlay_pred = cv2.addWeighted(img_disp_bgr, 0.6, pred_color, 0.4, 0)
                    overlay_gt = cv2.addWeighted(img_disp_bgr, 0.6, gt_color, 0.4, 0)

                    # stack images horizontally: original | pred | gt (convert to RGB for tensorboard)
                    stacked = np.concatenate([img_disp, overlay_pred[:, :, ::-1], overlay_gt[:, :, ::-1]], axis=1)
                    stacked = stacked.transpose(2, 0, 1)  # C,H,W
                    writer.add_image("val/sample_orig_pred_gt", stacked, epoch)

        # finalize metrics
        iou = np.divide(total_inter, (total_union + 1e-9))
        miou = np.nanmean(iou)
        pix_acc = total_correct / (total_pixels + 1e-9)
        avg_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} TrainLoss {avg_train_loss:.4f} ValLoss {avg_val_loss:.4f} mIoU {miou:.4f} PixAcc {pix_acc:.4f}")
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/mIoU", miou, epoch)
        writer.add_scalar("Val/PixelAcc", pix_acc, epoch)

        # save checkpoints
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "miou": float(miou),  # force plain Python float
            "epochs_since_improve": int(epochs_since_improve),
        }

        last_path = os.path.join(args.output_dir, "last.pt")
        best_path = os.path.join(args.output_dir, "best.pt")
        best_weights_path = os.path.join(args.output_dir, "best_model_state.pt")

        torch.save(ckpt, last_path)

        if miou > best_miou + args.early_stopping_min_delta:
            best_miou = miou
            epochs_since_improve = 0
            torch.save(ckpt, best_path)
            torch.save(model.state_dict(), best_weights_path)  # <- clean inference/validation file
            print(f"New best mIoU: {best_miou:.4f} -> {best_path}")
        else:
            epochs_since_improve += 1
            print(
                f"No improvement in mIoU for {epochs_since_improve} epoch(s). (patience={args.early_stopping_patience})")

        # Early stopping check (only after minimum epochs)
        if (epoch + 1) >= args.early_stopping_min_epochs and epochs_since_improve >= args.early_stopping_patience:
            print(
                f"Early stopping triggered. No improvement in mIoU for {epochs_since_improve} epochs (patience={args.early_stopping_patience}).")
            writer.close()
            print("Training finished early.")
            break


    writer.close()
    print("Training complete.")






if __name__ == "__main__":
    args = get_args()
    train(args)


