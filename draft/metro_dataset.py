from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
import numpy as np
from PIL import Image
import glob


class MetroDataset(Dataset):
    def __init__(self, root_dir, feature_extractor=None, augment=False):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        # === CLASS MAPPING ===
        self.id2label = {
            0: 'background',
            1: 'blindway',
            2: 'curb_ramp',
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_classes = len(self.id2label)  # 3

        # === LOAD FILE LISTS ===
        image_exts = ('.jpg', '.jpeg', '.png')
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(image_exts)
        ])
        self.masks = sorted([
            f for f in os.listdir(self.mask_dir)
            if f.lower().endswith(image_exts)
        ])

        assert len(self.images) == len(self.masks), \
            f"Image/Mask mismatch: {len(self.images)} images vs {len(self.masks)} masks"

        # === VALIDATE ALL MASKS ONCE (Optional but safe) ===
        self._validate_and_fix_masks()

    def _validate_and_fix_masks(self):
        print(f"Validating {len(self.masks)} masks...")
        fixed_count = 0
        for mask_name in self.masks:
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = np.array(Image.open(mask_path))

            # Only warn if values outside [0,2] and not 255
            invalid = ((mask < 0) | (mask > 2)) & (mask != 255)
            if invalid.any():
                bad_vals = np.unique(mask[invalid])
                print(f"  WARNING: {mask_name} has invalid labels: {bad_vals.tolist()}")
                mask = np.clip(mask, 0, 2)  # clamp to valid classes
                fixed_mask = Image.fromarray(mask.astype(np.uint8))
                fixed_mask.save(mask_path)
                fixed_count += 1

        if fixed_count == 0:
            print("  All masks are clean!")
        else:
            print(f"  Fixed {fixed_count} masks.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        # === FINAL SAFETY: Double-check mask (redundant but bulletproof) ===
        mask_np = np.array(mask)
        if mask_np.max() >= self.num_classes or mask_np.min() < 0:
            print(f"  RUNTIME FIX: {self.masks[idx]} still invalid → clamping...")
            mask_np = np.clip(mask_np, 0, self.num_classes - 1)
            mask = Image.fromarray(mask_np.astype(np.uint8))

        # === Apply feature extractor ===
        if self.feature_extractor:
            encoded = self.feature_extractor(
                images=image,
                segmentation_maps=mask,
                return_tensors="pt"
            )
            # Remove batch dim
            for k in encoded:
                encoded[k] = encoded[k].squeeze(0)
            return encoded

        return {"image": image, "mask": mask}