import os
import cv2
import shutil
from tqdm import tqdm
import albumentations as A

# === CONFIG ===
train_img_dir = r""
train_mask_dir = r""

# How many augmented variants to create *per original image*
augmentations_per_image = 2

# Allowed image & mask extensions (order matters if mask uses a different ext)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# === Mild image-only augmentation pipeline (no geometric transforms) ===
transform = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.6),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=12, p=0.5),
        A.Blur(blur_limit=2, p=0.25),
        # you can add other light color transforms if desired
    ],
    p=1.0,
)

# === Helpers ===
def find_mask_for_image(img_name, mask_dir):
    base = os.path.splitext(img_name)[0]
    for ext in IMG_EXTS:
        candidate = os.path.join(mask_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def unique_path(path):
    """If path exists, append _1, _2, ... before extension to avoid overwrite."""
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

# === Run augmentation ===
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)

image_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(IMG_EXTS)]
print(f"Found {len(image_files)} training images. Augmenting...")

added = 0
skipped_no_mask = 0
for img_name in tqdm(image_files):
    img_path = os.path.join(train_img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image {img_name}, skipping.")
        continue

    mask_path = find_mask_for_image(img_name, train_mask_dir)
    if mask_path is None:
        # If you require exact 1:1 pairs, it's safer to skip images without masks
        print(f"No matching mask found for {img_name}; skipping augmentation for this file.")
        skipped_no_mask += 1
        continue

    base, img_ext = os.path.splitext(img_name)
    mask_ext = os.path.splitext(mask_path)[1]

    for i in range(1, augmentations_per_image + 1):
        result = transform(image=img)
        aug_img = result["image"]

        new_img_name = f"{base}_aug{i}{img_ext}"
        new_img_path = os.path.join(train_img_dir, new_img_name)
        new_img_path = unique_path(new_img_path)

        # write augmented image
        success = cv2.imwrite(new_img_path, aug_img)
        if not success:
            print(f"Failed to write {new_img_path}; skipping this augmentation.")
            continue

        # copy mask as-is but with the same augmented basename
        new_mask_name = f"{base}_aug{i}{mask_ext}"
        new_mask_path = os.path.join(train_mask_dir, new_mask_name)
        new_mask_path = unique_path(new_mask_path)
        shutil.copyfile(mask_path, new_mask_path)

        added += 1

print(f"Done. Added {added} augmented images (and copied masks).")
if skipped_no_mask:
    print(f"Skipped {skipped_no_mask} images because no matching mask was found for them.")
