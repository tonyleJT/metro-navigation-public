import os
import shutil
from pathlib import Path
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
RAW_ROOT      = Path(r"")          # <-- folder that contains images/ and masks/
DATASET_ROOT  = Path(r"")      # <-- where the split will be created
TRAIN_RATIO   = 0.80                 # 80 % train, 20 % val
SEED          = 42
random.seed(SEED)

# colour → class_id (order matters for later evaluation!)
color_to_id = {
    (128, 128, 128): 0,   # background
    (255, 255, 0):   1,   # blindway
    (255, 0, 0):     2,   # curb_ramp
}
# optional: ignore-label for pixels that do not match any colour
IGNORE_COLOR = (0, 0, 0)   # black → will be set to 255 (ignore_index)

# ----------------------------------------------------------------------
# 2. CREATE FOLDER STRUCTURE
# ----------------------------------------------------------------------
for split in ["train", "val"]:
    (DATASET_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / split / "masks").mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 3. LIST ALL IMAGE / MASK PAIRS
# ----------------------------------------------------------------------
image_dir = RAW_ROOT / "images"
mask_dir  = RAW_ROOT / "masks"

image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
mask_paths  = sorted([p for p in mask_dir.iterdir()  if p.suffix.lower() == ".png"])

# sanity check: same number & same names
assert len(image_paths) == len(mask_paths), "Number of images != number of masks"
for img_p, mask_p in zip(image_paths, mask_paths):
    assert img_p.stem == mask_p.stem, f"Name mismatch: {img_p.name} vs {mask_p.name}"

pairs = list(zip(image_paths, mask_paths))

# ----------------------------------------------------------------------
# 4. SHUFFLE & SPLIT
# ----------------------------------------------------------------------
random.shuffle(pairs)
split_idx = int(len(pairs) * TRAIN_RATIO)
train_pairs = pairs[:split_idx]
val_pairs   = pairs[split_idx:]

print(f"Total pairs : {len(pairs)}")
print(f"Train       : {len(train_pairs)}")
print(f"Val         : {len(val_pairs)}")

# ----------------------------------------------------------------------
# 5. HELPER: RGB → INDEX mask
# ----------------------------------------------------------------------
def rgb_to_index_mask(rgb_mask: Image.Image) -> Image.Image:
    """Convert a PIL RGB mask → single-channel uint8 index mask."""
    arr = np.array(rgb_mask)                     # (H, W, 3)
    h, w = arr.shape[:2]
    index = np.full((h, w), 255, dtype=np.uint8) # 255 = ignore_index

    for (r, g, b), idx in color_to_id.items():
        match = (arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)
        index[match] = idx

    # optional: treat pure black as ignore as well
    black = (arr[..., 0] == 0) & (arr[..., 1] == 0) & (arr[..., 2] == 0)
    index[black] = 255

    return Image.fromarray(index)

# ----------------------------------------------------------------------
# 6. COPY & CONVERT
# ----------------------------------------------------------------------
def process_split(pairs_list, split_name):
    img_out = DATASET_ROOT / split_name / "images"
    mask_out = DATASET_ROOT / split_name / "masks"

    for img_path, mask_path in tqdm(pairs_list, desc=f"Processing {split_name}"):
        # copy image (unchanged)
        shutil.copy(img_path, img_out / img_path.name)

        # convert mask
        rgb_mask = Image.open(mask_path).convert("RGB")
        idx_mask = rgb_to_index_mask(rgb_mask)
        idx_mask.save(mask_out / (mask_path.stem + ".png"))

# ----------------------------------------------------------------------
process_split(train_pairs, "train")
process_split(val_pairs,   "val")
print("Done! Your dataset is ready at:", DATASET_ROOT)