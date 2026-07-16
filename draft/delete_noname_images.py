import os

# === CONFIG ===
images_dir = r""   # e.g. r"D:\dataset\images"
masks_dir = r""     # e.g. r"D:\dataset\masks"
image_exts = {".jpg", ".jpeg", ".png"}   # allowed image formats
mask_exts = {".png", ".jpg"}             # allowed mask formats
delete_files = True                      # set False first to test safely

# === STEP 1: collect base filenames ===
mask_names = {os.path.splitext(f)[0] for f in os.listdir(masks_dir)
              if os.path.splitext(f)[1].lower() in mask_exts}

# === STEP 2: scan images and delete unmatched ===
removed = 0
for img_file in os.listdir(images_dir):
    name, ext = os.path.splitext(img_file)
    if ext.lower() not in image_exts:
        continue  # skip non-image files

    if name not in mask_names:
        removed += 1
        file_path = os.path.join(images_dir, img_file)
        if delete_files:
            os.remove(file_path)
            print(f"Deleted: {img_file}")
        else:
            print(f"Would delete: {img_file}")

print(f"\nDone. {'Deleted' if delete_files else 'Would delete'} {removed} images without masks.")
