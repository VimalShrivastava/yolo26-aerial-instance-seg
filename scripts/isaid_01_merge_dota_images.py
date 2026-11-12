# Origin: iSAID data_prep/merge.py. Output folder renamed images -> Images to match isaid_02 (case matters on Linux). Hard-coded paths replaced by --root; logic unchanged.
import shutil, json
from pathlib import Path

from common import resolve_root
ROOT = resolve_root("ISAID_ROOT")

ISAID_DATASET       = ROOT / "Dataset"
DOTA_DOWNLOAD_TRAIN = ISAID_DATASET / "DOTA_train"
DOTA_DOWNLOAD_VAL   = ISAID_DATASET / "DOTA_val"

train_out = ISAID_DATASET / "train" / "Images"
val_out   = ISAID_DATASET / "val"   / "Images"
train_out.mkdir(exist_ok=True)
val_out.mkdir(exist_ok=True)

# Load expected filenames from JSON
with open(ISAID_DATASET / "train" / "Annotations" / "iSAID_train.json") as f:
    train_images = {img["file_name"] for img in json.load(f)["images"]}

with open(ISAID_DATASET / "val" / "Annotations" / "iSAID_val.json") as f:
    val_images = {img["file_name"] for img in json.load(f)["images"]}

print(f"Train expects : {len(train_images)} images")
print(f"Val expects   : {len(val_images)} images")

# Merge DOTA_train parts
print("\n--- Processing DOTA_train ---")
for part in ["part1", "part2", "part3"]:
    part_path = DOTA_DOWNLOAD_TRAIN / part / "images"
    if not part_path.exists():
        print(f"Skipping DOTA_train/{part} - not found")
        continue
    files = list(part_path.glob("*.png"))
    print(f"Processing {part}: {len(files)} images found")
    for f in files:
        if f.name in train_images:
            shutil.copy2(f, train_out / f.name)
        elif f.name in val_images:
            shutil.copy2(f, val_out / f.name)

# Merge DOTA_val parts
print("\n--- Processing DOTA_val ---")
for part in ["part1", "part2"]:
    part_path = DOTA_DOWNLOAD_VAL / part / "images"
    if not part_path.exists():
        print(f"Skipping DOTA_val/{part} - not found")
        continue
    files = list(part_path.glob("*.png"))
    print(f"Processing {part}: {len(files)} images found")
    for f in files:
        if f.name in train_images:
            shutil.copy2(f, train_out / f.name)
        elif f.name in val_images:
            shutil.copy2(f, val_out / f.name)

print(f"\n✅ Done!")
print(f"Train Images copied : {len(list(train_out.glob('*.png')))}")
print(f"Val   Images copied : {len(list(val_out.glob('*.png')))}")