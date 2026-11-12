# Origin: iSAID data_prep/coco_to_yolo_fixed.py. Hard-coded paths replaced by --root; logic unchanged.
import json
import shutil
import sys
from pathlib import Path
import numpy as np

from common import resolve_root
ROOT = resolve_root("ISAID_ROOT")

PATCHES_DIR = ROOT / "Dataset_patches"
OUTPUT_DIR  = ROOT / "yolo26_dataset"

# Standard category order (same as our fixed JSON)
STANDARD_NAMES = [
    'storage_tank', 'Large_Vehicle', 'Small_Vehicle', 'plane',
    'ship', 'Swimming_pool', 'Harbor', 'tennis_court',
    'Ground_Track_Field', 'Soccer_ball_field', 'baseball_diamond',
    'Bridge', 'basketball_court', 'Roundabout', 'Helicopter'
]

SPLITS = ["train", "val"]

# Safety guard added for the public release: the original script silently deleted
# <output>/images/<split> and <output>/labels/<split> before writing. Pass --overwrite to allow that.
OVERWRITE = "--overwrite" in sys.argv

def convert_split(split):
    print(f"\n{'='*50}")
    print(f"Converting {split} split...")

    ann_file  = PATCHES_DIR / split / "annotations" / f"iSAID_{split}_patches.json"
    img_dir   = PATCHES_DIR / split / "images"

    out_img_dir   = OUTPUT_DIR / "images" / split
    out_label_dir = OUTPUT_DIR / "labels" / split
    
    for d in (out_img_dir, out_label_dir):
        if d.exists() and any(d.iterdir()) and not OVERWRITE:
            sys.exit(f"{d} already contains files; re-run with --overwrite to delete and rebuild it")
    # Clean existing
    if out_img_dir.exists():
        shutil.rmtree(out_img_dir)
    if out_label_dir.exists():
        shutil.rmtree(out_label_dir)
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    # Verify category order matches standard
    print(f"Categories in JSON:")
    for cat in sorted(coco["categories"], key=lambda x: x["id"]):
        print(f"  ID {cat['id']:2d} → {cat['name']}")

    # COCO ID → YOLO index (0-based)
    # Since our JSON is already standardized:
    # ID 1 = storage_tank → YOLO index 0
    # ID 2 = Large_Vehicle → YOLO index 1
    # ...
    cat_ids  = sorted([cat["id"] for cat in coco["categories"]])
    cat_map  = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}  # 0-indexed!

    id2img = {img["id"]: img for img in coco["images"]}

    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    copied  = 0
    skipped = 0

    for img_info in coco["images"]:
        img_id   = img_info["id"]
        img_w    = img_info["width"]
        img_h    = img_info["height"]
        filename = img_info["file_name"]

        anns = ann_by_img.get(img_id, [])
        if not anns:
            skipped += 1
            continue

        # Copy image
        src = img_dir / filename
        dst = out_img_dir / filename
        if src.exists():
            shutil.copy2(src, dst)

        # Write YOLO label file
        label_path = out_label_dir / (Path(filename).stem + ".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                cat_id   = ann["category_id"]
                yolo_idx = cat_map[cat_id]  # 0-indexed

                seg = ann["segmentation"]
                if not seg:
                    continue

                seg = max(seg, key=len)
                pts = np.array(seg, dtype=np.float64).reshape(-1, 2)
                pts[:, 0] /= img_w
                pts[:, 1] /= img_h
                pts = np.clip(pts, 0, 1)

                coords = " ".join([f"{v:.6f}" for v in pts.flatten()])
                f.write(f"{yolo_idx} {coords}\n")

        copied += 1

    print(f"  Converted : {copied} images")
    print(f"  Skipped   : {skipped} images (no annotations)")

# Convert both splits
for split in SPLITS:
    convert_split(split)

# Save corrected dataset.yaml
yaml_content = f"""# iSAID Dataset - YOLO26 format
path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(STANDARD_NAMES)}
names: {STANDARD_NAMES}
"""

yaml_path = OUTPUT_DIR / "isaid.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n✅ Conversion complete!")
print(f"yaml saved: {yaml_path}")
print(f"Categories: {STANDARD_NAMES}")