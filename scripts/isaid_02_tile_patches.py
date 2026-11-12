# Origin: iSAID data_prep/split_patches.py (800 px patches, OVERLAP = 200 px, i.e. stride 600 px). Hard-coded paths replaced by --root; logic unchanged.
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

# ── Config ────────────────────────────────────────────────────────────
from common import resolve_root
ROOT = resolve_root("ISAID_ROOT")

DATASET_DIR  = ROOT / "Dataset"
OUTPUT_DIR   = ROOT / "Dataset_patches"
PATCH_SIZE   = 800
OVERLAP      = 200
MIN_AREA     = 200
# ─────────────────────────────────────────────────────────────────────

# ── Standard fixed category mapping ──────────────────────────────────
# This ensures BOTH train and val always use the same category IDs
# regardless of the order they appear in the original iSAID JSON
STANDARD_CATEGORIES = [
    {"id": 1,  "name": "storage_tank"},
    {"id": 2,  "name": "Large_Vehicle"},
    {"id": 3,  "name": "Small_Vehicle"},
    {"id": 4,  "name": "plane"},
    {"id": 5,  "name": "ship"},
    {"id": 6,  "name": "Swimming_pool"},
    {"id": 7,  "name": "Harbor"},
    {"id": 8,  "name": "tennis_court"},
    {"id": 9,  "name": "Ground_Track_Field"},
    {"id": 10, "name": "Soccer_ball_field"},
    {"id": 11, "name": "baseball_diamond"},
    {"id": 12, "name": "Bridge"},
    {"id": 13, "name": "basketball_court"},
    {"id": 14, "name": "Roundabout"},
    {"id": 15, "name": "Helicopter"}
]

# name → standard ID lookup
NAME2ID = {cat["name"]: cat["id"] for cat in STANDARD_CATEGORIES}

def get_patches(img_w, img_h, patch_size, overlap):
    stride  = patch_size - overlap
    patches = []
    y = 0
    while y < img_h:
        x  = 0
        y2 = min(y + patch_size, img_h)
        while x < img_w:
            x2 = min(x + patch_size, img_w)
            patches.append((x, y, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride
    return patches

def clip_bbox(bbox, x1, y1, patch_w, patch_h):
    bx, by, bw, bh = bbox
    bx2, by2 = bx + bw, by + bh
    cx1 = max(bx, x1) - x1
    cy1 = max(by, y1) - y1
    cx2 = min(bx2, x1 + patch_w) - x1
    cy2 = min(by2, y1 + patch_h) - y1
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return [cx1, cy1, cx2 - cx1, cy2 - cy1]

def clip_segmentation(segmentation, x1, y1, patch_w, patch_h):
    clipped = []
    for seg in segmentation:
        points = np.array(seg).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0] - x1, 0, patch_w)
        points[:, 1] = np.clip(points[:, 1] - y1, 0, patch_h)
        clipped.append(points.flatten().tolist())
    return clipped

def process_split(split):
    print(f"\n{'='*50}")
    print(f"Processing {split} split...")
    print(f"{'='*50}")

    img_dir  = DATASET_DIR / split / "Images"
    ann_file = DATASET_DIR / split / "Annotations" / f"iSAID_{split}.json"

    out_img_dir = OUTPUT_DIR / split / "images"
    out_ann_dir = OUTPUT_DIR / split / "annotations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    # ── Build category remapping: original ID → standard ID ──────────
    # This is the key fix — remap original IDs to our standard IDs
    orig_cat_id2name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    orig2standard    = {
        cat_id: NAME2ID[cat_name]
        for cat_id, cat_name in orig_cat_id2name.items()
        if cat_name in NAME2ID
    }

    print(f"Category remapping:")
    for orig_id, std_id in sorted(orig2standard.items()):
        print(f"  {orig_cat_id2name[orig_id]}: {orig_id} → {std_id}")

    # Build annotation lookup
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    new_images      = []
    new_annotations = []
    new_img_id      = 1
    new_ann_id      = 1

    for img_info in coco["images"]:
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"  ⚠ Missing: {img_info['file_name']}, skipping")
            continue

        img      = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        anns     = ann_by_img.get(img_info["id"], [])
        patches  = get_patches(img_w, img_h, PATCH_SIZE, OVERLAP)

        print(f"  {img_info['file_name']} ({img_w}x{img_h}) → {len(patches)} patches")

        for (px1, py1, px2, py2) in patches:
            patch_w = px2 - px1
            patch_h = py2 - py1

            patch_img  = img.crop((px1, py1, px2, py2))
            patch_name = f"{img_path.stem}_{px1}_{py1}_{px2}_{py2}.png"
            patch_img.save(out_img_dir / patch_name)

            new_images.append({
                "id"       : new_img_id,
                "file_name": patch_name,
                "width"    : patch_w,
                "height"   : patch_h
            })

            for ann in anns:
                new_bbox = clip_bbox(ann["bbox"], px1, py1, patch_w, patch_h)
                if new_bbox is None:
                    continue
                if new_bbox[2] * new_bbox[3] < MIN_AREA:
                    continue

                new_seg  = clip_segmentation(ann["segmentation"], px1, py1, patch_w, patch_h)
                new_area = new_bbox[2] * new_bbox[3]

                # ── Use standard category ID ──────────────────────────
                std_cat_id = orig2standard.get(ann["category_id"])
                if std_cat_id is None:
                    continue  # skip unknown categories

                new_annotations.append({
                    "id"          : new_ann_id,
                    "image_id"    : new_img_id,
                    "category_id" : std_cat_id,  # ← standardized ID
                    "bbox"        : new_bbox,
                    "segmentation": new_seg,
                    "area"        : new_area,
                    "iscrowd"     : ann.get("iscrowd", 0)
                })
                new_ann_id += 1

            new_img_id += 1

    # Save with STANDARD categories
    new_coco = {
        "images"     : new_images,
        "annotations": new_annotations,
        "categories" : STANDARD_CATEGORIES  # ← always standard
    }

    out_json = out_ann_dir / f"iSAID_{split}_patches.json"
    with open(out_json, "w") as f:
        json.dump(new_coco, f)

    print(f"\n✅ {split} done!")
    print(f"   Patches     : {len(new_images)}")
    print(f"   Annotations : {len(new_annotations)}")
    print(f"   JSON saved  : {out_json}")

# Run for both splits
if __name__ == "__main__":
    process_split("train")
    process_split("val")