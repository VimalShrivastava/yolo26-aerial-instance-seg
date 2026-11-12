# Origin: NWPU data_prep/prepare_nwpu.py (80:20 split, seed 42). Hard-coded paths replaced by --root; logic unchanged.

import json
import shutil
import random
from pathlib import Path
from PIL import Image

# ── Paths ──────────────────────────────────────────
from common import resolve_root
ROOT = resolve_root("NWPU_ROOT")

NWPU_DIR    = ROOT / "dataset"
IMG_DIR     = NWPU_DIR / "positive image set"
ANN_FILE    = NWPU_DIR / "annotations.json"
OUTPUT_DIR  = ROOT / "yolo_dataset"

# ── Settings ───────────────────────────────────────
TRAIN_RATIO = 0.8
SEED        = 42

# ── Category mapping ───────────────────────────────
CATEGORIES = {
    1: "airplane",
    2: "ship",
    3: "storage_tank",
    4: "baseball_diamond",
    5: "tennis_court",
    6: "basketball_court",
    7: "ground_track_field",
    8: "harbor",
    9: "bridge",
    10: "vehicle"
}

def create_dirs():
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True, exist_ok=True)
    print("✅ Directories created")

def load_annotations():
    with open(ANN_FILE) as f:
        data = json.load(f)
    
    # Build image id to filename map
    id2file = {
        img['id']: img['file_name']
        for img in data['images']
    }
    
    # Build image id to size map
    id2size = {
        img['id']: (img['width'], img['height'])
        for img in data['images']
    }
    
    # Group annotations by image
    img2anns = {}
    for ann in data['annotations']:
        iid = ann['image_id']
        if iid not in img2anns:
            img2anns[iid] = []
        img2anns[iid].append(ann)
    
    return data['images'], id2file, id2size, img2anns

def coco_seg_to_yolo(segmentation, img_w, img_h):
    """Convert COCO polygon to YOLO segmentation format"""
    yolo_segs = []
    for seg in segmentation:
        if len(seg) < 6:
            continue
        points = []
        for i in range(0, len(seg), 2):
            x = seg[i] / img_w
            y = seg[i+1] / img_h
            points.extend([x, y])
        yolo_segs.append(points)
    return yolo_segs

def split_dataset(images):
    random.seed(SEED)
    ids = [img['id'] for img in images]
    random.shuffle(ids)
    split = int(len(ids) * TRAIN_RATIO)
    return set(ids[:split]), set(ids[split:])

def process_split(img_ids, split, 
                  id2file, id2size, img2anns):
    copied = 0
    skipped = 0
    
    for img_id in img_ids:
        filename = id2file[img_id]
        src = IMG_DIR / filename
        
        if not src.exists():
            skipped += 1
            continue
        
        # Copy image
        dst = OUTPUT_DIR / "images" / split / filename
        shutil.copy2(src, dst)
        
        # Write YOLO label
        img_w, img_h = id2size[img_id]
        anns = img2anns.get(img_id, [])
        
        label_file = (OUTPUT_DIR / "labels" / split / 
                      filename.replace('.jpg', '.txt'))
        
        with open(label_file, 'w') as f:
            for ann in anns:
                if ann.get('iscrowd', 0):
                    continue
                
                cat_id = ann['category_id'] - 1
                segs = coco_seg_to_yolo(
                    ann['segmentation'], img_w, img_h)
                
                for seg in segs:
                    if len(seg) < 6:
                        continue
                    points_str = ' '.join(
                        f'{p:.6f}' for p in seg)
                    f.write(f"{cat_id} {points_str}\n")
        
        copied += 1
    
    print(f"  {split}: {copied} images, "
          f"{skipped} skipped")
    return copied

def create_yaml():
    yaml_content = f"""# NWPU VHR-10 Dataset
path: {OUTPUT_DIR}
train: images/train
val: images/val

nc: 10
names:
  0: airplane
  1: ship
  2: storage_tank
  3: baseball_diamond
  4: tennis_court
  5: basketball_court
  6: ground_track_field
  7: harbor
  8: bridge
  9: vehicle
"""
    yaml_path = OUTPUT_DIR / "nwpu.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"✅ YAML saved: {yaml_path}")

def main():
    print("=== NWPU VHR-10 Dataset Preparation ===\n")
    
    # Create directories
    create_dirs()
    
    # Load annotations
    images, id2file, id2size, img2anns = \
        load_annotations()
    print(f"✅ Loaded {len(images)} images, "
          f"{sum(len(v) for v in img2anns.values())} "
          f"annotations")
    
    # Split dataset
    train_ids, val_ids = split_dataset(images)
    print(f"✅ Split: {len(train_ids)} train, "
          f"{len(val_ids)} val")
    
    # Process splits
    print("\nProcessing splits:")
    process_split(train_ids, "train",
                  id2file, id2size, img2anns)
    process_split(val_ids, "val",
                  id2file, id2size, img2anns)
    
    # Create YAML
    create_yaml()
    
    print("\n=== Dataset preparation complete! ===")
    print(f"Output: {OUTPUT_DIR}")
    print(f"YAML  : {OUTPUT_DIR}/nwpu.yaml")

if __name__ == '__main__':
    main()