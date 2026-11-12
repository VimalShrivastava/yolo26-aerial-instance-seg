# Origin: NWPU mask_rcnn/train_nwpu_maskrcnn.py. Hard-coded paths replaced by --root; logic unchanged.
# train_nwpu_maskrcnn.py
print("Starting Mask RCNN training on NWPU...")

import torch
import torchvision
from torchvision.models.detection import \
    maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn \
    import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn \
    import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import json
import numpy as np
from pathlib import Path
from PIL import Image
import time
import cv2
import random

print("All imports successful!")

# ── Paths ──────────────────────────────────────────
from common import load_config, resolve_root
ROOT = resolve_root("NWPU_ROOT")

NWPU_DIR   = ROOT / "dataset"

IMG_DIR    = NWPU_DIR / "positive image set"

ANN_FILE   = NWPU_DIR / "annotations.json"

WORK_DIR   = ROOT / "work_dirs" / "paper" / "mask_rcnn" / "maskrcnn_nwpu"

SPLIT_FILE = ROOT / "yolo_dataset" / "train_val_split.json"

CFG          = load_config("maskrcnn_nwpu.yaml")
NUM_CLASSES  = CFG["num_classes"]
NUM_EPOCHS   = CFG["epochs"]
BATCH_SIZE   = CFG["batch_size"]
LR           = CFG["lr"]
MOMENTUM     = CFG["momentum"]
WEIGHT_DECAY = CFG["weight_decay"]
SEED         = CFG["seed"]

print(f"Paths configured!")
print(f"IMG_DIR  : {IMG_DIR}")
print(f"ANN_FILE : {ANN_FILE}")
print(f"WORK_DIR : {WORK_DIR}")

# ── Dataset ────────────────────────────────────────
class NWPUDataset(Dataset):
    def __init__(self, img_dir, ann_file,
                 img_ids):
        self.img_dir = Path(img_dir)
        with open(ann_file) as f:
            data = json.load(f)
        self.images = [
            img for img in data['images']
            if img['id'] in img_ids]
        self.img2anns = {}
        for ann in data['annotations']:
            iid = ann['image_id']
            if iid not in self.img2anns:
                self.img2anns[iid] = []
            self.img2anns[iid].append(ann)
        print(f"  Dataset: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id   = img_info['id']
        filename = img_info['file_name']
        img_w    = img_info['width']
        img_h    = img_info['height']

        img_path = self.img_dir / filename
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        anns = self.img2anns.get(img_id, [])
        anns = [a for a in anns
                if not a.get('iscrowd', 0)]

        boxes  = []
        labels = []
        masks  = []
        areas  = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            seg  = ann['segmentation']
            mask = np.zeros(
                (img_h, img_w), dtype=np.uint8)
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(
                        poly, dtype=np.int32
                    ).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

        if len(boxes) == 0:
            target = {
                'boxes'   : torch.zeros(
                    (0, 4), dtype=torch.float32),
                'labels'  : torch.zeros(
                    (0,), dtype=torch.int64),
                'masks'   : torch.zeros(
                    (0, img_h, img_w),
                    dtype=torch.uint8),
                'image_id': torch.tensor([img_id]),
                'area'    : torch.zeros(
                    (0,), dtype=torch.float32),
                'iscrowd' : torch.zeros(
                    (0,), dtype=torch.int64),
            }
        else:
            target = {
                'boxes'   : torch.tensor(
                    boxes, dtype=torch.float32),
                'labels'  : torch.tensor(
                    labels, dtype=torch.int64),
                'masks'   : torch.tensor(
                    np.array(masks),
                    dtype=torch.uint8),
                'image_id': torch.tensor([img_id]),
                'area'    : torch.tensor(
                    areas, dtype=torch.float32),
                'iscrowd' : torch.zeros(
                    len(boxes), dtype=torch.int64),
            }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(
        weights='DEFAULT')
    in_features = model.roi_heads\
        .box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads\
        .mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = \
        MaskRCNNPredictor(
            in_features_mask, 256, num_classes)
    return model


def load_split():
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as f:
            split = json.load(f)
        train_ids = set(split['train'])
        val_ids   = set(split['val'])
        print(f"Loaded split: {len(train_ids)} "
              f"train, {len(val_ids)} val")
    else:
        with open(ANN_FILE) as f:
            data = json.load(f)
        ids = [img['id'] for img in data['images']]
        random.seed(SEED)
        random.shuffle(ids)
        split_idx = int(len(ids) * 0.8)
        train_ids = set(ids[:split_idx])
        val_ids   = set(ids[split_idx:])
        SPLIT_FILE.parent.mkdir(
            parents=True, exist_ok=True)
        with open(SPLIT_FILE, 'w') as f:
            json.dump({
                'train': list(train_ids),
                'val'  : list(val_ids)
            }, f)
        print(f"Generated split: "
              f"{len(train_ids)} train, "
              f"{len(val_ids)} val")
    return train_ids, val_ids


# ── Main ───────────────────────────────────────────
print("\nInitializing training...")

WORK_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda') \
    if torch.cuda.is_available() \
    else torch.device('cpu')
print(f"Device: {device}")
print(f"GPU   : {torch.cuda.get_device_name(0)}")

train_ids, val_ids = load_split()

print("\nLoading datasets...")
train_dataset = NWPUDataset(
    IMG_DIR, ANN_FILE, train_ids)
val_dataset   = NWPUDataset(
    IMG_DIR, ANN_FILE, val_ids)

train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    collate_fn  = collate_fn,
    num_workers = 0)

print("\nLoading Mask R-CNN model...")
model = get_model(NUM_CLASSES)
model.to(device)

params = [p for p in model.parameters()
          if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr           = LR,
    momentum     = MOMENTUM,
    weight_decay = WEIGHT_DECAY)

lr_scheduler = torch.optim.lr_scheduler\
    .MultiStepLR(
        optimizer,
        milestones = CFG["lr_milestones"],
        gamma      = CFG["lr_gamma"])

print(f"\nStarting {NUM_EPOCHS} epoch training...")
best_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("-" * 40)

    model.train()
    total_loss = 0
    start = time.time()

    for i, (images, targets) in \
            enumerate(train_loader):
        images  = [img.to(device)
                   for img in images]
        targets = [{k: v.to(device)
                    for k, v in t.items()}
                   for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(train_loader)}] "
                  f"Loss: {losses.item():.4f} "
                  f"Time: {elapsed:.0f}s")

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch} avg loss: "
          f"{avg_loss:.4f}")

    lr_scheduler.step()

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(
            model.state_dict(),
            WORK_DIR / "best_model.pth")
        print(f"  ✅ Saved best model "
              f"(loss: {best_loss:.4f})")

    torch.save({
        'epoch'    : epoch,
        'model'    : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss'     : avg_loss,
    }, WORK_DIR / "last_checkpoint.pth")

print("\n✅ Mask RCNN NWPU Training Complete!")
print(f"Best model: {WORK_DIR}/best_model.pth")