# Origin: iSAID mask_rcnn/train_maskrcnn.py. Hard-coded paths replaced by --root; logic unchanged.

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
from pathlib import Path
import time

# ── Config ────────────────────────────────────────────────────────────
from common import load_config, resolve_root
ROOT = resolve_root("ISAID_ROOT")

PATCHES_DIR  = ROOT / "Dataset_patches"
WORK_DIR     = ROOT / "work_dirs" / "paper" / "maskrcnn_torchvision"
CFG          = load_config("maskrcnn_isaid.yaml")
NUM_CLASSES  = CFG["num_classes"]   # 15 iSAID classes + background
NUM_EPOCHS   = CFG["epochs"]
BATCH_SIZE   = CFG["batch_size"]
LR           = CFG["lr"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────

WORK_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Dataset ───────────────────────────────────────────────────────────
class iSAIDDataset(Dataset):
    def __init__(self, patches_dir, split):
        self.img_dir  = patches_dir / split / "images"
        ann_file      = patches_dir / split / "annotations" / f"iSAID_{split}_patches.json"

        with open(ann_file) as f:
            coco = json.load(f)

        # Build category mapping: coco_id → 1-indexed (0 = background)
        cat_ids      = sorted([cat["id"] for cat in coco["categories"]])
        self.cat_map = {cid: idx+1 for idx, cid in enumerate(cat_ids)}

        # Build image lookup
        id2img = {img["id"]: img for img in coco["images"]}

        # Group annotations by image, keep only images with annotations
        ann_by_img = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for img_id, anns in ann_by_img.items():
            self.samples.append((id2img[img_id], anns))

        print(f"[{split}] {len(self.samples)} images with annotations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info, anns = self.samples[idx]

        # Load image
        img_path = self.img_dir / img_info["file_name"]
        img      = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img_tensor   = torchvision.transforms.functional.to_tensor(img)

        # Build targets
        boxes   = []
        labels  = []
        masks   = []

        for ann in anns:
            # BBox: COCO [x,y,w,h] → [x1,y1,x2,y2]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x+w, y+h
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_map[ann["category_id"]])

            # Mask from segmentation polygon
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                import cv2
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

        if not boxes:
            # Return empty target if no valid annotations
            target = {
                "boxes"  : torch.zeros((0, 4), dtype=torch.float32),
                "labels" : torch.zeros((0,), dtype=torch.int64),
                "masks"  : torch.zeros((0, img_h, img_w), dtype=torch.uint8),
                "image_id": torch.tensor([idx])
            }
        else:
            target = {
                "boxes"  : torch.tensor(boxes, dtype=torch.float32),
                "labels" : torch.tensor(labels, dtype=torch.int64),
                "masks"  : torch.tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([idx])
            }

        return img_tensor, target

# ── Model ─────────────────────────────────────────────────────────────
def get_model(num_classes):
    # Load pretrained on COCO
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace bbox head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

# ── Collate ───────────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))

# ── Training loop ─────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0
    start      = time.time()

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i+1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  Epoch [{epoch}] Step [{i+1}/{len(loader)}] "
                  f"Loss: {losses.item():.4f} "
                  f"Time: {elapsed:.1f}s")
            start = time.time()

    return total_loss / len(loader)

# ── Main ──────────────────────────────────────────────────────────────
def main():
    # Datasets
    train_dataset = iSAIDDataset(PATCHES_DIR, "train")
    val_dataset   = iSAIDDataset(PATCHES_DIR, "val")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    # Model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=CFG["momentum"], weight_decay=CFG["weight_decay"]
    )

    # LR scheduler — reduce at epoch 8 and 11
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=CFG["lr_milestones"], gamma=CFG["lr_gamma"]
    )

    # Training
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    best_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")

        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        lr_scheduler.step()

        print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = WORK_DIR / f"epoch_{epoch}.pth"
        torch.save({
            "epoch"     : epoch,
            "model"     : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "loss"      : avg_loss
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Keep best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), WORK_DIR / "best_model.pth")
            print(f"Best model updated! Loss: {best_loss:.4f}")

    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")
    print(f"Models saved to: {WORK_DIR}")

if __name__ == "__main__":
    main()