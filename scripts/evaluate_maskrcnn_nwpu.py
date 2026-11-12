# Origin: NWPU mask_rcnn/evaluate_nwpu_maskrcnn.py. Hard-coded paths replaced by --root; logic unchanged.
print("Starting Mask R-CNN evaluation on NWPU...")

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


from common import resolve_root
ROOT = resolve_root("NWPU_ROOT")

ROOT_DIR = ROOT
NWPU_DIR = ROOT_DIR / "dataset"
IMG_DIR = NWPU_DIR / "positive image set"
ANN_FILE = NWPU_DIR / "annotations.json"
DEFAULT_SPLIT_FILE = ROOT_DIR / "yolo_dataset" / "train_val_split.json"

DEFAULT_TRAIN_WORK_DIR = ROOT_DIR / "work_dirs" / "paper" / "mask_rcnn" / "maskrcnn_nwpu"
DEFAULT_MODEL_PATH = DEFAULT_TRAIN_WORK_DIR / "best_model.pth"

NUM_CLASSES = 11
SEED = 42

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
    10: "vehicle",
}


class NWPUDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_ids):
        self.img_dir = Path(img_dir)
        with open(ann_file, "r") as f:
            data = json.load(f)

        self.images = [
            img for img in data["images"]
            if img["id"] in img_ids
        ]
        self.img2anns = {}
        for ann in data["annotations"]:
            self.img2anns.setdefault(ann["image_id"], []).append(ann)

        print(f"Val dataset: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        img_path = self.img_dir / filename
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        anns = self.img2anns.get(img_id, [])
        anns = [ann for ann in anns if not ann.get("iscrowd", 0)]

        boxes = []
        labels = []
        masks = []
        areas = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])
            areas.append(ann["area"])

            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            segmentation = ann["segmentation"]
            if isinstance(segmentation, list):
                for poly in segmentation:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

        if not boxes:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, img_h, img_w), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    return model


def load_eval_ids(split_file, split_key):
    """Return the set of image ids for `split_key` ('val' or 'test') from split_file.

    Only falls back to a freshly generated 80/20 split (seed 42) when split_file
    is the default DEFAULT_SPLIT_FILE and missing (matching the behavior of
    train_maskrcnn_nwpu.py). A missing non-default split file is always an
    error.
    """
    if split_file.exists():
        with open(split_file, "r") as f:
            split = json.load(f)
        if split_key not in split:
            raise KeyError(f"{split_file} has no '{split_key}' key (keys: {list(split)})")
        return set(split[split_key])

    if split_file != DEFAULT_SPLIT_FILE:
        raise FileNotFoundError(f"Split file not found: {split_file}")

    if split_key != "val":
        raise FileNotFoundError(
            f"{DEFAULT_SPLIT_FILE} not found and there is no fallback generator for "
            f"split_key={split_key!r} (only the legacy 'val' fallback is supported)."
        )
    with open(ANN_FILE, "r") as f:
        data = json.load(f)
    ids = [img["id"] for img in data["images"]]
    random.seed(SEED)
    random.shuffle(ids)
    split_idx = int(len(ids) * 0.8)
    return set(ids[split_idx:])


def build_val_annotations(val_ids):
    with open(ANN_FILE, "r") as f:
        full_data = json.load(f)

    val_data = {
        "info": full_data.get("info", {}),
        "licenses": full_data.get("licenses", []),
        "categories": full_data["categories"],
        "images": [
            img for img in full_data["images"]
            if img["id"] in val_ids
        ],
        "annotations": [
            ann for ann in full_data["annotations"]
            if ann["image_id"] in val_ids
        ],
    }
    with open(VAL_ANN_FILE, "w") as f:
        json.dump(val_data, f)


def run_inference(model, val_loader, device, score_threshold):
    results = []
    print(f"\nRunning inference on {len(val_loader)} images...")

    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            if (idx + 1) % 20 == 0:
                print(f"  {idx + 1}/{len(val_loader)}")

            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                img_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks = output["masks"].cpu().numpy()

                for box, score, label, mask in zip(
                    boxes, scores, labels, masks
                ):
                    if score < score_threshold or label == 0:
                        continue

                    x1, y1, x2, y2 = box
                    binary = (mask[0] > 0.5).astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(binary))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    results.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2 - x1),
                            float(y2 - y1),
                        ],
                        "score": float(score),
                        "segmentation": rle,
                    })

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)

    print(f"Total predictions: {len(results)}")
    print(f"Predictions saved: {RESULTS_FILE}")


def evaluate_overall(coco_gt, coco_dt, iou_type):
    print(f"\n--- {iou_type.upper()} Evaluation ---")
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "mAP50_95": float(evaluator.stats[0] * 100),
        "mAP50": float(evaluator.stats[1] * 100),
        "mAP75": float(evaluator.stats[2] * 100),
    }


def evaluate_per_class(coco_gt, coco_dt):
    print("\n=== Per-Class AP50 ===")
    rows = []

    for cat_id, cat_name in CATEGORIES.items():
        row = {"Category": cat_name}
        for iou_type, column in (("segm", "Mask_AP50"), ("bbox", "Box_AP50")):
            evaluator = COCOeval(coco_gt, coco_dt, iou_type)
            evaluator.params.catIds = [cat_id]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            row[column] = round(float(evaluator.stats[1] * 100), 2)

        print(
            f"  {cat_name:<20} "
            f"Mask: {row['Mask_AP50']:>6.2f}%  "
            f"Box: {row['Box_AP50']:>6.2f}%"
        )
        rows.append(row)

    return rows


def save_reports(mask_metrics, box_metrics, per_class_rows, split):
    summary = {
        "mask_mAP50": round(mask_metrics["mAP50"], 2),
        "mask_mAP50_95": round(mask_metrics["mAP50_95"], 2),
        "box_mAP50": round(box_metrics["mAP50"], 2),
        "box_mAP50_95": round(box_metrics["mAP50_95"], 2),
        "predictions_file": str(RESULTS_FILE),
        "val_annotations_file": str(VAL_ANN_FILE),
        "model_path": str(MODEL_PATH),
        "split": split,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_summary = pd.DataFrame([{
        "Category": "ALL",
        "Mask_AP50": summary["mask_mAP50"],
        "Box_AP50": summary["box_mAP50"],
    }])
    per_class_df = pd.concat(
        [per_class_df, per_class_summary],
        ignore_index=True,
    )

    overall_df = pd.DataFrame([
        {"Metric": "Mask mAP50", "Value (%)": summary["mask_mAP50"]},
        {"Metric": "Mask mAP50-95", "Value (%)": summary["mask_mAP50_95"]},
        {"Metric": "Box mAP50", "Value (%)": summary["box_mAP50"]},
        {"Metric": "Box mAP50-95", "Value (%)": summary["box_mAP50_95"]},
    ])

    paths_df = pd.DataFrame([
        {"Item": "Model", "Path": summary["model_path"]},
        {"Item": "Predictions", "Path": summary["predictions_file"]},
        {"Item": "Validation annotations", "Path": summary["val_annotations_file"]},
    ])

    with pd.ExcelWriter(EXCEL_FILE) as writer:
        overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
        per_class_df.to_excel(writer, sheet_name="Per-Class AP50", index=False)
        paths_df.to_excel(writer, sheet_name="Files", index=False)

    print("\n=== Mask R-CNN NWPU Results ===")
    print(f"Mask mAP50    : {summary['mask_mAP50']:.2f}%")
    print(f"Mask mAP50-95 : {summary['mask_mAP50_95']:.2f}%")
    print(f"Box  mAP50    : {summary['box_mAP50']:.2f}%")
    print(f"Box  mAP50-95 : {summary['box_mAP50_95']:.2f}%")
    print(f"\nSaved Excel : {EXCEL_FILE}")
    print(f"Saved JSON  : {SUMMARY_JSON}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Mask R-CNN on NWPU VHR-10."
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Regenerate predictions even if predictions.json exists.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Minimum prediction score kept for COCO evaluation.",
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which split to evaluate against.",
    )
    parser.add_argument(
        "--split-file", type=Path, default=DEFAULT_SPLIT_FILE,
        help="train_val_split.json-style file with train/val/test id lists "
             "(default: the 80:20 split, seed 42).",
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help="Checkpoint to evaluate, e.g. work_dirs/paper/mask_rcnn/maskrcnn_nwpu/best_model.pth",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    # Default args reproduce the old work_dirs/maskrcnn_nwpu_evaluation layout
    # exactly; any non-default split/model-path/split-file gets its own
    # folder so it never clobbers an existing evaluation.
    if model_path == DEFAULT_MODEL_PATH and args.split == "val":
        eval_dir_name = "maskrcnn_nwpu_evaluation"
        eval_work_dir = ROOT_DIR / "work_dirs" / "paper" / "mask_rcnn" / eval_dir_name
    else:
        run_name = model_path.parent.name
        eval_dir_name = f"{run_name}_evaluation_{args.split}"
        eval_work_dir = ROOT_DIR / "work_dirs" / eval_dir_name
    output_dir = eval_work_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    global OUTPUT_DIR, MODEL_PATH, RESULTS_FILE, VAL_ANN_FILE, SUMMARY_JSON, EXCEL_FILE
    OUTPUT_DIR = output_dir
    MODEL_PATH = model_path
    RESULTS_FILE = OUTPUT_DIR / "predictions.json"
    VAL_ANN_FILE = OUTPUT_DIR / "val_anns.json"
    SUMMARY_JSON = OUTPUT_DIR / "summary_metrics.json"
    EXCEL_FILE = OUTPUT_DIR / "maskrcnn_nwpu_evaluation.xlsx"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not ANN_FILE.exists():
        raise FileNotFoundError(f"Annotation file not found: {ANN_FILE}")
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Model : {MODEL_PATH}")
    print(f"Split : {args.split}  (from {args.split_file})")
    print(f"Output: {OUTPUT_DIR}")

    eval_ids = load_eval_ids(args.split_file, args.split)
    print(f"{args.split} images: {len(eval_ids)}")
    build_val_annotations(eval_ids)

    needs_inference = args.force_inference or not RESULTS_FILE.exists()
    if needs_inference:
        eval_dataset = NWPUDataset(IMG_DIR, ANN_FILE, eval_ids)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        print("\nLoading model weights...")
        model = get_model(NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")

        run_inference(model, eval_loader, device, args.score_threshold)
    else:
        print(f"\nUsing existing predictions: {RESULTS_FILE}")

    print("\n=== COCO Evaluation ===")
    coco_gt = COCO(str(VAL_ANN_FILE))
    coco_dt = coco_gt.loadRes(str(RESULTS_FILE))

    mask_metrics = evaluate_overall(coco_gt, coco_dt, "segm")
    box_metrics = evaluate_overall(coco_gt, coco_dt, "bbox")
    per_class_rows = evaluate_per_class(coco_gt, coco_dt)
    save_reports(mask_metrics, box_metrics, per_class_rows, args.split)


if __name__ == "__main__":
    main()
