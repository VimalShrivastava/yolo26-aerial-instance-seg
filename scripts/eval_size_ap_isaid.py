# Origin: iSAID evaluation/compute_size_ap_isaid.py (the script behind the paper's iSAID tables). Hard-coded paths replaced by --root; logic unchanged.
"""
COCO-style size-bucketed AP (APs / APm / APl, box + mask) for Mask R-CNN,
YOLO11m, and YOLO26m on the iSAID validation set (yolo26_dataset/images/val +
yolo26_dataset/labels/val, 5541 images, YOLO-seg polygon labels).

Why this script exists (see investigation notes below for the full trail):

  * Mask R-CNN's own evaluate.py (Codes/mask_rcnn/evaluate.py) scores against
    Dataset_patches/val (9581 images, includes 4040 background-only patches)
    using an in-memory pycocotools COCOeval -- it never writes a
    predictions.json to disk, and its GT file (iSAID_val_patches.json) uses
    sequential integer image ids (1..9581) that have nothing to do with
    filenames.
  * YOLO11m/YOLO26m's evaluate scripts (Codes/yolo11/codes/evaluate_yolo11.py,
    Codes/yolo26/Codes/evaluate_yolo26.py) call Ultralytics' model.val() with
    save_json=True on yolo26_dataset directly. Their predictions.json files
    (work_dirs/paper/yolo11/yolo11m_isaid/eval_results/predictions.json and
    work_dirs/paper/yolo26/yolo26m_isaid_musgd/eval_results/predictions.json)
    already use a low score threshold (min score observed = 0.001, i.e. a
    full precision-recall sweep is usable as-is) -- but their "image_id" is
    the filename STEM AS A STRING (e.g. "P0003_0_0_800_800"), not
    int(filename stem): iSAID patch filenames aren't numeric, so the
    int-id assumption that holds on the sibling NWPU project does not carry
    over here. This was verified by inspection, not assumed.
  * Mask R-CNN's GT image ids (int, sequential, full 9581-image set) do NOT
    match YOLO's image ids (str, filename stem, non-empty 5541-image subset)
    -- reusing one as GT for the other would silently score against the
    wrong images/ids, exactly the failure mode this script is written to
    avoid. And since Mask R-CNN has no existing predictions.json at all
    (its evaluate.py never saves one) there is nothing to remap anyway.
  * Conclusion: build ONE unified COCO GT directly from
    yolo26_dataset/labels/val, with image_id = filename stem (string,
    matching what the YOLO predictions.json files already use). Reuse
    YOLO11m/YOLO26m's existing low-threshold predictions.json as-is. Run
    fresh Mask R-CNN inference restricted to the same 5541 images (this is
    unavoidable -- nothing reusable exists for it at the right threshold or
    the right image set), writing predictions with the same string
    image_id scheme so all three models score against literally the same
    GT file.

Category order: reuses CLASS_NAMES from
Codes/visualization/create_isaid_challenge_category_figure.py (0-indexed,
matches yolo26_dataset/isaid.yaml's `names:` order). Independently
re-verified against Mask R-CNN's trained category order by cross-referencing
Dataset_patches/{train,val}/annotations/iSAID_*_patches.json's `categories`
list (sorted by id) -- Codes/mask_rcnn/train_maskrcnn.py builds its label
mapping as `{cid: idx+1 for idx, cid in enumerate(sorted(cat_ids))}` directly
from that same list, so category_id N in the checkpoint's output head
corresponds to CLASS_NAMES[N-1]. The two lists match name-for-name (only
capitalization differs, e.g. "Large_Vehicle" vs "large_vehicle").

Usage (mask_rcnn conda env has torch+torchvision+ultralytics+pycocotools):

    python scripts/eval_size_ap_isaid.py --root <iSAID root>

Re-run flags:
    --force-rebuild-gt          rebuild the unified GT json even if cached
    --force-maskrcnn-inference  re-run Mask R-CNN inference even if a cached
                                 predictions file already exists
    --small-min/--small-max/--medium-max   override the custom area buckets
                                 (default small=[10,144), medium=[144,1024),
                                 large=[1024,inf), px^2 of RASTERIZED mask
                                 area, not bbox area)

The script never overwrites an existing CSV: reruns save as ..._v2.csv, etc.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# =========================================================
# PATHS
# =========================================================

from common import resolve_root
ROOT = resolve_root("ISAID_ROOT")

ROOT_DIR = ROOT

VAL_IMAGE_DIR = ROOT_DIR / "yolo26_dataset" / "images" / "val"
VAL_LABEL_DIR = ROOT_DIR / "yolo26_dataset" / "labels" / "val"

MASKRCNN_CKPT = ROOT_DIR / "work_dirs" / "paper" / "maskrcnn_torchvision" / "best_model.pth"
YOLO11_PRED_JSON = ROOT_DIR / "work_dirs" / "paper" / "yolo11" / "yolo11m_isaid" / "eval_results" / "predictions.json"
YOLO26_PRED_JSON = ROOT_DIR / "work_dirs" / "paper" / "yolo26" / "yolo26m_isaid_musgd" / "eval_results" / "predictions.json"

# "already reported" mAP baselines used for the cross-check section. No
# summary_metrics.json exists in this repo; the closest thing is the final
# row of each Ultralytics training run's results.csv (Ultralytics' own
# metric object, NOT pycocotools) and Mask R-CNN's evaluation_results.xlsx
# (pycocotools, but scored on the full 9581-image Dataset_patches/val,
# COCO-default area thresholds, not the 5541-image yolo26_dataset subset).
YOLO11_RESULTS_CSV = ROOT_DIR / "work_dirs" / "paper" / "yolo11" / "yolo11m_isaid" / "results.csv"
YOLO26_RESULTS_CSV = ROOT_DIR / "work_dirs" / "paper" / "yolo26" / "yolo26m_isaid_musgd" / "results.csv"

OUT_DIR = ROOT_DIR / "work_dirs" / "paper" / "size_ap_eval"
GT_JSON_PATH = OUT_DIR / "isaid_val_unified_gt.json"
MASKRCNN_PRED_PATH = OUT_DIR / "maskrcnn_predictions_val.json"
CSV_OUT_PATH = OUT_DIR / "size_ap_isaid_summary.csv"
CLASSWISE_CSV_OUT_PATH = OUT_DIR / "size_ap_isaid_classwise.csv"

# =========================================================
# CLASSES (reused verbatim from create_isaid_challenge_category_figure.py,
# independently re-checked above against the Mask R-CNN checkpoint's
# trained category order)
# =========================================================

CLASS_NAMES = [
    "storage_tank",
    "large_vehicle",
    "small_vehicle",
    "plane",
    "ship",
    "swimming_pool",
    "harbor",
    "tennis_court",
    "ground_track_field",
    "soccer_ball_field",
    "baseball_diamond",
    "bridge",
    "basketball_court",
    "roundabout",
    "helicopter",
]
NUM_CLASSES = len(CLASS_NAMES) + 1  # +1 background, matches Mask R-CNN's trained head


# =========================================================
# UNIFIED GT CONSTRUCTION
# =========================================================

def build_unified_gt() -> dict:
    """Build one COCO-format GT dict directly from yolo26_dataset/labels/val.

    image_id = filename stem (string) -- matches what YOLO11m/YOLO26m's own
    saved predictions.json already use (verified by inspection, see module
    docstring), NOT int(stem): iSAID patch filenames like
    "P0003_0_0_800_800" aren't numeric.

    area = rasterized mask pixel area via pycocotools mask.area(), computed
    from the SAME rasterized RLE as bbox (via mask.toBbox()), so area and
    bbox are mutually consistent for every instance.
    """
    label_files = sorted(VAL_LABEL_DIR.glob("*.txt"))
    images, annotations = [], []
    ann_id = 1
    skipped_degenerate = 0
    skipped_missing_image = 0

    for label_path in label_files:
        stem = label_path.stem
        img_path = VAL_IMAGE_DIR / f"{stem}.png"
        if not img_path.exists():
            skipped_missing_image += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        images.append({"id": stem, "file_name": f"{stem}.png", "width": w, "height": h})

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 7:  # cls + at least 3 (x,y) pairs
                continue
            cls_id = int(float(parts[0]))
            coords = np.array(parts[1:], dtype=np.float64)
            xs = np.clip(coords[0::2] * w, 0, w)
            ys = np.clip(coords[1::2] * h, 0, h)
            if xs.size < 3:
                skipped_degenerate += 1
                continue
            polygon = np.stack([xs, ys], axis=1).flatten().tolist()

            rle = maskUtils.merge(maskUtils.frPyObjects([polygon], h, w))
            area = float(maskUtils.area(rle))
            if area <= 0:
                skipped_degenerate += 1
                continue
            bbox = [float(v) for v in maskUtils.toBbox(rle)]

            annotations.append({
                "id": ann_id,
                "image_id": stem,
                "category_id": cls_id + 1,
                "segmentation": [polygon],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

    if skipped_missing_image:
        print(f"  WARNING: {skipped_missing_image} label file(s) had no matching image, skipped")
    if skipped_degenerate:
        print(f"  WARNING: {skipped_degenerate} degenerate/zero-area polygon instance(s) skipped")

    categories = [{"id": i + 1, "name": CLASS_NAMES[i]} for i in range(len(CLASS_NAMES))]
    return {"images": images, "annotations": annotations, "categories": categories}


def load_or_build_gt(force_rebuild: bool) -> dict:
    if GT_JSON_PATH.exists() and not force_rebuild:
        print(f"Loading cached unified GT: {GT_JSON_PATH}")
        with open(GT_JSON_PATH) as f:
            return json.load(f)

    print("Building unified COCO-format GT from yolo26_dataset/labels/val ...")
    t0 = time.time()
    gt = build_unified_gt()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(GT_JSON_PATH, "w") as f:
        json.dump(gt, f)
    print(f"  {len(gt['images'])} images, {len(gt['annotations'])} instances "
          f"({time.time()-t0:.1f}s) -> {GT_JSON_PATH}")
    return gt


# =========================================================
# GT AREA DISTRIBUTION REPORT
# =========================================================

def print_area_distribution(gt: dict, custom_buckets, default_buckets) -> None:
    areas = np.array([a["area"] for a in gt["annotations"]], dtype=np.float64)
    n = len(areas)
    print(f"\nGT mask-area distribution (n={n} instances, px^2, rasterized mask area):")
    print(f"  min={areas.min():.1f}  median={np.median(areas):.1f}  max={areas.max():.1f}  mean={areas.mean():.1f}")

    for label, buckets in [("custom", custom_buckets), ("coco-default", default_buckets)]:
        print(f"  -- {label} thresholds --")
        covered = np.zeros(n, dtype=bool)
        for name, lo, hi in buckets:
            in_bucket = (areas >= lo) & (areas < hi)
            covered |= in_bucket
            cnt = int(in_bucket.sum())
            pct = 100.0 * cnt / n if n else 0.0
            flag = "  <-- EMPTY BUCKET (AP will be -1.0 / N/A)" if cnt == 0 else ""
            print(f"    {name:8s} [{lo:>10.1f}, {hi:>10.1f}) : {cnt:7d} ({pct:5.2f}%){flag}")
        uncovered = int((~covered).sum())
        if uncovered:
            pct = 100.0 * uncovered / n
            print(f"    (outside all three buckets, e.g. area < small-min): {uncovered} ({pct:.2f}%)")


# =========================================================
# MASK R-CNN: FRESH INFERENCE ON THE SAME 5541-IMAGE VAL SET
# =========================================================

def get_maskrcnn_model(num_classes: int) -> torch.nn.Module:
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


class ValImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: Path):
        self.files = sorted(image_dir.glob("*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        tensor = torchvision.transforms.functional.to_tensor(img)
        return tensor, path.stem


def _collate(batch):
    return tuple(zip(*batch))


def run_maskrcnn_inference(conf_thresh: float, device: torch.device, batch_size: int, workers: int):
    """Run Mask R-CNN on yolo26_dataset/images/val (NOT Dataset_patches --
    that's a different, larger image set including background-only patches)
    and dump low-threshold COCO-format predictions, image_id = filename stem
    (string), so they score against the same unified GT as YOLO11m/YOLO26m.

    Raw predictions are kept as-is: no filtering to GT-present classes, so
    wrong-class hallucinations count against Mask R-CNN exactly like they
    would for the YOLO models.
    """
    print(f"Running fresh Mask R-CNN inference on {VAL_IMAGE_DIR} (conf>={conf_thresh}) ...")
    model = get_maskrcnn_model(NUM_CLASSES)
    checkpoint = torch.load(MASKRCNN_CKPT, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # torchvision's RoIHeads internally filters detections below
    # score_thresh=0.05 by default, BEFORE they ever reach our score check
    # below -- AP needs the full precision-recall sweep, so this must be
    # lowered to conf_thresh here, not just filtered downstream.
    model.roi_heads.score_thresh = conf_thresh

    dataset = ValImageDataset(VAL_IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=workers, collate_fn=_collate)

    results_bbox, results_segm = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (images, stems) in enumerate(loader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, stem in zip(outputs, stems):
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks = output["masks"].cpu().numpy()
                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    if score < conf_thresh:
                        continue
                    cat_id = int(label)  # identity mapping: trained label 1..15 == category_id 1..15
                    x1, y1, x2, y2 = [float(v) for v in box]
                    results_bbox.append({
                        "image_id": stem, "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
                    })
                    binary_mask = (mask[0] > 0.5)
                    results_segm.append({
                        "image_id": stem, "category_id": cat_id,
                        "segmentation": mask_to_rle(binary_mask), "score": float(score),
                    })
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(loader)}] {time.time()-t0:.1f}s elapsed")

    print(f"  done in {time.time()-t0:.1f}s: {len(results_bbox)} bbox dets, {len(results_segm)} segm dets")
    return results_bbox, results_segm


def load_or_run_maskrcnn(conf_thresh: float, device: torch.device, batch_size: int,
                          workers: int, force: bool):
    if MASKRCNN_PRED_PATH.exists() and not force:
        print(f"Loading cached Mask R-CNN predictions: {MASKRCNN_PRED_PATH}")
        with open(MASKRCNN_PRED_PATH) as f:
            cached = json.load(f)
        return cached["bbox"], cached["segm"]

    results_bbox, results_segm = run_maskrcnn_inference(conf_thresh, device, batch_size, workers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MASKRCNN_PRED_PATH, "w") as f:
        json.dump({"bbox": results_bbox, "segm": results_segm}, f)
    print(f"  cached -> {MASKRCNN_PRED_PATH}")
    return results_bbox, results_segm


# =========================================================
# YOLO PREDICTIONS (REUSED AS-IS, ALREADY LOW-THRESHOLD)
# =========================================================

def load_yolo_predictions(path: Path, valid_image_ids: set):
    with open(path) as f:
        preds = json.load(f)
    scores = [p["score"] for p in preds]
    min_score = min(scores) if scores else float("nan")
    print(f"  {path.name}: {len(preds)} dets, min score={min_score:.4f}, "
          f"{len(set(p['image_id'] for p in preds))} unique image_ids")
    if min_score > 0.05:
        print(f"  WARNING: min score {min_score:.4f} looks like a display-oriented threshold, "
              f"not a full PR sweep -- AP numbers from this file may be truncated low.")
    # bbox and segm entries are identical dicts here except which fields are
    # populated; COCOeval only reads the fields relevant to its iouType.
    preds = [p for p in preds if p["image_id"] in valid_image_ids]
    return preds, preds


# =========================================================
# COCOEVAL RUNNER
# =========================================================

# pycocotools' COCO-default maxDets=[1,10,100] silently truncates detections
# past rank-100 per image before matching. iSAID val patches are far denser
# than COCO images: 159/5541 patches have >100 GT instances (up to 613 in
# one 800x800 patch). Ultralytics' own NMS keeps up to max_det=300/image and
# its own val() AP computation applies no per-image cap at all -- so at the
# pycocotools default, genuine true positives beyond rank-100 in those dense
# patches were being scored as missed, which is why the recomputed AP came
# out ~3 points below Ultralytics' reported number on every metric,
# uniformly (verified: raising maxDets to 300 closes most of that gap).
# Match Ultralytics' cap instead of the COCO-photo-scene default.
MAX_DETS = [1, 10, 300]


def _run_cocoeval_on_dt(coco_gt: COCO, coco_dt, iou_type: str, area_rng, area_lbl, cat_ids=None):
    E = COCOeval(coco_gt, coco_dt, iou_type)
    E.params.areaRng = area_rng
    E.params.areaRngLbl = area_lbl
    E.params.maxDets = MAX_DETS
    if cat_ids is not None:
        E.params.catIds = cat_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    # pycocotools bug (see cocoeval.py's own docstring: "this function can
    # *only* be applied on the default parameter setting"): summarize()'s
    # headline stats[0] (AP @ IoU=0.50:0.95 | area=all) calls its internal
    # _summarize() with the hardcoded DEFAULT maxDets=100, not
    # self.params.maxDets[2] like every other stat does. Since 100 is no
    # longer in MAX_DETS, that lookup comes back empty and stats[0] silently
    # becomes -1. Recompute it directly from the precision array using the
    # maxDets we actually configured, the same way stats[1] (AP50) etc.
    # already correctly do internally.
    aind = [i for i, a in enumerate(E.params.areaRngLbl) if a == "all"]
    mind = [i for i, m in enumerate(E.params.maxDets) if m == MAX_DETS[2]]
    prec = E.eval["precision"][:, :, :, aind, mind]
    E.stats[0] = float(np.mean(prec[prec > -1])) if np.any(prec > -1) else -1.0
    return E.stats


def run_cocoeval(coco_gt: COCO, dt_list: list, iou_type: str, area_rng, area_lbl):
    if not dt_list:
        return np.full(12, -1.0)
    coco_dt = coco_gt.loadRes(copy.deepcopy(dt_list))
    return _run_cocoeval_on_dt(coco_gt, coco_dt, iou_type, area_rng, area_lbl)


# =========================================================
# PER-CLASS AP (whole-image 'all' area range, one COCOeval per category so a
# category with zero predictions/GT in a class doesn't skew the others --
# same MAX_DETS=300 fix as the main sweep, for consistency)
# =========================================================

def run_classwise_eval(coco_gt: COCO, models: dict, cat_id_to_name: dict) -> list[dict]:
    area_rng, area_lbl = [[0, 1e10]], ["all"]
    rows = []
    for model_name, preds in models.items():
        for iou_type in ["bbox", "segm"]:
            dt_list = preds[iou_type]
            if not dt_list:
                for cat_id, cat_name in sorted(cat_id_to_name.items()):
                    rows.append({"model": model_name, "iou_type": iou_type, "category": cat_name,
                                 "AP": -1.0, "AP50": -1.0, "AP75": -1.0})
                continue
            coco_dt = coco_gt.loadRes(copy.deepcopy(dt_list))
            print(f"  {model_name} / {iou_type} ...")
            for cat_id, cat_name in sorted(cat_id_to_name.items()):
                with contextlib.redirect_stdout(io.StringIO()):
                    stats = _run_cocoeval_on_dt(coco_gt, coco_dt, iou_type, area_rng, area_lbl, cat_ids=[cat_id])
                rows.append({"model": model_name, "iou_type": iou_type, "category": cat_name,
                             "AP": stats[0], "AP50": stats[1], "AP75": stats[2]})
    return rows


def print_classwise_table(rows: list) -> None:
    print(f"\n{'='*100}")
    print("Per-class AP -- pycocotools, maxDets=300, area='all'")
    print(f"{'='*100}")
    by_model_type = {}
    for r in rows:
        by_model_type.setdefault((r["model"], r["iou_type"]), []).append(r)
    for (model_name, iou_type), rs in by_model_type.items():
        print(f"\n{model_name} / {iou_type}")
        print(f"  {'Category':<22}{'AP':>8}{'AP50':>8}{'AP75':>8}")
        for r in rs:
            print(f"  {r['category']:<22}{fmt_stat(r['AP']):>8}{fmt_stat(r['AP50']):>8}{fmt_stat(r['AP75']):>8}")
        valid = [r["AP"] for r in rs if r["AP"] >= 0]
        if valid:
            print(f"  {'Mean':<22}{100*sum(valid)/len(valid):>7.2f}%")


def fmt_stat(v: float) -> str:
    return "N/A" if v is None or v < 0 else f"{v*100:.2f}"


# =========================================================
# OUTPUT TABLE / CSV
# =========================================================

# AR100 -> AR300: pycocotools' fixed stat-index layout is unchanged (index 8
# is still "AR @ maxDets[2]"), only what maxDets[2] itself means changed
# (100 -> 300, see MAX_DETS above), so the label is renamed to match.
STAT_COLS = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR300", "ARs", "ARm", "ARl"]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def print_table(rows: list, scheme_label: str):
    print(f"\n{'='*100}")
    print(f"Size-bucketed AP -- {scheme_label} area thresholds")
    print(f"{'='*100}")
    header = f"{'Model':<12}{'Type':<6}" + "".join(f"{c:>8}" for c in STAT_COLS)
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = "".join(f"{fmt_stat(row[c]):>8}" for c in STAT_COLS)
        print(f"{row['model']:<12}{row['iou_type']:<6}{vals}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small-min", type=float, default=10.0)
    parser.add_argument("--small-max", type=float, default=144.0)
    parser.add_argument("--medium-max", type=float, default=1024.0)
    parser.add_argument("--conf-thresh", type=float, default=0.001,
                         help="Score threshold for fresh Mask R-CNN inference (full PR sweep).")
    parser.add_argument("--maskrcnn-batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force-rebuild-gt", action="store_true")
    parser.add_argument("--force-maskrcnn-inference", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Unified GT ----
    gt = load_or_build_gt(args.force_rebuild_gt)
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    valid_image_ids = set(im["id"] for im in gt["images"])

    inf = 1e10
    custom_buckets = [
        ("small", args.small_min, args.small_max),
        ("medium", args.small_max, args.medium_max),
        ("large", args.medium_max, inf),
    ]
    default_buckets = [("small", 0.0, 1024.0), ("medium", 1024.0, 9216.0), ("large", 9216.0, inf)]
    print_area_distribution(gt, custom_buckets, default_buckets)

    area_schemes = {
        "custom": ([[0, inf]] + [[lo, hi] for _, lo, hi in custom_buckets], ["all", "small", "medium", "large"]),
        "coco-default": ([[0, inf], [0, 1024], [1024, 9216], [9216, inf]], ["all", "small", "medium", "large"]),
    }

    # ---- 2. Predictions per model ----
    print("\nLoading/generating predictions ...")
    print("YOLO11m:")
    y11_bbox, y11_segm = load_yolo_predictions(YOLO11_PRED_JSON, valid_image_ids)
    print("YOLO26m:")
    y26_bbox, y26_segm = load_yolo_predictions(YOLO26_PRED_JSON, valid_image_ids)
    print("Mask R-CNN:")
    mr_bbox, mr_segm = load_or_run_maskrcnn(args.conf_thresh, device, args.maskrcnn_batch,
                                             args.workers, args.force_maskrcnn_inference)

    models = {
        "MaskRCNN": {"bbox": mr_bbox, "segm": mr_segm},
        "YOLO11m": {"bbox": y11_bbox, "segm": y11_segm},
        "YOLO26m": {"bbox": y26_bbox, "segm": y26_segm},
    }

    # ---- 3. COCOeval sweep ----
    csv_rows = []
    tables = {}
    for scheme_name, (area_rng, area_lbl) in area_schemes.items():
        rows = []
        for model_name, preds in models.items():
            for iou_type in ["bbox", "segm"]:
                print(f"\n--- {model_name} / {iou_type} / {scheme_name} ---")
                stats = run_cocoeval(coco_gt, preds[iou_type], iou_type, area_rng, area_lbl)
                row = {"model": model_name, "iou_type": iou_type}
                row.update({c: stats[i] for i, c in enumerate(STAT_COLS)})
                rows.append(row)
                csv_rows.append({"scheme": scheme_name, **row})
        tables[scheme_name] = rows
        print_table(rows, scheme_name)

    # ---- 4. Cross-check against previously reported numbers ----
    print(f"\n{'='*100}")
    print("Cross-check: recomputed 'all'-bucket AP50 / AP50-95 vs previously reported")
    print(f"{'='*100}")

    def csv_last_row_metrics(csv_path: Path):
        if not csv_path.exists():
            return None
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        last = rows[-1]
        return {
            "box_map50": float(last["metrics/mAP50(B)"]),
            "box_map": float(last["metrics/mAP50-95(B)"]),
            "mask_map50": float(last["metrics/mAP50(M)"]),
            "mask_map": float(last["metrics/mAP50-95(M)"]),
        }

    custom_rows_by_key = {(r["model"], r["iou_type"]): r for r in tables["custom"]}

    def report_crosscheck(model_name, iou_type, ours_ap50, ours_ap, reported_ap50, reported_ap, source, same_tool):
        d50 = ours_ap50 - reported_ap50
        d = ours_ap - reported_ap
        tag = "(same tool: pycocotools both sides, expect near-exact match)" if same_tool else \
              "(different tool: Ultralytics' own val() vs pycocotools -- some gap is expected/documented, not a bug)"
        print(f"{model_name:<10} {iou_type:<5} vs {source:<28} "
              f"AP50: ours={ours_ap50*100:6.2f} reported={reported_ap50*100:6.2f} diff={d50*100:+6.2f}   "
              f"AP50-95: ours={ours_ap*100:6.2f} reported={reported_ap*100:6.2f} diff={d*100:+6.2f}  {tag}")

    # Mask R-CNN: compare against evaluation_results.xlsx if present (same
    # tool, but different image scope: full 9581-image Dataset_patches/val
    # incl. background-only patches, vs our 5541-image non-empty subset, and
    # COCO-default area thresholds there vs our custom 'all' bucket -- 'all'
    # itself is threshold-independent so that part is comparable).
    mr_xlsx = ROOT_DIR / "work_dirs" / "paper" / "maskrcnn_torchvision" / "results" / "evaluation_results.xlsx"
    if mr_xlsx.exists():
        try:
            import openpyxl
            wb = openpyxl.load_workbook(mr_xlsx, data_only=True)
            ws = wb["Overall Results"]
            bbox_reported = ws["B6"].value, ws["C6"].value
            segm_reported = ws["B11"].value, ws["C11"].value
            r = custom_rows_by_key[("MaskRCNN", "bbox")]
            report_crosscheck("MaskRCNN", "bbox", r["AP50"], r["AP"],
                               float(bbox_reported[1]) / 100, float(bbox_reported[0]) / 100,
                               "evaluation_results.xlsx (9581-img scope)", same_tool=True)
            r = custom_rows_by_key[("MaskRCNN", "segm")]
            report_crosscheck("MaskRCNN", "segm", r["AP50"], r["AP"],
                               float(segm_reported[1]) / 100, float(segm_reported[0]) / 100,
                               "evaluation_results.xlsx (9581-img scope)", same_tool=True)
        except Exception as exc:  # pragma: no cover
            print(f"  (could not read {mr_xlsx}: {exc})")
    else:
        print("  No evaluation_results.xlsx found for Mask R-CNN -- skipping its cross-check.")

    for model_name, csv_path in [("YOLO11m", YOLO11_RESULTS_CSV), ("YOLO26m", YOLO26_RESULTS_CSV)]:
        metrics = csv_last_row_metrics(csv_path)
        if metrics is None:
            print(f"  No results.csv found for {model_name} -- skipping its cross-check.")
            continue
        r = custom_rows_by_key[(model_name, "bbox")]
        report_crosscheck(model_name, "bbox", r["AP50"], r["AP"],
                           metrics["box_map50"], metrics["box_map"],
                           f"{csv_path.name} final epoch (Ultralytics val())", same_tool=False)
        r = custom_rows_by_key[(model_name, "segm")]
        report_crosscheck(model_name, "segm", r["AP50"], r["AP"],
                           metrics["mask_map50"], metrics["mask_map"],
                           f"{csv_path.name} final epoch (Ultralytics val())", same_tool=False)

    # ---- 5. CSV output (auto-versioned) ----
    csv_path = unique_path(CSV_OUT_PATH)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scheme", "model", "iou_type"] + STAT_COLS)
        writer.writeheader()
        for row in csv_rows:
            out = {k: (fmt_stat(row[k]) if k in STAT_COLS else row[k]) for k in writer.fieldnames}
            writer.writerow(out)
    print(f"\nCSV written: {csv_path}")

    # ---- 6. Per-class AP (pycocotools, same GT/predictions/MAX_DETS as above) ----
    print(f"\n{'='*100}")
    print("Per-class AP sweep")
    print(f"{'='*100}")
    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    classwise_rows = run_classwise_eval(coco_gt, models, cat_id_to_name)
    print_classwise_table(classwise_rows)

    classwise_csv_path = unique_path(CLASSWISE_CSV_OUT_PATH)
    with open(classwise_csv_path, "w", newline="") as f:
        fieldnames = ["model", "iou_type", "category", "AP", "AP50", "AP75"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in classwise_rows:
            out = {k: (fmt_stat(row[k]) if k in ("AP", "AP50", "AP75") else row[k]) for k in fieldnames}
            writer.writerow(out)
    print(f"\nClasswise CSV written: {classwise_csv_path}")


if __name__ == "__main__":
    main()
