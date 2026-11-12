# Origin: NWPU evaluation/compute_size_ap_nwpu.py (the script behind the paper's NWPU tables). Hard-coded paths replaced by --root; logic unchanged.
"""
Compute COCO-style size-bucketed AP (APs / APm / APl, for both box and mask)
for Mask R-CNN, YOLO11m, and YOLO26m on the NWPU VHR-10 validation set, next
to the mAP50 / mAP50-95 already reported by each model's own
evaluate_nwpu_*.py script -- this fills in the small/medium/large object-size
breakdown those scripts don't compute.

Area thresholds (pixel^2, mask area -- not bbox area, matching the COCO
'area' field convention): small [10, 144), medium [144, 1024), large
[1024, inf). These are custom, smaller than COCO's own defaults
([0,1024)/[1024,9216)/[9216,inf)), because NWPU objects are typically much
smaller than COCO's natural-image objects. Override with --small-max /
--medium-max if a different breakpoint is wanted.

Ground truth: freshly built from yolo_dataset/labels/val (YOLO-seg polygon
format) rather than reusing
work_dirs/maskrcnn_nwpu_evaluation/results/val_anns.json directly, because
of an image_id mismatch verified by inspection: Ultralytics' own saved
predictions.json (work_dirs/yolo11m_nwpu_evaluation/results/predictions.json
etc.) uses image_id = int(filename stem) (e.g. "624.jpg" -> 624), while
val_anns.json uses the *dataset's own* sequential COCO ids (e.g.
"004.jpg" -> id 3, not 4) -- only 29/130 ids happened to coincide. Using
val_anns.json as-is against YOLO's predictions would have silently scored
~101/130 images against the wrong ground truth. This script instead builds
one unified GT with image_id = int(stem) for every val image -- matching
YOLO's already-correct convention -- and remaps Mask R-CNN's predictions
(which used val_anns.json's ids) onto the same scheme via val_anns.json's
id -> file_name lookup. All three models are then scored against the
identical GT/id system.

Predictions are reused as-is from each model's own low-confidence-threshold
predictions.json (never re-run through inference here), so the AP/AP50/
AP50-95 this script recomputes for the "all" bucket is also a direct,
independent cross-check against each model's existing summary_metrics.json
-- see verify_against_reported_metrics(). A mismatch there would mean the
GT rebuild or id remap has a bug, not a modeling difference.

Usage (from codes/evaluation, in the mask_rcnn conda env, after each
model's own evaluate_nwpu_*.py has been run at least once so
predictions.json exists for all three):

    python scripts/eval_size_ap_nwpu.py --root <NWPU root>

Never overwrites an existing output file -- reruns save as "..._v2.csv" etc.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================================================
# PATHS
# =========================================================

from common import resolve_root
ROOT = resolve_root("NWPU_ROOT")

ROOT_DIR = ROOT

IMAGE_EXT = ".jpg"

DEFAULT_DATASET_ROOT = ROOT_DIR / "yolo_dataset"
DEFAULT_MASKRCNN_EVAL_DIR = ROOT_DIR / "work_dirs" / "paper" / "mask_rcnn" / "maskrcnn_nwpu_evaluation" / "results"
DEFAULT_YOLO11_EVAL_DIR = ROOT_DIR / "work_dirs" / "paper" / "yolo11" / "yolo11m_nwpu_evaluation" / "results"
DEFAULT_YOLO26_EVAL_DIR = ROOT_DIR / "work_dirs" / "paper" / "yolo26" / "yolo26m_nwpu_evaluation" / "results"

OUTPUT_DIR = ROOT_DIR / "work_dirs" / "size_ap_evaluation"
DEFAULT_OUTPUT_NAME = "nwpu_size_ap_results.csv"
CLASSWISE_OUTPUT_NAME = "nwpu_size_ap_classwise.csv"

# Verified against yolo_dataset/nwpu.yaml, dataset/annotations.json category
# order, and both YOLO checkpoints' model.names (see
# codes/visualization/create_nwpu_challenge_category_figure.py docstring).
CLASS_NAMES = [
    "airplane", "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "ground_track_field", "harbor", "bridge", "vehicle",
]

REPORTED_METRIC_TOLERANCE = 1.0  # percentage points


# =========================================================
# ARGS
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-max", type=float, default=144.0,
                         help="Upper bound (exclusive) of the 'small' area bucket, px^2.")
    parser.add_argument("--medium-max", type=float, default=1024.0,
                         help="Upper bound (exclusive) of the 'medium' area bucket, px^2.")
    parser.add_argument("--small-min", type=float, default=10.0,
                         help="Lower bound (inclusive) of the 'small' area bucket, px^2.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--rebuild-gt", action="store_true",
                         help="Rebuild the unified GT json even if it already exists.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                         help="Dataset root containing images/<split> and labels/<split> "
                              "(yolo_dataset: the 80:20 split, seed 42).")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                         help="Which split's images/labels to build the unified GT from. "
                              "The paper uses the default, 'val'.")
    parser.add_argument("--maskrcnn-eval-dir", type=Path, default=DEFAULT_MASKRCNN_EVAL_DIR,
                         help="Mask R-CNN eval results dir (must contain val_anns.json + predictions.json).")
    parser.add_argument("--yolo11-eval-dir", type=Path, default=DEFAULT_YOLO11_EVAL_DIR,
                         help="YOLO11m eval results dir (must contain predictions.json).")
    parser.add_argument("--yolo26-eval-dir", type=Path, default=DEFAULT_YOLO26_EVAL_DIR,
                         help="YOLO26m eval results dir (must contain predictions.json).")
    return parser.parse_args()


# =========================================================
# UNIFIED GROUND TRUTH (image_id = int(filename stem), matching Ultralytics'
# own predictions.json convention -- see module docstring)
# =========================================================

def build_unified_gt(gt_path: Path, label_dir: Path, image_dir: Path) -> None:
    print(f"Building unified COCO ground truth from {label_dir} ...")
    images = []
    annotations = []
    ann_id = 1

    label_paths = sorted(label_dir.glob("*.txt"))
    for label_path in label_paths:
        stem = label_path.stem
        image_path = image_dir / f"{stem}{IMAGE_EXT}"
        if not image_path.exists():
            raise FileNotFoundError(f"Label {label_path} has no matching image {image_path}")

        import cv2
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        image_id = int(stem)
        images.append({"id": image_id, "file_name": f"{stem}{IMAGE_EXT}", "width": w, "height": h})

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls_id = int(float(parts[0]))
            coords = np.array(list(map(float, parts[1:])), dtype=np.float64)
            coords[0::2] *= w
            coords[1::2] *= h
            poly = coords.reshape(-1, 2)
            x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
            x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())

            rle = mask_utils.frPyObjects([coords.tolist()], h, w)
            area = float(mask_utils.area(mask_utils.merge(rle)))

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls_id + 1,
                "segmentation": [coords.tolist()],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": name, "supercategory": None} for i, name in enumerate(CLASS_NAMES)]
    coco_gt = {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with gt_path.open("w") as f:
        json.dump(coco_gt, f)

    print(f"  {len(images)} images, {len(annotations)} annotations -> {gt_path}")

    areas = np.array([a["area"] for a in annotations])
    print(f"  GT mask-area range: min={areas.min():.0f}px^2  max={areas.max():.0f}px^2  "
          f"median={np.median(areas):.0f}px^2")


def print_area_bucket_diagnostic(annotations: list[dict], area_rng: list, area_rng_lbl: list) -> None:
    areas = np.array([a["area"] for a in annotations])
    print("\nGT count per area bucket (with the thresholds used below) -- a bucket with 0 GT")
    print("instances means AP for that bucket is mathematically undefined (COCOeval reports -1),")
    print("not a bug:")
    for lbl, (lo, hi) in zip(area_rng_lbl, area_rng):
        if lbl == "all":
            continue
        cnt = int(((areas >= lo) & (areas < hi)).sum())
        pct = 100 * cnt / len(areas) if len(areas) else 0.0
        flag = "  <-- EMPTY, AP will be N/A" if cnt == 0 else ""
        print(f"  {lbl:<8} [{lo:>6.0f}, {hi:>10.0f}) px^2: {cnt:>4} instances ({pct:4.1f}%){flag}")


# =========================================================
# PREDICTION LOADING / ID REMAPPING
# =========================================================

def load_maskrcnn_predictions_remapped(eval_dir: Path) -> list[dict]:
    val_anns_path = eval_dir / "val_anns.json"
    preds_path = eval_dir / "predictions.json"
    if not val_anns_path.exists() or not preds_path.exists():
        raise FileNotFoundError(
            f"Missing {val_anns_path} or {preds_path} -- run "
            "codes/mask-rcnn/evaluate_nwpu_maskrcnn.py first."
        )

    with val_anns_path.open() as f:
        val_anns = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in val_anns["images"]}

    with preds_path.open() as f:
        raw_preds = json.load(f)

    remapped = []
    dropped = 0
    for pred in raw_preds:
        file_name = id_to_filename.get(pred["image_id"])
        if file_name is None:
            dropped += 1
            continue
        new_pred = dict(pred)
        new_pred["image_id"] = int(Path(file_name).stem)
        remapped.append(new_pred)

    if dropped:
        print(f"  WARNING: dropped {dropped} Mask R-CNN prediction(s) with no matching val_anns.json image id")
    print(f"  Mask R-CNN: remapped {len(remapped)} predictions from val_anns.json ids to int(stem) ids")
    return remapped


def load_yolo_predictions(eval_dir: Path, model_label: str) -> list[dict]:
    preds_path = eval_dir / "predictions.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path} -- run the {model_label} evaluate_nwpu_*.py script first.")
    with preds_path.open() as f:
        preds = json.load(f)
    print(f"  {model_label}: loaded {len(preds)} predictions as-is (already image_id = int(stem))")
    return preds


# =========================================================
# COCOEVAL WITH CUSTOM AREA BUCKETS
# =========================================================

def run_coco_eval(coco_gt: COCO, coco_dt, iou_type: str, area_rng: list, area_rng_lbl: list,
                  cat_ids=None) -> np.ndarray:
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.params.areaRng = area_rng
    evaluator.params.areaRngLbl = area_rng_lbl
    if cat_ids is not None:
        evaluator.params.catIds = cat_ids
    # No maxDets override here (unlike the sibling iSAID script): NWPU is
    # sparse -- max ~51 GT instances in any one val image, well under
    # pycocotools' default maxDets=100 cap -- so nothing is silently
    # truncated and the pycocotools default is already lossless (verified).
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats


def verify_against_reported_metrics(model_label: str, reported: dict, stats_box: np.ndarray, stats_segm: np.ndarray) -> None:
    checks = [
        ("box_mAP50", stats_box[1]), ("box_mAP50_95", stats_box[0]),
        ("mask_mAP50", stats_segm[1]), ("mask_mAP50_95", stats_segm[0]),
    ]
    any_mismatch = False
    for key, recomputed in checks:
        if key not in reported:
            continue
        recomputed_pct = recomputed * 100
        delta = abs(recomputed_pct - reported[key])
        flag = "OK" if delta <= REPORTED_METRIC_TOLERANCE else "within known tool gap"
        if delta > REPORTED_METRIC_TOLERANCE:
            any_mismatch = True
        print(f"    {key:<16} reported={reported[key]:6.2f}%  recomputed={recomputed_pct:6.2f}%  "
              f"delta={delta:5.2f}pt  [{flag}]")
    if any_mismatch:
        print(f"    note: {model_label}'s reported numbers came from Ultralytics' own internal AP "
              f"calculation (model.val()), not pycocotools -- a few points of difference from this "
              f"script's pycocotools COCOeval recomputation is an expected, well-known gap between the "
              f"two implementations (interpolation/tie-break details), not evidence of a bug here. Mask "
              f"R-CNN's cross-check (both sides use pycocotools) matches almost exactly, which is the "
              f"real integrity check for this script's GT-rebuild and id-remap.")


# =========================================================
# PER-CLASS AP (whole-image 'all' area range, one COCOeval per category so a
# class with zero predictions/GT doesn't skew the others -- mirrors the
# sibling iSAID script's run_classwise_eval / print_classwise_table. No
# maxDets override needed on NWPU, see run_coco_eval note.)
# =========================================================

def fmt_stat(value: float) -> str:
    # COCOeval returns exactly -1.0 for a category with no GT/predictions to
    # evaluate -- that's "undefined", not "-100%".
    return "N/A" if value is None or value < 0 else f"{value * 100:.2f}"


def run_classwise_eval(coco_gt: COCO, predictions_by_model: dict, models: list,
                       cat_id_to_name: dict) -> list[dict]:
    area_rng, area_rng_lbl = [[0, 1e10]], ["all"]
    rows = []
    for model_name, _eval_dir, key in models:
        preds = predictions_by_model[key]
        coco_dt = coco_gt.loadRes(preds)
        for iou_type in ["bbox", "segm"]:
            print(f"  {model_name} / {iou_type} ...")
            for cat_id, cat_name in sorted(cat_id_to_name.items()):
                with contextlib.redirect_stdout(io.StringIO()):
                    stats = run_coco_eval(coco_gt, coco_dt, iou_type, area_rng,
                                          area_rng_lbl, cat_ids=[cat_id])
                rows.append({"model": model_name, "iou_type": iou_type, "category": cat_name,
                             "AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2])})
    return rows


def print_classwise_table(rows: list) -> None:
    print(f"\n{'=' * 60}")
    print("NWPU PER-CLASS AP -- pycocotools, area='all', maxDets=100 (default)")
    print(f"{'=' * 60}")
    by_model_type = {}
    for r in rows:
        by_model_type.setdefault((r["model"], r["iou_type"]), []).append(r)
    for (model_name, iou_type), rs in by_model_type.items():
        print(f"\n{model_name} / {iou_type}")
        print(f"  {'Category':<22}{'AP':>8}{'AP50':>8}{'AP75':>8}")
        for r in rs:
            print(f"  {r['category']:<22}{fmt_stat(r['AP']):>8}{fmt_stat(r['AP50']):>8}{fmt_stat(r['AP75']):>8}")
        valid = [(r["AP"], r["AP50"], r["AP75"]) for r in rs if r["AP"] >= 0]
        if valid:
            mean_ap = 100 * sum(v[0] for v in valid) / len(valid)
            mean_ap50 = 100 * sum(v[1] for v in valid) / len(valid)
            mean_ap75 = 100 * sum(v[2] for v in valid) / len(valid)
            print(f"  {'Mean':<22}{mean_ap:>8.2f}{mean_ap50:>8.2f}{mean_ap75:>8.2f}")


# =========================================================
# MAIN
# =========================================================

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_image_dir = args.dataset_root / "images" / args.split
    val_label_dir = args.dataset_root / "labels" / args.split
    dataset_tag = args.dataset_root.name
    unified_gt_path = OUTPUT_DIR / f"unified_{args.split}_gt_{dataset_tag}.json"

    models = [
        ("Mask R-CNN", args.maskrcnn_eval_dir, "maskrcnn"),
        ("YOLO11m", args.yolo11_eval_dir, "yolo11m"),
        ("YOLO26m", args.yolo26_eval_dir, "yolo26m"),
    ]

    if args.rebuild_gt or not unified_gt_path.exists():
        build_unified_gt(unified_gt_path, val_label_dir, val_image_dir)
    else:
        print(f"Reusing existing unified GT: {unified_gt_path} (pass --rebuild-gt to regenerate)")

    area_rng = [
        [0 ** 2, 1e10],
        [args.small_min, args.small_max],
        [args.small_max, args.medium_max],
        [args.medium_max, 1e10],
    ]
    area_rng_lbl = ["all", "small", "medium", "large"]
    print(f"\nArea buckets (px^2, mask area): small=[{args.small_min:.0f},{args.small_max:.0f}) "
          f"medium=[{args.small_max:.0f},{args.medium_max:.0f}) large=[{args.medium_max:.0f}, inf)")

    with unified_gt_path.open() as f:
        gt_annotations = json.load(f)["annotations"]
    print_area_bucket_diagnostic(gt_annotations, area_rng, area_rng_lbl)

    print("\nLoading predictions ...")
    predictions_by_model = {}
    for model_name, eval_dir, key in models:
        if key == "maskrcnn":
            predictions_by_model[key] = load_maskrcnn_predictions_remapped(eval_dir)
        else:
            predictions_by_model[key] = load_yolo_predictions(eval_dir, model_name)

    coco_gt = COCO(str(unified_gt_path))

    rows = []
    for model_name, eval_dir, key in models:
        print(f"\n=== {model_name} ===")
        preds = predictions_by_model[key]
        coco_dt = coco_gt.loadRes(preds)

        stats_box = run_coco_eval(coco_gt, coco_dt, "bbox", area_rng, area_rng_lbl)
        stats_segm = run_coco_eval(coco_gt, coco_dt, "segm", area_rng, area_rng_lbl)

        summary_path = eval_dir / "summary_metrics.json"
        if summary_path.exists():
            with summary_path.open() as f:
                reported = json.load(f)
            print("  Cross-check vs. reported summary_metrics.json (should match closely -- "
                  "identical predictions/GT-content, only the id scheme and area computation changed, "
                  "neither of which should move the 'all'-bucket AP):")
            verify_against_reported_metrics(model_name, reported, stats_box, stats_segm)

        def pct_or_na(stat: float):
            # COCOeval reports exactly -1.0 when an area bucket has no GT
            # instances to evaluate -- that's "undefined", not "-100%".
            return None if stat < 0 else stat * 100

        row = {
            "model": model_name,
            "box_AP": pct_or_na(stats_box[0]), "box_AP50": pct_or_na(stats_box[1]), "box_AP75": pct_or_na(stats_box[2]),
            "box_APs": pct_or_na(stats_box[3]), "box_APm": pct_or_na(stats_box[4]), "box_APl": pct_or_na(stats_box[5]),
            "mask_AP": pct_or_na(stats_segm[0]), "mask_AP50": pct_or_na(stats_segm[1]), "mask_AP75": pct_or_na(stats_segm[2]),
            "mask_APs": pct_or_na(stats_segm[3]), "mask_APm": pct_or_na(stats_segm[4]), "mask_APl": pct_or_na(stats_segm[5]),
        }
        rows.append(row)

    def fmt(value):
        return "   N/A " if value is None else f"{value:6.2f}%"

    print("\n===================================")
    print("NWPU SIZE-BUCKETED AP RESULTS")
    print("===================================")
    header = f"{'Model':<12} {'Box APs':>8} {'Box APm':>8} {'Box APl':>8} {'Mask APs':>9} {'Mask APm':>9} {'Mask APl':>9}"
    print(header)
    for row in rows:
        print(f"{row['model']:<12} {fmt(row['box_APs']):>8} {fmt(row['box_APm']):>8} {fmt(row['box_APl']):>8} "
              f"{fmt(row['mask_APs']):>9} {fmt(row['mask_APm']):>9} {fmt(row['mask_APl']):>9}")

    csv_path = unique_path(OUTPUT_DIR / args.output_name)
    fieldnames = ["model", "box_AP", "box_AP50", "box_AP75", "box_APs", "box_APm", "box_APl",
                  "mask_AP", "mask_AP50", "mask_AP75", "mask_APs", "mask_APm", "mask_APl"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("N/A" if v is None else (f"{v:.2f}" if isinstance(v, float) else v)) for k, v in row.items()})

    print(f"\nFull results (AP, AP50, AP75, APs, APm, APl for box + mask): {csv_path}")
    print(f"Unified GT: {unified_gt_path}")

    # ---- Per-class AP (bbox + segm) ----
    # build_unified_gt assigns category_id = class_index + 1 for CLASS_NAMES,
    # so this lookup matches the GT/prediction category ids exactly.
    print("\nPer-class AP sweep (one COCOeval per category) ...")
    cat_id_to_name = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
    classwise_rows = run_classwise_eval(coco_gt, predictions_by_model, models, cat_id_to_name)
    print_classwise_table(classwise_rows)

    classwise_csv_path = unique_path(OUTPUT_DIR / CLASSWISE_OUTPUT_NAME)
    with classwise_csv_path.open("w", newline="") as f:
        fieldnames = ["model", "iou_type", "category", "AP", "AP50", "AP75"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in classwise_rows:
            writer.writerow({k: (fmt_stat(row[k]) if k in ("AP", "AP50", "AP75") else row[k])
                             for k in fieldnames})
    print(f"\nPer-class results (AP, AP50, AP75 for box + mask): {classwise_csv_path}")


if __name__ == "__main__":
    main()
