# Origin: NWPU yolo26/evaluate_nwpu_yolo26.py. Hard-coded paths replaced by --root; logic unchanged.

import argparse
import json
from pathlib import Path
import hashlib

import pandas as pd
from ultralytics import YOLO


from common import resolve_root
ROOT = resolve_root("NWPU_ROOT")

ROOT_DIR = ROOT
DEFAULT_MODEL_PATH = ROOT_DIR / "work_dirs" / "paper" / "yolo26" / "yolo26m_nwpu" / "weights" / "best.pt"
DEFAULT_YAML_PATH = ROOT_DIR / "yolo_dataset" / "nwpu.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO26m on NWPU VHR-10.")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which split to evaluate against. The paper uses the default, 'val'; "
             "'test' requires a --data-yaml that has a test: key.",
    )
    parser.add_argument(
        "--data-yaml", type=Path, default=DEFAULT_YAML_PATH,
        help="Dataset yaml to evaluate against.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help="Checkpoint to evaluate (work_dirs/<run_name>/weights/best.pt).",
    )
    return parser.parse_args()


ARGS = parse_args()
MODEL_PATH = ARGS.model_path
YAML_PATH = ARGS.data_yaml

# Default args reproduce the exact old folder name (yolo26m_nwpu_evaluation);
# a non-default split/model-path gets its own folder so it never clobbers an
# existing val-split evaluation.
RUN_NAME = MODEL_PATH.parent.parent.name
EVAL_DIR_NAME = f"{RUN_NAME}_evaluation" if ARGS.split == "val" else f"{RUN_NAME}_evaluation_{ARGS.split}"
EVAL_WORK_DIR = ROOT_DIR / "work_dirs" / "paper" / "yolo26" / EVAL_DIR_NAME
OUTPUT_DIR = EVAL_WORK_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_FILE = OUTPUT_DIR / f"{RUN_NAME}_evaluation.xlsx"
SUMMARY_JSON = OUTPUT_DIR / "summary_metrics.json"


def file_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_reports(model, results):
    summary = {
        "mask_mAP50": round(float(results.seg.map50 * 100), 2),
        "mask_mAP50_95": round(float(results.seg.map * 100), 2),
        "box_mAP50": round(float(results.box.map50 * 100), 2),
        "box_mAP50_95": round(float(results.box.map * 100), 2),
        "model_path": str(MODEL_PATH),
        "model_sha256": file_sha256(MODEL_PATH),
        "data_yaml": str(YAML_PATH),
        "split": ARGS.split,
        "ultralytics_output_dir": str(results.save_dir),
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    per_class_rows = []
    for i, name in model.names.items():
        mask_ap50 = float(results.seg.ap50[i] * 100)
        box_ap50 = float(results.box.ap50[i] * 100)
        print(
            f"  {name:<20} "
            f"Mask: {mask_ap50:>6.2f}%  "
            f"Box: {box_ap50:>6.2f}%"
        )
        per_class_rows.append({
            "Category": name,
            "Mask_AP50": round(mask_ap50, 2),
            "Box_AP50": round(box_ap50, 2),
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df = pd.concat([
        per_class_df,
        pd.DataFrame([{
            "Category": "ALL",
            "Mask_AP50": summary["mask_mAP50"],
            "Box_AP50": summary["box_mAP50"],
        }]),
    ], ignore_index=True)

    overall_df = pd.DataFrame([
        {"Metric": "Mask mAP50", "Value (%)": summary["mask_mAP50"]},
        {"Metric": "Mask mAP50-95", "Value (%)": summary["mask_mAP50_95"]},
        {"Metric": "Box mAP50", "Value (%)": summary["box_mAP50"]},
        {"Metric": "Box mAP50-95", "Value (%)": summary["box_mAP50_95"]},
    ])

    files_df = pd.DataFrame([
        {"Item": "Model", "Path": summary["model_path"]},
        {"Item": "Model SHA256", "Path": summary["model_sha256"]},
        {"Item": "Data YAML", "Path": summary["data_yaml"]},
        {"Item": "Ultralytics output", "Path": summary["ultralytics_output_dir"]},
    ])

    with pd.ExcelWriter(EXCEL_FILE) as writer:
        overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
        per_class_df.to_excel(writer, sheet_name="Per-Class AP50", index=False)
        files_df.to_excel(writer, sheet_name="Files", index=False)

    return summary


def main():
    print("=== YOLO26m Evaluation on NWPU VHR-10 ===\n")
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {YAML_PATH}")
    print(f"Split : {ARGS.split}")
    print(f"Output: {OUTPUT_DIR}\n")

    model = YOLO(str(MODEL_PATH))
    results = model.val(
        data=str(YAML_PATH),
        split=ARGS.split,
        imgsz=800,
        batch=4,
        device=0,
        workers=0,
        verbose=True,
        plots=True,
        save_json=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
    )

    print("\n=== Per-Class AP50 ===")
    summary = build_reports(model, results)

    print("\n=== YOLO26m NWPU Results ===")
    print(f"Mask mAP50    : {summary['mask_mAP50']:.2f}%")
    print(f"Mask mAP50-95 : {summary['mask_mAP50_95']:.2f}%")
    print(f"Box  mAP50    : {summary['box_mAP50']:.2f}%")
    print(f"Box  mAP50-95 : {summary['box_mAP50_95']:.2f}%")
    print(f"\nSaved Excel: {EXCEL_FILE}")
    print(f"Saved JSON : {SUMMARY_JSON}")
    print(f"Inference  : {results.speed['inference']:.1f} ms/image")


if __name__ == "__main__":
    main()
