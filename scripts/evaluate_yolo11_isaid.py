# Origin: iSAID yolo11/evaluate_yolo11.py. Hard-coded paths replaced by --root; logic unchanged.

from ultralytics import YOLO
from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────
from common import resolve_root
ROOT = resolve_root("ISAID_ROOT")

MODEL_PATH = ROOT / "work_dirs" / "paper" / "yolo11" / "yolo11m_isaid" / "weights" / "best.pt"

YAML_PATH = ROOT / "yolo26_dataset" / "isaid.yaml"

OUTPUT_DIR = ROOT / "work_dirs" / "paper" / "yolo11" / "yolo11m_isaid" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Evaluate ───────────────────────────────────────
def main():
    print("=== YOLO11m Evaluation on iSAID ===\n")
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {YAML_PATH}\n")

    model = YOLO(str(MODEL_PATH))

    results = model.val(
        data    = str(YAML_PATH),
        imgsz   = 800,
        batch   = 4,
        device  = 0,
        workers = 0,
        verbose = True,
        plots   = True,
        save_json = True,
        project = str(OUTPUT_DIR.parent),
        name    = "eval_results"
    )

    # ── Print Summary ──────────────────────────────
    print("\n=== YOLO11m iSAID Results ===")
    print(f"Mask mAP50    : "
          f"{results.seg.map50*100:.2f}%")
    print(f"Mask mAP50-95 : "
          f"{results.seg.map*100:.2f}%")
    print(f"Box  mAP50    : "
          f"{results.box.map50*100:.2f}%")
    print(f"Box  mAP50-95 : "
          f"{results.box.map*100:.2f}%")

    # ── Per-Class Results ──────────────────────────
    print("\n=== Per-Class Mask AP50 ===")
    class_names = model.names
    
    data = []
    for i, (name) in enumerate(
            class_names.values()):
        mask_ap50 = results.seg.ap50[i] * 100
        box_ap50  = results.box.ap50[i] * 100
        print(f"  {name:<20}: "
              f"Mask={mask_ap50:.1f}%  "
              f"Box={box_ap50:.1f}%")
        data.append({
            'Category'    : name,
            'Mask_AP50'   : round(mask_ap50, 2),
            'Box_AP50'    : round(box_ap50, 2),
        })

    # ── Save to Excel ──────────────────────────────
    df = pd.DataFrame(data)
    
    # Add summary row
    summary = pd.DataFrame([{
        'Category'    : 'ALL (mAP50)',
        'Mask_AP50'   : round(
            results.seg.map50*100, 2),
        'Box_AP50'    : round(
            results.box.map50*100, 2),
    }])
    df = pd.concat([df, summary], 
                   ignore_index=True)

    excel_path = OUTPUT_DIR / \
        "yolo11m_evaluation.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"\n✅ Results saved: {excel_path}")

    # ── Speed ──────────────────────────────────────
    print(f"\n=== Speed ===")
    print(f"Preprocess : "
          f"{results.speed['preprocess']:.1f}ms")
    print(f"Inference  : "
          f"{results.speed['inference']:.1f}ms")
    print(f"Postprocess: "
          f"{results.speed['postprocess']:.1f}ms")

if __name__ == '__main__':
    main()