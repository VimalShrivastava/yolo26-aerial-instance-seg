# Origin: iSAID yolo26/evaluate_yolo26.py. Hard-coded paths replaced by --root; logic unchanged.

from ultralytics import YOLO
from pathlib import Path
import pandas as pd

from common import resolve_root
ROOT = resolve_root("ISAID_ROOT")

MODEL_PATH = ROOT / "work_dirs" / "paper" / "yolo26" / "yolo26m_isaid_musgd" / "weights" / "best.pt"

YAML_PATH = ROOT / "yolo26_dataset" / "isaid.yaml"

OUTPUT_DIR = ROOT / "work_dirs" / "paper" / "yolo26" / "yolo26m_isaid_musgd" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== YOLO26m MuSGD Evaluation ===\n")

    model = YOLO(str(MODEL_PATH))

    results = model.val(
        data      = str(YAML_PATH),
        imgsz     = 800,
        batch     = 4,
        device    = 0,
        workers   = 0,
        verbose   = True,
        plots     = True,
        save_json = True,
        project   = str(OUTPUT_DIR.parent),
        name      = "eval_results"
    )

    print("\n=== YOLO26m MuSGD Results ===")
    print(f"Mask mAP50    : "
          f"{results.seg.map50*100:.2f}%")
    print(f"Mask mAP50-95 : "
          f"{results.seg.map*100:.2f}%")
    print(f"Box  mAP50    : "
          f"{results.box.map50*100:.2f}%")
    print(f"Box  mAP50-95 : "
          f"{results.box.map*100:.2f}%")

    print("\n=== Per-Class Mask AP50 ===")
    class_names = model.names
    data = []

    for i, name in enumerate(
            class_names.values()):
        mask_ap50 = results.seg.ap50[i] * 100
        box_ap50  = results.box.ap50[i] * 100
        print(f"  {name:<20}: "
              f"Mask={mask_ap50:.1f}%  "
              f"Box={box_ap50:.1f}%")
        data.append({
            'Category' : name,
            'Mask_AP50': round(mask_ap50, 2),
            'Box_AP50' : round(box_ap50, 2),
        })

    df = pd.DataFrame(data)
    summary = pd.DataFrame([{
        'Category' : 'ALL (mAP50)',
        'Mask_AP50': round(
            results.seg.map50*100, 2),
        'Box_AP50' : round(
            results.box.map50*100, 2),
    }])
    df = pd.concat([df, summary],
                   ignore_index=True)

    excel_path = OUTPUT_DIR / \
        "yolo26m_musgd_evaluation.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"\n✅ Saved: {excel_path}")

    print(f"\n=== Speed ===")
    print(f"Inference: "
          f"{results.speed['inference']:.1f}ms")

if __name__ == '__main__':
    main()