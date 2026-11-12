# Origin: the four original per-model scripts (iSAID train_yolo_musgd.py / train_yolo11.py and
# NWPU train_nwpu_yolo26.py / train_nwpu_yolo11.py). They differed only in the values now stored in
# configs/*.yaml; the call is the same `YOLO(model).train(...)` with the same keyword arguments.
"""Train YOLO11m-seg / YOLO26m-seg from a YAML config.

    python scripts/train_yolo.py --root <dataset root> --config configs/yolo26m_isaid.yaml

``data`` and ``project`` inside the config are relative to ``--root``.
"""
import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from common import resolve_root

if __name__ == '__main__':
    root = resolve_root("DATASET_ROOT")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(cfg["model"])

    results = model.train(
        data=str(root / cfg["data"]),
        project=str(root / cfg["project"]),
        name=cfg["name"],
        **cfg["train"],
    )

    print("\nTraining complete!")
