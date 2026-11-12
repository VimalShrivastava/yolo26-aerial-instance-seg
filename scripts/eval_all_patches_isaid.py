"""Re-evaluate YOLO11m / YOLO26m / Mask R-CNN on ALL 9,581 iSAID validation patches and compare with
the 5,541 annotated patches used for the paper tables.

Inference only, same checkpoints, maxDets = 300, COCO-default size buckets.
Stage 1: inference on the 4,040 background-only patches. The predictions on the annotated patches are the
         ones already written by evaluate_yolo*_isaid.py and eval_size_ap_isaid.py (run those first).
Stage 2: pycocotools on (a) the 5,541 annotated patches and (b) all 9,581 patches.

Usage:  python scripts/eval_all_patches_isaid.py --root <iSAID root>
Output: <root>/work_dirs/paper/all_patches_eval/all_vs_annotated_patches.csv
        (a copy is stored in results/isaid/).
Re-scoring the 5,541 annotated patches reproduces the paper tables, which is the built-in validity check.
"""
import copy, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO_SCRIPTS = str(Path(__file__).resolve().parent)
sys.path.insert(0, REPO_SCRIPTS)
from common import resolve_root

ROOT = resolve_root("ISAID_ROOT")
OUT = ROOT / "work_dirs/paper/all_patches_eval"
OUT.mkdir(parents=True, exist_ok=True)
SCR = OUT / "inference"
SCR.mkdir(exist_ok=True)

all_imgs = sorted((ROOT / "Dataset_patches/val/images").glob("*.png"))
annot = {p.stem for p in (ROOT / "yolo26_dataset/labels/val").glob("*.txt")}
empty = [p for p in all_imgs if p.stem not in annot]
assert len(all_imgs) == 9581 and len(annot) == 5541 and len(empty) == 4040, (len(all_imgs), len(annot), len(empty))
print("patches:", len(all_imgs), "annotated:", len(annot), "background-only:", len(empty), flush=True)


# ---------------------------------------------------------------- stage 1
def stage1():
    from ultralytics import YOLO
    lst = SCR / "empty_val.txt"
    lst.write_text("\n".join(p.as_posix() for p in empty), encoding="utf-8")
    names = [l for l in (ROOT / "yolo26_dataset/isaid.yaml").read_text().splitlines() if l.startswith("names:")][0]
    (SCR / "empty.yaml").write_text(
        f"path: {SCR.as_posix()}\ntrain: {lst.as_posix()}\nval: {lst.as_posix()}\nnc: 15\n{names}\n")
    for tag, w in (("y26", ROOT / "work_dirs/paper/yolo26/yolo26m_isaid_musgd/weights/best.pt"),
                   ("y11", ROOT / "work_dirs/paper/yolo11/yolo11m_isaid/weights/best.pt")):
        if (SCR / tag / "predictions.json").exists():
            continue
        t0 = time.time()
        # workers=0: DataLoader workers on Windows cannot pickle the dataset wrapper
        YOLO(str(w)).val(data=str(SCR / "empty.yaml"), imgsz=800, batch=4, device=0, workers=0, save_json=True,
                         plots=False, verbose=False, project=str(SCR), name=tag, exist_ok=True)
        print(f"{tag} inference on empty patches done in {time.time() - t0:.0f}s", flush=True)
    if not (SCR / "maskrcnn_empty.json").exists():
        sys.argv = [sys.argv[0], "--root", str(ROOT)]
        import eval_size_ap_isaid as mod

        class EmptyDS(mod.ValImageDataset):
            def __init__(self, image_dir):
                self.files = empty

        mod.ValImageDataset = EmptyDS
        t0 = time.time()
        bb, sg = mod.run_maskrcnn_inference(0.001, torch.device("cuda"), 4, 0)
        json.dump({"bbox": bb, "segm": sg}, open(SCR / "maskrcnn_empty.json", "w"))
        print(f"maskrcnn inference on empty patches done in {time.time() - t0:.0f}s", flush=True)
    # YOLO val may drop a labels cache next to the images; remove it if it appeared
    c = ROOT / "Dataset_patches/val/labels.cache"
    if c.exists():
        c.unlink()
        print("removed", c)


# ---------------------------------------------------------------- stage 2
CLASSES = ["storage_tank", "large_vehicle", "small_vehicle", "plane", "ship", "swimming_pool", "harbor", "tennis_court",
           "ground_track_field", "soccer_ball_field", "baseball_diamond", "bridge", "basketball_court", "roundabout",
           "helicopter"]
INF = 1e10
AREA = [[0, INF], [0, 1024], [1024, 9216], [9216, INF]]


def stage2():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = json.load(open(ROOT / "work_dirs/paper/size_ap_eval/isaid_val_unified_gt.json"))
    have = {i["id"] for i in gt["images"]}
    assert have == annot, "cached GT image set differs from annotated set"
    for p in empty:
        with Image.open(p) as im:
            w, h = im.size
        gt["images"].append({"id": p.stem, "file_name": p.name, "width": w, "height": h})
    print("GT images:", len(gt["images"]), "instances:", len(gt["annotations"]), flush=True)
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    ids_all = sorted(i["id"] for i in gt["images"])
    ids_ann = sorted(annot)

    def yolo(tag, old):
        a = json.load(open(old))
        a = [p for p in a if p["image_id"] in annot]
        b = json.load(open(SCR / tag / "predictions.json"))
        assert all(p["image_id"] not in annot for p in b)
        return a + b

    mr_old = json.load(open(ROOT / "work_dirs/paper/size_ap_eval/maskrcnn_predictions_val.json"))
    mr_new = json.load(open(SCR / "maskrcnn_empty.json"))
    preds = {
        "MaskRCNN": {"bbox": mr_old["bbox"] + mr_new["bbox"], "segm": mr_old["segm"] + mr_new["segm"]},
        "YOLO11m": (lambda d: {"bbox": d, "segm": d})(
            yolo("y11", ROOT / "work_dirs/paper/yolo11/yolo11m_isaid/eval_results/predictions.json")),
        "YOLO26m": (lambda d: {"bbox": d, "segm": d})(
            yolo("y26", ROOT / "work_dirs/paper/yolo26/yolo26m_isaid_musgd/eval_results/predictions.json")),
    }
    for m in preds:
        n_new = sum(1 for p in preds[m]["segm"] if p["image_id"] not in annot)
        print(f"{m}: {len(preds[m]['segm'])} dets total, {n_new} on background-only patches", flush=True)

    rows = []

    def ev(model, iou, scope, ids):
        dt = coco_gt.loadRes(copy.deepcopy(preds[model][iou]))
        E = COCOeval(coco_gt, dt, iou)
        E.params.imgIds = ids
        E.params.maxDets = [1, 10, 300]
        E.params.areaRng = AREA
        E.params.areaRngLbl = ["all", "small", "medium", "large"]
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            E.evaluate()
            E.accumulate()
        P = E.eval["precision"]  # T,R,K,A,M

        def mean(x):
            x = x[x > -1]
            return float(x.mean()) if x.size else float("nan")

        m = 2  # maxDets = 300
        row = dict(model=model, iou=iou, scope=scope, AP=mean(P[:, :, :, 0, m]), AP50=mean(P[0, :, :, 0, m]),
                   AP75=mean(P[5, :, :, 0, m]), APs=mean(P[:, :, :, 1, m]), APm=mean(P[:, :, :, 2, m]),
                   APl=mean(P[:, :, :, 3, m]))
        for k, c in enumerate(CLASSES):
            row["cls_" + c] = mean(P[:, :, k, 0, m])
        rows.append(row)
        print(f"{model:9s}{iou:5s}{scope:5s} AP={row['AP'] * 100:6.2f} AP50={row['AP50'] * 100:6.2f}", flush=True)

    for scope, ids in (("ann", ids_ann), ("all", ids_all)):
        for model in preds:
            for iou in ("segm", "bbox"):
                ev(model, iou, scope, ids)
    with open(OUT / "all_vs_annotated_patches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("saved", OUT / "all_vs_annotated_patches.csv", flush=True)


if __name__ == "__main__":
    stage1()
    stage2()
    print("ALL_DONE", flush=True)
