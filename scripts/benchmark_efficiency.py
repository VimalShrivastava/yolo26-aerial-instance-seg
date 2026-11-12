# Origin: iSAID evaluation/benchmark_efficiency.py (reads iSAID and NWPU checkpoints). Hard-coded paths replaced by --root; logic unchanged.
"""Computational-efficiency benchmark for the paper's Table X: params /
GFLOPs@800 / per-image latency / FPS, measured from the ACTUAL trained
checkpoints of Mask R-CNN, YOLO11m-seg and YOLO26m-seg on iSAID and
NWPU VHR-10.

Protocol (identical for every model and both datasets):
  - input 800x800 (the training/eval imgsz of all six checkpoints),
    batch=1, FP32, CUDA, cudnn.benchmark
  - real validation images, aspect-preserving letterbox to exactly 800x800
    (pad value 114): iSAID patches <=800 are only padded, NWPU images up to
    ~1000 px are scaled down first
  - 30 warmup forward passes, then per-image wall time with
    torch.cuda.synchronize() before/after each timed call
  - timed region = the call that yields final detections:
      * Mask R-CNN: model([img]) -- torchvision's forward already contains
        its transform, RPN, RoI heads and postprocess
      * YOLO11m: fused net(x) + non_max_suppression (YOLO11 requires NMS)
      * YOLO26m: fused net(x) only (end-to-end head, NMS-free; asserted at
        runtime via net.end2end)
    Image decode and host->device copy are NOT timed for any model.
  - FLOPs convention: GFLOPs = 2 x MACs (the convention Ultralytics uses).
      * Mask R-CNN: fvcore FlopCountAnalysis (reports MACs) x 2 on
        (1, 3, 800, 800)
      * YOLO: ultralytics.utils.torch_utils.get_flops(net, 800) -- thop MACs
        x 2 internally. Do NOT use model.info(): it always profiles at
        640x640 regardless of the training imgsz (this is exactly how the
        earlier draft table ended up with 113.0/121.2 GFLOPs -- those are
        the fused models' 640 numbers, verbatim).
  - YOLO params/GFLOPs are reported for the FUSED (deployment) model; the
    unfused training graph of YOLO26m is larger (27.0M vs 23.5M fused).
  All checkpoints are 3-channel RGB models.

Usage (mask_rcnn conda env):
  python benchmark_efficiency.py --dataset isaid --stage flops
  python benchmark_efficiency.py --dataset isaid --stage y11   -n 0
  python benchmark_efficiency.py --dataset isaid --stage y26   -n 0
  python benchmark_efficiency.py --dataset isaid --stage mrcnn -n 0
  python benchmark_efficiency.py --dataset nwpu  --stage y11 ...   (same)

Stages are separate processes on purpose: each one exits (and frees the GPU)
before the next starts, so a hung run can never squat on the card. -n 0
means the full val set; with fewer than (WARMUP+N) images available the
warmup reuses the first image so every val image still gets timed.

Results accumulate in work_dirs/paper/efficiency_benchmark/bench_results.json,
keyed "<dataset>_<Model>_stats" / "<dataset>_<Model>_latency".

Measured 2026-07-14/15 on RTX 3090, torch 2.1.0+cu118, ultralytics 8.4.33
(iSAID: full 5541-image val set; NWPU: full 130-image val set).

CAUTION -- CPU contention inflates latency: this box is a shared desktop, and
concurrent apps delay CUDA kernel launches inside the timed region. Verified
2026-07-15: the same YOLO11m stage measured 13.0 ms on a quiet machine and
20.6-21.9 ms with desktop activity, at only 29-39% GPU utilization and full
boost clocks (i.e. the GPU was waiting on the CPU, not throttling). YOLO11m
is the most sensitive stage because its NMS adds per-image host-side work.
Re-run only on a quiet machine, and sanity-check that YOLO11m's iSAID and
NWPU latencies agree (~13 ms): the architecture and input size are identical,
so any gap between them is measurement contamination, not a real effect.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from common import resolve_root
ISAID_ROOT_DIR = resolve_root("ISAID_ROOT", "--isaid-root", required=False)
NWPU_ROOT_DIR = resolve_root("NWPU_ROOT", "--nwpu-root", required=False)

DATASETS = {
    "isaid": {
        "val": ISAID_ROOT_DIR / "yolo26_dataset" / "images" / "val",
        "mr_ckpt": ISAID_ROOT_DIR / "work_dirs" / "paper" / "maskrcnn_torchvision" / "best_model.pth",
        "y11_ckpt": ISAID_ROOT_DIR / "work_dirs" / "paper" / "yolo11" / "yolo11m_isaid" / "weights" / "best.pt",
        "y26_ckpt": ISAID_ROOT_DIR / "work_dirs" / "paper" / "yolo26" / "yolo26m_isaid_musgd" / "weights" / "best.pt",
        "nc": 16,  # 15 classes + background (Mask R-CNN head)
    },
    "nwpu": {
        "val": NWPU_ROOT_DIR / "yolo_dataset" / "images" / "val",
        "mr_ckpt": NWPU_ROOT_DIR / "work_dirs" / "paper" / "mask_rcnn" / "maskrcnn_nwpu" / "best_model.pth",
        "y11_ckpt": NWPU_ROOT_DIR / "work_dirs" / "paper" / "yolo11" / "yolo11m_nwpu" / "weights" / "best.pt",
        "y26_ckpt": NWPU_ROOT_DIR / "work_dirs" / "paper" / "yolo26" / "yolo26m_nwpu" / "weights" / "best.pt",
        "nc": 11,  # 10 classes + background
    },
}
OUT = ISAID_ROOT_DIR / "work_dirs" / "paper" / "efficiency_benchmark" / "bench_results.json"
DEV = torch.device("cuda")
IMGSZ = 800
WARMUP = 30


def build_maskrcnn(ckpt: Path, nc: int):
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, nc)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, nc)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    m.load_state_dict(sd)
    ck_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    return m.eval(), ck_params


def save(key: str, val: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1))
    print(f"[saved {key}] {val}")


def report(key: str, times_ms: list):
    t = np.array(times_ms)
    save(key, {"n_images": len(t), "mean_ms": round(t.mean(), 2),
               "median_ms": round(float(np.median(t)), 2), "std_ms": round(t.std(), 2),
               "p90_ms": round(float(np.percentile(t, 90)), 2), "fps": round(1000 / t.mean(), 1)})


def val_files(val_dir: Path, n):
    files = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if n and n < len(files):
        idx = np.linspace(0, len(files) - 1, n).astype(int)  # spread over the whole set
        files = [files[i] for i in idx]
    return files


def load_letterboxed(path: Path) -> np.ndarray:
    """Aspect-preserving resize so the longest side is <=800, then pad to
    exactly 800x800 with value 114 (Ultralytics convention)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(IMGSZ / w, IMGSZ / h)
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        arr = np.asarray(im)
    h, w = arr.shape[:2]
    if (h, w) != (IMGSZ, IMGSZ):
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=arr.dtype)
        canvas[:h, :w] = arr
        arr = canvas
    return arr


def timed_loop(files, prepare, call, key):
    """Warm up on repeats of the first image (cudnn autotune + clock boost),
    then time call(x) for EVERY val image with CUDA sync around each call."""
    times = []
    with torch.inference_mode():
        x0 = prepare(files[0])
        for _ in range(WARMUP):
            call(x0)
            torch.cuda.synchronize()
        for f in files:
            x = prepare(f)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    report(key, times)


def stage_flops(ds: str, cfg: dict):
    torch.set_num_threads(8)
    m, ck_params = build_maskrcnn(cfg["mr_ckpt"], cfg["nc"])
    p = sum(q.numel() for q in m.parameters())
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(m, ([torch.rand(3, IMGSZ, IMGSZ)],))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        macs = fca.total() / 1e9
    except ImportError:
        print("fvcore not installed -- Mask R-CNN GFLOPs skipped (pip install fvcore)")
        macs = None
    save(f"{ds}_MaskRCNN_stats", {
        "params_M": round(p / 1e6, 2), "ckpt_params_M": round(ck_params / 1e6, 2),
        "MACs_G": round(macs, 1) if macs else None,
        "GFLOPs_800": round(2 * macs, 1) if macs else None})

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    for name, key in [("YOLO11m", "y11_ckpt"), ("YOLO26m", "y26_ckpt")]:
        y = YOLO(str(cfg[key]))
        raw = torch.load(str(cfg[key]), map_location="cpu", weights_only=False)
        raw_model = raw.get("ema") or raw.get("model")  # EMA is what YOLO() loads
        ck_p = sum(q.numel() for q in raw_model.parameters())
        net = y.model.float().eval()
        p_unfused = sum(q.numel() for q in net.parameters())
        f800 = get_flops(net, IMGSZ)  # already GFLOPs (2*MACs)
        f640 = get_flops(net, 640)
        net_f = y.model.fuse().float().eval()
        p_fused = sum(q.numel() for q in net_f.parameters())
        f800_fused = get_flops(net_f, IMGSZ)
        save(f"{ds}_{name}_stats", {
            "params_M": round(p_unfused / 1e6, 2), "ckpt_params_M": round(ck_p / 1e6, 2),
            "params_fused_M": round(p_fused / 1e6, 2),
            "GFLOPs_800": round(f800, 1), "GFLOPs_800_fused": round(f800_fused, 1),
            "GFLOPs_640": round(f640, 1)})


def stage_mrcnn(ds: str, cfg: dict, n):
    m, _ = build_maskrcnn(cfg["mr_ckpt"], cfg["nc"])
    m.to(DEV)

    def prepare(f):
        return torch.from_numpy(load_letterboxed(f)).permute(2, 0, 1).float().div_(255).to(DEV)

    timed_loop(val_files(cfg["val"], n), prepare, lambda x: m([x]), f"{ds}_MaskRCNN_latency")


def stage_yolo(ds: str, name: str, ckpt: Path, nc: int, val_dir: Path, n):
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    y = YOLO(str(ckpt))
    net = y.model.fuse().float().to(DEV).eval()
    end2end = bool(getattr(net, "end2end", False))
    print(f"{name}: fused, end2end={end2end} (NMS {'not needed' if end2end else 'included in timing'})")

    def prepare(f):
        return torch.from_numpy(load_letterboxed(f)).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(DEV)

    def call(x):
        p = net(x)
        if not end2end:
            non_max_suppression(p[0] if isinstance(p, (list, tuple)) else p,
                                conf_thres=0.25, iou_thres=0.7, nc=nc - 1, max_det=300)

    timed_loop(val_files(val_dir, n), prepare, call, f"{ds}_{name}_latency")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--stage", required=True, choices=["flops", "y11", "y26", "mrcnn"])
    ap.add_argument("-n", type=int, default=0, help="images to time (0 = full val set)")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    torch.backends.cudnn.benchmark = True
    n = args.n if args.n > 0 else None
    if args.stage == "flops":
        stage_flops(args.dataset, cfg)
    elif args.stage == "y11":
        stage_yolo(args.dataset, "YOLO11m", cfg["y11_ckpt"], cfg["nc"], cfg["val"], n)
    elif args.stage == "y26":
        stage_yolo(args.dataset, "YOLO26m", cfg["y26_ckpt"], cfg["nc"], cfg["val"], n)
    elif args.stage == "mrcnn":
        stage_mrcnn(args.dataset, cfg, n)
    print("STAGE_DONE")
