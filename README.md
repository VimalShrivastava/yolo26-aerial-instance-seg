# YOLO26 for Instance Segmentation in Remotely Sensed Images

Code, configurations and data splits for the paper

> V. Srivastava, V. K. Shrivastava, S. Bera. *A Comprehensive Evaluation of YOLO26 for Instance Segmentation in Remotely Sensed Images.* Submitted to *Pattern Analysis and Applications*, 2026.

The repository reproduces every table in the paper: YOLO26m, YOLO11m and Mask R-CNN (ResNet-50-FPN) trained and evaluated on iSAID and NWPU VHR-10 with the same data splits, the same 800 × 800 inputs and the same COCO evaluation code.

## Version

The code and configurations behind the numbers in the paper are the GitHub release **`v1.0.0`** of this repository. Later commits may differ; use the `v1.0.0` tag to reproduce the paper.

## Environment

All experiments ran on Windows 11 with one NVIDIA GeForce RTX 3090 (24 GB), CUDA 11.8.

| Package | Version |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.1.0 (+cu118) |
| torchvision | 0.16.0 (+cu118) |
| Ultralytics | 8.4.33 |
| pycocotools | 2.0.11 |
| NumPy / OpenCV | 1.26.4 / 4.13.0.92 |

```bash
conda create -n yolo26-rs python=3.9 -y
conda activate yolo26-rs
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

If a script's output is redirected on Windows and fails on a `→` or `✅` character, set `PYTHONIOENCODING=utf-8`.

## Data

The images are not redistributed here; download them from the original providers.

* **iSAID**: https://captain-whu.github.io/iSAID (annotations) with the DOTA-v1.0 images. Tiling uses 800 × 800 px patches with a 200 px overlap (stride 600 px), implemented in `scripts/isaid_02_tile_patches.py`.
* **NWPU VHR-10**: https://www.kaggle.com/datasets/larbisck/nwpu-vhr-10, with the COCO-format instance masks released by Garg et al. (ISDNet, 2021) as `annotations.json`.

`splits/` contains the exact file lists used in the paper (no images):

| File | Content |
|---|---|
| `splits/nwpu_vhr10/train.txt`, `val.txt` | NWPU VHR-10 80:20 split, 520 / 130 images |
| `splits/nwpu_vhr10/split_seed42.json` | the same split as image ids, plus the method |
| `splits/isaid/{train,val}_source_images.txt` | the official iSAID train / val images (1,411 / 458) |
| `splits/isaid/{train,val}_patches_all.txt` | every 800 px patch produced by the tiling (28,249 / 9,581) |
| `splits/isaid/{train,val}_patches_annotated.txt` | patches with at least one annotated instance (16,857 / 5,541) |

The NWPU split is `random.seed(42)`; `random.shuffle` over the image ids of `annotations.json`; the first `int(0.8 * 650) = 520` ids are the training set. **Seed: 42.**

Only the annotated patches are used. `isaid_03_coco_to_yolo.py` drops patches without annotations, and the Mask R-CNN dataset class only loads patches that have annotations. All iSAID results in the paper tables are therefore computed on the 5,541 annotated validation patches, and YOLO training uses the 16,857 annotated training patches. `scripts/eval_all_patches_isaid.py` also scores all 9,581 validation patches (including the 4,040 background-only ones) with the same checkpoints; see `results/isaid/all_vs_annotated_patches.csv`.

### Expected directory layout

Every script takes its dataset root from `--root <dir>` (or the environment variable `ISAID_ROOT` / `NWPU_ROOT`).

```
<iSAID root>/
  Dataset/DOTA_train/part{1,2,3}/images/   DOTA-v1.0 images
  Dataset/DOTA_val/part{1,2}/images/
  Dataset/train/Annotations/iSAID_train.json
  Dataset/val/Annotations/iSAID_val.json
  Dataset_patches/        created by isaid_02   (800 px patches + COCO json)
  yolo26_dataset/         created by isaid_03   (YOLO-seg labels + isaid.yaml)
  work_dirs/paper/        training and evaluation outputs

<NWPU root>/
  dataset/annotations.json
  dataset/positive image set/*.jpg
  yolo_dataset/           created by nwpu_prepare_split (+ train_val_split.json)
  work_dirs/paper/
```

## Reproducing the results

Run from the repository root. `$ISAID` and `$NWPU` are the two dataset roots.

| Step | Command |
|---|---|
| 1. Merge DOTA images into iSAID train/val folders | `python scripts/isaid_01_merge_dota_images.py --root $ISAID` |
| 2. Tile iSAID into 800 × 800 patches (200 px overlap) | `python scripts/isaid_02_tile_patches.py --root $ISAID` |
| 3. Convert iSAID COCO → YOLO-seg | `python scripts/isaid_03_coco_to_yolo.py --root $ISAID` |
| 4. NWPU VHR-10 → YOLO-seg, 80:20 split (seed 42) | `python scripts/nwpu_prepare_split.py --root $NWPU` |
| 5. Train YOLO26m / YOLO11m | `python scripts/train_yolo.py --root $ISAID --config configs/yolo26m_isaid.yaml` (also `yolo11m_isaid`, `yolo26m_nwpu`, `yolo11m_nwpu`; use `--root $NWPU` for NWPU) |
| 6. Train Mask R-CNN | `python scripts/train_maskrcnn_isaid.py --root $ISAID` and `python scripts/train_maskrcnn_nwpu.py --root $NWPU` |
| 7. Evaluate on iSAID | `python scripts/evaluate_yolo26_isaid.py --root $ISAID`, `python scripts/evaluate_yolo11_isaid.py --root $ISAID`, then `python scripts/eval_size_ap_isaid.py --root $ISAID`; optional check on all 9,581 patches: `python scripts/eval_all_patches_isaid.py --root $ISAID` |
| 8. Evaluate on NWPU | `python scripts/evaluate_yolo26_nwpu.py --root $NWPU`, `evaluate_yolo11_nwpu.py`, `evaluate_maskrcnn_nwpu.py`, then `python scripts/eval_size_ap_nwpu.py --root $NWPU` |
| 9. Parameters, GFLOPs, latency | `python scripts/benchmark_efficiency.py --isaid-root $ISAID --nwpu-root $NWPU --dataset isaid --stage flops` (stages `flops`, `y11`, `y26`, `mrcnn`; repeat with `--dataset nwpu`) |

Notes:

* `isaid_03_coco_to_yolo.py` deletes and rebuilds `yolo26_dataset/images/<split>` and `labels/<split>`. It refuses to run when they already contain files unless you pass `--overwrite`.
* `eval_size_ap_isaid.py` produces the iSAID tables (overall, size-bucketed and per-class AP). It scores all three models against one ground-truth file built from `yolo26_dataset/labels/val` and runs the Mask R-CNN inference itself. `eval_size_ap_nwpu.py` does the same for NWPU from the three `evaluate_*_nwpu.py` outputs.
* **COCO protocol details.** The evaluation scripts report two size-bucket schemes, a custom one and COCO's own; the paper tables use COCO's (small < 32², medium 32²–96², large > 96² px² of mask area). On iSAID `maxDets` is raised from 100 to 300 (iSAID patches are far denser than COCO images; 131 of the 5,541 annotated validation patches contain more than 100 instances of a single category), and the same cap applies to every model. On NWPU the pycocotools default `maxDets = 100` is used.
* The latency stage should be run on an otherwise idle machine. CPU contention from other programs inflated the measured latency by up to 1.7× on the workstation used here (see `results/efficiency/bench_results.json`).
* The latencies reported in the paper were averaged over 5,511 timed images; the current script times all 5,541 annotated validation patches. A difference of 30 images has a negligible effect on the averaged latency.
* Ultralytics checkpoints store the weights under the `ema` key; `benchmark_efficiency.py` already accounts for that.

### Training settings

Exactly as run; the machine-readable versions are in `configs/`, and the arguments Ultralytics recorded for each run are in `results/*/<run>/args.yaml`.

| | YOLO11m | YOLO26m | Mask R-CNN (R50-FPN) |
|---|---|---|---|
| Initialisation | COCO pretrained | COCO pretrained | COCO pretrained (torchvision `DEFAULT`) |
| Optimizer | SGD | `auto` → MuSGD | SGD |
| Initial LR / momentum / weight decay | 0.01 / 0.937 / 5e-4 | 0.01 / 0.937 / 5e-4 | 0.005 / 0.9 / 5e-4 (iSAID), 1e-4 (NWPU) |
| Epochs | ≤ 100 (early stop, patience 20) | ≤ 100 (early stop, patience 20) | 12 (LR ÷ 10 after epochs 8 and 11) |
| Input size | 800 | 800 | torchvision default transform (`min_size=800`, `max_size=1333`); full 800 × 800 patches are unchanged, smaller border patches are scaled up |
| Batch size | 4 | 4 | 4 (iSAID), 2 (NWPU) |
| Seed | 0 (Ultralytics default, `deterministic: true`) | 0 | not set (only the NWPU split is seeded, with 42) |

Other YOLO settings were left at their Ultralytics defaults, among them `mosaic: 1.0`, `close_mosaic: 10`, `amp: true`, `lrf: 0.01`, `warmup_epochs: 3`.
On iSAID the YOLO11m run completed all 100 epochs (best epoch 100). The YOLO26m run stopped after epoch 98 of 100 (best epoch 96) and the reported weights are its `best.pt`; see `results/isaid/*/results.csv`.

**Optimizer note.** With `optimizer="auto"`, Ultralytics 8.4.33 logs that it ignores `lr0` and `momentum` and picks MuSGD with `lr = 0.01` and `momentum = 0.9`. The warm-up loop then writes `args.momentum` into every parameter group, so the saved optimizer state of the iSAID YOLO26m checkpoint (`last.pt`, epoch 97) shows momentum 0.937 and weight decay 5e-4 on all groups. The effective settings therefore match the YOLO11m run.

## Main results (mask AP, %)

| Model | iSAID AP | iSAID AP50 | NWPU AP | NWPU AP50 |
|---|---|---|---|---|
| Mask R-CNN | 38.84 | 64.59 | 66.19 | 92.82 |
| YOLO11m | 43.72 | 71.95 | 70.30 | 94.34 |
| YOLO26m | 43.62 | 71.95 | 71.24 | 93.62 |

Small result files (per-class and size-bucketed AP, efficiency benchmark, training curves, recorded training arguments) are in `results/`; see `results/README.md`.

## Trained weights

The six checkpoints behind the paper's tables are attached to the GitHub release `v1.0.0` (they are not stored in the git history). Checksums are in `SHA256SUMS.txt` in the release.

| File | Model | Dataset |
|---|---|---|
| `yolo26m_isaid_best.pt` | YOLO26m-seg (Ultralytics `best.pt`, epoch 96) | iSAID |
| `yolo11m_isaid_best.pt` | YOLO11m-seg (Ultralytics `best.pt`, epoch 100) | iSAID |
| `maskrcnn_isaid_best_model.pth` | Mask R-CNN R50-FPN (torchvision state dict, epoch 12) | iSAID |
| `yolo26m_nwpu_best.pt` | YOLO26m-seg (Ultralytics `best.pt`) | NWPU VHR-10 |
| `yolo11m_nwpu_best.pt` | YOLO11m-seg (Ultralytics `best.pt`) | NWPU VHR-10 |
| `maskrcnn_nwpu_best_model.pth` | Mask R-CNN R50-FPN (torchvision state dict) | NWPU VHR-10 |

Place them under `<root>/work_dirs/paper/...` as in the layout above (for example `yolo26/yolo26m_isaid_musgd/weights/best.pt`) to run the evaluation and benchmark scripts without retraining.

## Citation

See `CITATION.cff`. This will be updated with the journal reference once the paper is published.

## Acknowledgements

Supported by the Indian Space Research Organisation (ISRO) under the RESPOND Programme (Grant No. ISRO/RES/4/689/21-22).

## License

This repository is released under the GNU Affero General Public License v3.0 (see `LICENSE`), the license of Ultralytics, on which the YOLO code depends.
