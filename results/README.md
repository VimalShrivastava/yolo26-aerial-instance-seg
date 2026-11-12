# Result summaries

Small text/JSON/CSV outputs of the original runs. Machine-specific root directories inside the files were replaced by `<root>`; no number was changed. Large artefacts (prediction JSONs, ground-truth JSONs, checkpoints) are not stored here.

| File | Produced by | Used for |
|---|---|---|
| `isaid/size_ap_isaid_summary.csv` | `scripts/eval_size_ap_isaid.py` | iSAID overall and size-bucketed AP (box and mask). The paper tables use the rows with `scheme = coco-default` |
| `isaid/size_ap_isaid_classwise.csv` | `scripts/eval_size_ap_isaid.py` | iSAID per-class AP / AP50 / AP75 (`maxDets = 300`) |
| `isaid/all_vs_annotated_patches.csv` | `scripts/eval_all_patches_isaid.py` | Mask and box AP (overall, AP50/75, size buckets, per class, `maxDets = 300`) on the 5,541 annotated validation patches (`scope = ann`, reproduces the paper tables) and on all 9,581 patches (`scope = all`) |
| `isaid/<run>/results.csv`, `args.yaml` | Ultralytics (training run) | training curves and the arguments each run actually used |
| `isaid/maskrcnn_train_loss_per_epoch.csv` | read from the saved `epoch_N.pth` checkpoints | Mask R-CNN training loss per epoch (the original script printed but did not log it) |
| `nwpu/nwpu_size_ap_results_coco_default.csv` | `scripts/eval_size_ap_nwpu.py` | NWPU overall and size-bucketed AP; the paper tables use this file |
| `nwpu/nwpu_size_ap_results.csv` | `scripts/eval_size_ap_nwpu.py` | same, with the custom size buckets (small = 10–144 px²) |
| `nwpu/nwpu_size_ap_classwise.csv` | `scripts/eval_size_ap_nwpu.py` | NWPU per-class AP |
| `nwpu/*_summary_metrics.json` | `scripts/evaluate_*_nwpu.py` | Ultralytics / pycocotools summary per model on the NWPU validation split |
| `nwpu/<run>/results.csv`, `args.yaml` | Ultralytics (training run) | training curves and arguments |
| `efficiency/bench_results.json` | `scripts/benchmark_efficiency.py` | parameters, GFLOPs at 800 × 800, latency and FPS (batch 1, FP32, RTX 3090) |

`N/A` in a size-bucketed column means the bucket contains no ground-truth instances.
