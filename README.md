# DR-Grading

5-class diabetic retinopathy grading (ICDR 0-4) on fundus photos, with mtmDR / vtDR evaluation
compared against the results of the original paper.

## Layout

```
train_dr.py          training / fine-tuning entry point
eval_mtm_vt.py       mtmDR / vtDR evaluation of a checkpoint
model/               one file per model family; model/__init__.py = registry (--model) + build_model()
  retfound_mae.py    RETFound_mae (ViT-L/16, MAE)
  dinov2.py          Dinov2, RETFound_dinov2 (timm DINOv2 ViTs)
  dinov3.py          Dinov3 (torch.hub)
  timm_model.py      Timm (any timm model, e.g. Swin V2-L)
  hybrid.py          Hybrid (EfficientNet-B0 + Swin V2-T concat)
  swin_ms.py         SwinMS (Swin V2 with pooled features of several stages concatenated)
  pretrained.py      load_pretrained(): --finetune weights (HF hub / local)
utils/
  engine.py          train_one_epoch / evaluate
  datasets.py        ImageFolder datasets, data-ratio subsampling
  transforms.py      CLAHE, timm default and fundus augmentation
  metrics.py         binary metrics, bootstrap CI, Youden threshold, report numbers
  lr_decay.py lr_sched.py misc.py pos_embed.py
tools/
  prepare_dr_data.py   raw handover data -> Fundus_Dataset/DR5/{train,val,test}/{0..4} + manifest.csv
  preprocess_fundus.py crop retinal disc, pad square, resize 512 -> Fundus_Dataset/DR5_512
  svm_dr.py            SVM on backbone features (optional)
```

## Setup

```
pip install -r requirements.txt   # plus torch / torchvision for your CUDA version
```

All commands run from the repo root. Windows / single GPU: call `python train_dr.py` directly (no torchrun).
If the console shows encoding errors, set `PYTHONIOENCODING=utf-8`.

## 0. Data

```
python tools/prepare_dr_data.py        # split txt -> Fundus_Dataset/DR5/{train,val,test}/{0..4} + manifest.csv
python tools/preprocess_fundus.py      # crop fundus disc, pad square, resize 512 -> Fundus_Dataset/DR5_512
```

`Fundus_Dataset/DR5_512_prosval` = train/test junctions to `DR5_512` + val = prospective val only (281 images).
This is the `--data_path` used below.

## 1. Training

### Swin V2-L 384 + CLAHE

```
python train_dr.py --model Timm --model_arch swinv2_large_window12to24_192to384 --data_path Fundus_Dataset/DR5_512_prosval --nb_classes 5 --input_size 384 --clahe --grad_ckpt --batch_size 4 --accum_iter 4 --epochs 20 --warmup_epochs 2 --blr 1e-3 --layer_decay 0.9 --drop_path 0.2 --num_workers 4 --task DR5_swinv2L_384_clahe_v2_ep20 --adaptation finetune
```

- RTX 4070 12 GB: batch 4 + `--grad_ckpt` (~5.4 GB, ~6 img/s, ~20 min/epoch). Batch 8 spills out of VRAM -> very slow.
- `--accum_iter 4` -> effective batch 16. Smaller/faster: `--model_arch swinv2_base_window12to24_192to384 --batch_size 8 --accum_iter 2`.

### SwinMS: Swin V2-L with pooled features of several stages concatenated

```
python train_dr.py --model SwinMS --model_arch swinv2_large_window12to24_192to384 --ms_stages 0 1 2 3 --data_path Fundus_Dataset/DR5_512_prosval --nb_classes 5 --input_size 384 --clahe --grad_ckpt --batch_size 4 --accum_iter 4 --epochs 20 --warmup_epochs 2 --blr 1e-3 --layer_decay 0.9 --drop_path 0.2 --num_workers 4 --task DR5_swinv2L_384_clahe_ms0123_ep20 --adaptation finetune
```

- Plain Swin V2 classifies from the last stage only; SwinMS also pools stages 1-3 (higher resolution,
  small lesions such as micro-aneurysms), each with its own LayerNorm, concatenates (2880-d for all 4) -> Linear.
- `--ms_stages 1 2 3` drops stage 0; `--ms_head_dropout 0.2` adds dropout before the head.
- Backbone parameter names are the same as timm Swin V2, so layer-wise LR decay still applies.

### Hybrid EfficientNet-B0 + Swin V2-T

```
python train_dr.py --model Hybrid --data_path Fundus_Dataset/DR5_512_prosval --nb_classes 5 --input_size 384 --clahe --batch_size 16 --epochs 20 --warmup_epochs 2 --blr 2.5e-3 --drop_path 0.2 --num_workers 4 --task DR5_hybrid_effb0_swinT_384_clahe_v1 --adaptation finetune
```

- `--layer_decay` is ignored (uniform LR); `--grad_ckpt` not supported/needed.

### RETFound (ViT-L DINOv2, shanghai weights) 224

```
python train_dr.py --model RETFound_dinov2 --model_arch retfound_dinov2 --finetune RETFound_dinov2_shanghai --data_path Fundus_Dataset/DR5_512_prosval --nb_classes 5 --input_size 224 --batch_size 16 --epochs 20 --warmup_epochs 3 --blr 2.5e-3 --layer_decay 0.5 --drop_path 0.3 --num_workers 4 --task DR5_shanghai_224_v4_ld05_dp03 --adaptation finetune
```

- 448: add `--input_size 448 --grad_ckpt` (batch 16 ~6 GB).

### `--model` / `--model_arch`

`--model` is required (choices below). `--model_arch` is only used by Timm / SwinMS / Dinov2 / Dinov3.
Both are saved in the checkpoint, so `eval_mtm_vt.py` does not need them.
`python train_dr.py --help` prints the same table.

| `--model` | `--model_arch` | pre-trained weights |
|---|---|---|
| `Timm` | **required**: any timm name, e.g. `swinv2_large_window12to24_192to384`, `convnext_base`, `efficientnet_b3` | timm ImageNet (automatic) |
| `Dinov2` | **required**: `dinov2_vits14` / `dinov2_vitb14` / `dinov2_vitl14` / `dinov2_vitg14` | `--finetune <local .pth>` |
| `Dinov3` | **required**: torch.hub entry, e.g. `dinov3_vits16`, `dinov3_vitl16` | `--finetune <local .pth>` |
| `RETFound_dinov2` | ignored (ViT-L/14) | `--finetune RETFound_dinov2_shanghai` / `RETFound_dinov2_meh` (HF hub) |
| `RETFound_mae` | ignored (ViT-L/16) | `--finetune RETFound_mae_natureCFP` / `_meh` / `_shanghai` / `_natureOCT` (HF hub) |
| `Hybrid` | ignored (EfficientNet-B0 + Swin V2-T) | torchvision ImageNet (automatic) |
| `SwinMS` | **required**: timm Swin V2 name, e.g. `swinv2_large_window12to24_192to384` | timm ImageNet (automatic) |

### Project-specific options

| option | meaning |
|---|---|
| `--model Timm --model_arch <timm name>` | any timm model with its ImageNet weights (swinv2_*, convnext_*, efficientnet_*, ...) |
| `--model Hybrid` | EfficientNet-B0 + Swin V2-T concat |
| `--model SwinMS --ms_stages 0 1 2 3 --ms_head_dropout 0.0` | Swin V2 multi-stage pooled features (stages to concat, dropout before head) |
| `--clahe` | CLAHE on Y channel (clip 4, 8x8) before all transforms, identical to deployment |
| `--aug_mode fundus` | mild fundus augmentation (rotation, 70-100% crop, flips, light colour jitter); default = timm RandAugment |
| `--grad_ckpt` | gradient checkpointing (timm models): less VRAM, slower |
| `--no_tqdm` | default is a tqdm progress bar per epoch / eval; this switches back to a text line every 20 (train) / 10 (eval) iters (better when redirecting to a log file). Epoch summaries are printed either way |
| `--seed N` | default 0; GPU runs are close but not bit-identical (cudnn.benchmark, AMP) |

## 2. Evaluation (mtmDR / vtDR, compared with the original paper)

```
python eval_mtm_vt.py --resume output_dir/<task>/checkpoint-best.pth --data_path Fundus_Dataset/DR5_512_prosval
```

- Model type, input size, CLAHE etc. are read from the checkpoint's saved args.
- Prints accuracy / sensitivity / specificity (with bootstrap 95% CI, overall and per device) next to the original paper's numbers.
- Large models (Swin V2-L): add `--batch_size 16`.
- Writes `predictions_val.csv`, `predictions_test.csv`, `mtm_vt_metrics.csv` next to the checkpoint (or `--out_dir`).

## 3. SVM on backbone features (optional)

```
python tools/svm_dr.py --resume output_dir/<task>/checkpoint-best.pth --data_path Fundus_Dataset/DR5_512_prosval
```

- Features cached in `<checkpoint folder>/svm/features_*.npz` (GPU only needed the first time).

## 4. Output locations

| what | default | change with |
|---|---|---|
| checkpoint-best.pth, log.txt, metrics_*.csv, confusion_matrix_test.jpg | `./output_dir/<task>/` | `--output_dir <path>` (train_dr.py) |
| TensorBoard logs | `./output_logs/<task>/` | `--log_dir <path>` (train_dr.py) |
| predictions_*.csv, mtm_vt_metrics.csv | folder of `--resume` | `--out_dir <path>` (eval_mtm_vt.py) |
| svm results / feature cache | `<folder of --resume>/svm/` | `--out_dir <path>` (svm_dr.py) |
