# DR-Grading

5-class diabetic retinopathy grading (ICDR 0-4) on fundus photos, with mtmDR / vtDR evaluation
compared against the results of the original paper.

## Layout

```
train_dr.py          training / fine-tuning entry point
eval_mtm_vt.py       mtmDR / vtDR evaluation of a checkpoint
eval_mtm_vt_postprocess.py  predictions_*.csv + saved detections -> deployed post-processing (ALFA) -> mtmDR / vtDR (CPU)
postprocess/         unmodified copy of the deployed post-processing v2.1.9 (ALFA tables, lesion thresholds); see its README
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
  datasets.py        ImageFolder datasets, device filter (--train_devices / --val_devices), data-ratio subsampling
  transforms.py      CLAHE, timm default and fundus augmentation
  metrics.py         binary metrics, bootstrap CI, Youden threshold, report numbers
  lr_decay.py lr_sched.py misc.py pos_embed.py
tools/
  prepare_dr_data.py   raw handover data -> Fundus_Dataset/DR5/{train,val,test}/{0..4} + manifest.csv
  preprocess_fundus.py crop retinal disc, pad square, resize 512 -> Fundus_Dataset/DR5_512
  dedup_dataset.py     de-duplicated copy of a dataset (problem*.txt lists)
  build_valpros_dataset.py  DR5_512 -> DR5_512_valpros_dedup (val = prospective test, de-duplicated train)
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

`Fundus_Dataset/DR5_512_valpros_dedup` (used by the best model below) — 7938 images, ~1.8 GB, real files (no junctions):

```
python tools/build_valpros_dataset.py --src Fundus_Dataset/DR5_512 --dst Fundus_Dataset/DR5_512_valpros_dedup --problems D:/data/fundus_yolo/problem.txt D:/data/fundus_yolo/problem2.txt
```

- train = retro train + prospective train (duplicates removed), **val = prospective test (466)**, test = retro test (1181).
- Its `manifest.csv` (columns `image`, `device`, `cohort`, ...) is required by `--train_devices` / `--val_devices`
  and by the per-device rows of `eval_mtm_vt.py`; copy it together with the images.

## 1. Training

### SwinMS, desktop only (Canon + Nidek) — best so far

```
python train_dr.py --model SwinMS --model_arch swinv2_large_window12to24_192to384 --ms_stages 0 1 2 3 --data_path Fundus_Dataset/DR5_512_valpros_dedup --train_devices Canon Nidek --val_devices Canon Nidek --nb_classes 5 --input_size 384 --clahe --grad_ckpt --batch_size 4 --accum_iter 4 --epochs 10 --warmup_epochs 1 --blr 1e-3 --layer_decay 0.9 --drop_path 0.2 --num_workers 4 --task DR5_swinMS_desktop_v1 --adaptation finetune
python eval_mtm_vt.py --resume output_dir/DR5_swinMS_desktop_v1/checkpoint-best.pth --data_path Fundus_Dataset/DR5_512_valpros_dedup --batch_size 16 --report_split val
```

- Trains on 5381 desktop images (retro Canon + pros Canon + Nidek); JYE01 is never seen. Best epoch 6.
- Original run (RETFound repo), prospective test 466, argmax: mtmDR sens 0.844 / spec 0.968 (JYE01 0.786 / 0.958),
  vtDR 0.955 / 0.998. With `--tta flip4`: mtmDR sens 0.856.
- No-random-crop variant: add `--crop_scale 1 1 --crop_ratio 1 1`.
- Other input sizes (Swin archs, `--model Timm` / `SwinMS`): `--input_size 512 --window_size 16` (or 32), `448` -> 14 or 28.
  Every stage's token grid must be divisible by the window. RTX A5000, batch 4 + `--grad_ckpt`: 512/w16 5.2 GB ~9 img/s,
  512/w32 10.4 GB ~4 img/s, 448/w28 7.4 GB ~6 img/s (384/w24: 5.4 GB ~10 img/s). The data is 512, so > 512 only upsamples.

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
| `Timm` | **required**: any timm name, e.g. `swinv2_large_window12to24_192to384`, `convnext_base`, `efficientnet_b3` | timm ImageNet (automatic), or `--finetune <our checkpoint-best.pth>` |
| `Dinov2` | **required**: `dinov2_vits14` / `dinov2_vitb14` / `dinov2_vitl14` / `dinov2_vitg14` | `--finetune <local .pth>` |
| `Dinov3` | **required**: torch.hub entry, e.g. `dinov3_vits16`, `dinov3_vitl16` | `--finetune <local .pth>` |
| `RETFound_dinov2` | ignored (ViT-L/14) | `--finetune RETFound_dinov2_shanghai` / `RETFound_dinov2_meh` (HF hub) |
| `RETFound_mae` | ignored (ViT-L/16) | `--finetune RETFound_mae_natureCFP` / `_meh` / `_shanghai` / `_natureOCT` (HF hub) |
| `Hybrid` | ignored (EfficientNet-B0 + Swin V2-T) | torchvision ImageNet (automatic) |
| `SwinMS` | **required**: timm Swin V2 name, e.g. `swinv2_large_window12to24_192to384` | timm ImageNet (automatic), or `--finetune <our checkpoint-best.pth>` |

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
| `--train_devices JYE01 ...` / `--val_devices ...` | keep only these devices in TRAIN / VAL (checkpoint selection), from `<data_path>/manifest.csv`; TEST is never filtered |
| `--crop_scale MIN MAX` / `--crop_ratio MIN MAX` | RandomResizedCrop range for `--aug_mode default` (timm default 0.08 1 / 0.75 1.333); `1 1` + `1 1` = no random crop |
| `--seed N` | default 0; GPU runs are close but not bit-identical (cudnn.benchmark, AMP) |
| `--balance none\|device_class` | TRAIN sampling. `device_class` = WeightedRandomSampler (with replacement, same epoch length), weight p(class)/p(class \| device), so every device gets the class distribution of the whole TRAIN set while keeping its share of samples (JYE01 train is 77% grade 0). Single GPU only |
| `--select_metric mtm_bacc_auc\|score` | VAL metric for `checkpoint-best.pth`. Default `mtm_bacc_auc` = 0.5·(mtmDR sens+spec)/2 at argmax + 0.5·mtmDR AUC (P2+P3+P4). `score` = (macro F1 + macro AUC + kappa)/3, used by all runs before 2026-11-21 (e.g. DR5_swinMS_desktop_v1): add it to reproduce them. Both are written to `metrics_val.csv` every epoch |
| `--finetune <checkpoint-best.pth>` (Timm / SwinMS) | Start from one of our own checkpoints instead of ImageNet, e.g. two-stage training: stage 1 on APTOS 2019 (`--data_path .../APTOS2019_512 --train_devices APTOS --val_devices APTOS`), stage 2 on DR5_512 with `--finetune <stage-1 checkpoint-best.pth>`. Weights only (optimizer / epoch are not restored, unlike `--resume`); `--model`, `--model_arch`, `--ms_stages`, `--input_size`, `--window_size` must match the checkpoint |
| `--finetune_head keep\|reinit` | With `--finetune`: keep the checkpoint's classifier head (default, if same `--nb_classes`) or re-initialise it |

## 2. Evaluation (mtmDR / vtDR, compared with the original paper)

```
python eval_mtm_vt.py --resume output_dir/<task>/checkpoint-best.pth --data_path Fundus_Dataset/DR5_512_prosval
```

- Model type, input size, CLAHE etc. are read from the checkpoint's saved args.
- Prints accuracy / sensitivity / specificity (with bootstrap 95% CI, overall and per device) next to the original paper's numbers.
- Large models (Swin V2-L): add `--batch_size 16`.
- `--report_split val` for `DR5_512_valpros_dedup` (its val split holds the prospective test set); only argmax is reported then.
- `--tta hflip|flip4`: average the softmax over flipped views; results go to `<checkpoint folder>/tta_<mode>/`.
- Writes `predictions_val.csv`, `predictions_test.csv`, `mtm_vt_metrics.csv` next to the checkpoint (or `--out_dir`).

## 2b. Evaluation with the deployed post-processing (classifier + detection, ALFA)

```
python eval_mtm_vt_postprocess.py --pred_csv output_dir/<task>/predictions_val.csv --det_root D:/exp_result/fundus/Det_Raw/det_raw_0JCN --age 40
```

- CPU only, no torch (numpy + pandas), a few seconds for 466 images; safe to run while training.
- `--pred_csv`: `predictions_{val,test}.csv` from `eval_mtm_vt.py` (Cls = argmax, cls_prob = max of p0..p4).
  `--det_root`: raw detections `<det_root>/<device>/raw/<image stem>.json` (all boxes of the detection service,
  cropped/padded pixels + `offset`; e.g. `D:\exp_result\fundus\Det_Raw\det_raw_0JCN` = jye weights for all devices).
- Per image the deployed `postprocess()` is called unchanged: IoM-NMS 0.5 -> boxes to the original image ->
  `ALFA.contradict_DR5` -> lesion thresholds. Not reproduced: the API quality filter and spot-light elimination.
- `--age` (default 40 = API default) selects the type-0 lesion thresholds from `postprocess/threshold_config.json`
  (40-50: SE 0.4 / HE 0.45 ...; >=50: SE 0.24 / HE 0.324 ...).
- `--threshold_config <json>`: other lesion thresholds (same format); omitted = `postprocess/threshold_config.json` (deployed).
  `postprocess/threshold_config_tuned.json` differs only in type 0, age >= 50: IH (2) 0.2 -> 0.08, HE (8) 0.324 -> 0.06,
  LS (12) 0.359 -> 0.08, so it changes results only with `--age` >= 50.
- `--type_map JYE01=0,Canon=0,Nidek=0` (default): postprocess type per device (0 = JYE01 tables + age thresholds,
  1 = desktop tables). Use the detections of the same weights, e.g. `--det_root .../det_raw_0J_1CN --type_map JYE01=0,Canon=1,Nidek=1`.
- `--ci wald` (default, as the report) or `--ci bootstrap` (as `eval_mtm_vt.py`).
- Prints raw (argmax) and alfa rows, overall and per device, 5-class acc + quadratic kappa, how many images
  ALFA changed (fixed / broke). **Without `--out_dir` nothing is written** (terminal only).
- `--out_dir <path>`: also write `predictions_postprocess.csv` (per image cls_raw / det_init / cls_alfa),
  `mtm_vt_metrics_postprocess.csv`, `run_config.json` (sha256 of the `postprocess/` files).
- `--out_dir <path> --image_root D:/data/fundus_2 --save_boxes`: also write the final boxes (normalised cx cy w h,
  as the API). The image size only affects these boxes, not the grade.

## 3. SVM on backbone features (optional)

```
python tools/svm_dr.py --resume output_dir/<task>/checkpoint-best.pth --data_path Fundus_Dataset/DR5_512_valpros_dedup --tta flip4
python tools/svm_dr.py ... --svm_train_devices Canon Nidek JYE01   # SVM trained on these TRAIN devices only
bash scripts/run_svm.sh                                            # seeds 0-2 x 2 SVM training sets, + ALFA
```

- The fine-tuned backbone is frozen; the feature right before the Linear head is cached in
  `<checkpoint folder>/svm_features/features_<split>[_<tta>].npz` (GPU only needed the first time).
  `--tta flip4` averages the features over the 4 flipped views.
- `--select cv` (default): kernel / C / class_weight chosen by patient-grouped 5-fold CV inside TRAIN
  (out-of-fold mtmDR (Sens+Spec)/2); VAL and TEST are never used for a choice. `--select val` = old behaviour
  (only valid if VAL is not the test set).
- Probabilities: logistic regression on the out-of-fold SVM decision values. `predictions_val.csv` /
  `predictions_test.csv` have the `eval_mtm_vt.py` format, so `eval_mtm_vt_postprocess.py` (ALFA) runs on them.

## 4. Output locations

| what | default | change with |
|---|---|---|
| checkpoint-best.pth, log.txt, metrics_*.csv, confusion_matrix_test.jpg | `./output_dir/<task>/` | `--output_dir <path>` (train_dr.py) |
| TensorBoard logs | `./output_logs/<task>/` | `--log_dir <path>` (train_dr.py) |
| predictions_*.csv, mtm_vt_metrics.csv | folder of `--resume` | `--out_dir <path>` (eval_mtm_vt.py) |
| predictions_postprocess.csv, mtm_vt_metrics_postprocess.csv, run_config.json | not written (terminal only) | `--out_dir <path>` (eval_mtm_vt_postprocess.py) |
| svm results / feature cache | `<folder of --resume>/svm_<tta>_<select>_<devices>/`, `<folder of --resume>/svm_features/` | `--out_dir` / `--feat_dir` (svm_dr.py) |

## 5. Experimental results

**Table 1. TTA flip4, no ALFA**

| Version | mtmDR Acc | mtmDR Sens | mtmDR Spec | vtDR Acc | vtDR Sens | vtDR Spec |
|---|---|---|---|---|---|---|
| REPORT (paper) | 0.9464 | 0.9000 | 0.9574 | 0.9957 | 0.9545 | 0.9977 |
| Original repo model (DR5_swinMS_desktop_v1): ImageNet → DR5 desktop | 0.9464 | 0.8556 | **0.9681** | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → DR5 desktop (retrained, new checkpoint selection) | **0.9485** | 0.8556 | **0.9707** | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → DR5 desktop, input 512 | 0.9378 | 0.7778 | **0.9761** | 0.9936 | 0.9091 | 0.9977 |
| ImageNet → DR5 desktop + JYE01 | 0.9378 | 0.8556 | 0.9574 | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → DR5 desktop + JYE01 (device × class balanced sampling) | 0.9120 | 0.8333 | 0.9309 | 0.9914 | 0.9545 | 0.9932 |
| ImageNet → APTOS 2019 → DR5 desktop | **0.9528** | 0.8889 | **0.9681** | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → EyePACS + APTOS 2019 (5 epochs) → DR5 desktop | 0.9421 | 0.8222 | **0.9707** | 0.9914 | 0.9545 | 0.9932 |
| ImageNet → EyePACS + APTOS 2019 (10 epochs) → DR5 desktop | 0.9442 | 0.8111 | **0.9761** | 0.9914 | 0.9545 | 0.9932 |

**Table 2. TTA flip4 + ALFA (deployed lesion thresholds, age 40)**

| Version | mtmDR Acc | mtmDR Sens | mtmDR Spec | vtDR Acc | vtDR Sens | vtDR Spec |
|---|---|---|---|---|---|---|
| REPORT (paper) | 0.9464 | 0.9000 | 0.9574 | 0.9957 | 0.9545 | 0.9977 |
| Original repo model (DR5_swinMS_desktop_v1): ImageNet → DR5 desktop | 0.9421 | 0.7889 | **0.9787** | 0.9957 | 0.9091 | **1.0000** |
| ImageNet → DR5 desktop (retrained, new checkpoint selection) | 0.9421 | 0.7889 | **0.9787** | 0.9957 | 0.9091 | **1.0000** |
| ImageNet → DR5 desktop, input 512 | 0.9356 | 0.7667 | **0.9761** | 0.9957 | 0.9091 | **1.0000** |
| ImageNet → DR5 desktop + JYE01 | 0.9442 | 0.8111 | **0.9761** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → DR5 desktop + JYE01 (device × class balanced sampling) | 0.9421 | 0.7889 | **0.9787** | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → APTOS 2019 → DR5 desktop | 0.9464 | 0.8111 | **0.9787** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → EyePACS + APTOS 2019 (5 epochs) → DR5 desktop | 0.9464 | 0.8000 | **0.9814** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → EyePACS + APTOS 2019 (10 epochs) → DR5 desktop | 0.9464 | 0.8000 | **0.9814** | **0.9979** | 0.9545 | **1.0000** |

**Table 3. TTA flip4 + ALFA (deployed lesion thresholds, age 60)**

| Version | mtmDR Acc | mtmDR Sens | mtmDR Spec | vtDR Acc | vtDR Sens | vtDR Spec |
|---|---|---|---|---|---|---|
| REPORT (paper) | 0.9464 | 0.9000 | 0.9574 | 0.9957 | 0.9545 | 0.9977 |
| Original repo model (DR5_swinMS_desktop_v1): ImageNet → DR5 desktop | **0.9528** | 0.8778 | **0.9707** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → DR5 desktop (retrained, new checkpoint selection) | **0.9528** | 0.8778 | **0.9707** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → DR5 desktop, input 512 | 0.9464 | 0.8333 | **0.9734** | 0.9936 | 0.9091 | 0.9977 |
| ImageNet → DR5 desktop + JYE01 | **0.9485** | 0.8556 | **0.9707** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → DR5 desktop + JYE01 (device × class balanced sampling) | 0.9464 | 0.8667 | **0.9654** | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → APTOS 2019 → DR5 desktop | **0.9549** | 0.8889 | **0.9707** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → EyePACS + APTOS 2019 (5 epochs) → DR5 desktop | **0.9549** | 0.8556 | **0.9787** | **0.9979** | 0.9545 | **1.0000** |
| ImageNet → EyePACS + APTOS 2019 (10 epochs) → DR5 desktop | **0.9528** | 0.8556 | **0.9761** | **0.9979** | 0.9545 | **1.0000** |

**Table 4. TTA flip4 + ALFA (tuned lesion thresholds, age 60)**

| Version | mtmDR Acc | mtmDR Sens | mtmDR Spec | vtDR Acc | vtDR Sens | vtDR Spec |
|---|---|---|---|---|---|---|
| REPORT (paper) | 0.9464 | 0.9000 | 0.9574 | 0.9957 | 0.9545 | 0.9977 |
| Original repo model (DR5_swinMS_desktop_v1): ImageNet → DR5 desktop | **0.9506** | 0.9000 | **0.9628** | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → DR5 desktop (retrained, new checkpoint selection) | **0.9485** | 0.8889 | **0.9628** | 0.9957 | 0.9545 | 0.9977 |
| ImageNet → DR5 desktop, input 512 | 0.9421 | 0.8444 | **0.9654** | 0.9893 | 0.9091 | 0.9932 |
| ImageNet → DR5 desktop + JYE01 | 0.9399 | 0.8667 | 0.9574 | 0.9914 | 0.9545 | 0.9932 |
| ImageNet → DR5 desktop + JYE01 (device × class balanced sampling) | 0.9227 | 0.8778 | 0.9335 | 0.9936 | 0.9545 | 0.9955 |
| ImageNet → APTOS 2019 → DR5 desktop | **0.9528** | **0.9111** | **0.9628** | 0.9936 | 0.9545 | 0.9955 |
| ImageNet → EyePACS + APTOS 2019 (5 epochs) → DR5 desktop | 0.9442 | 0.8556 | **0.9654** | 0.9914 | 0.9545 | 0.9932 |
| ImageNet → EyePACS + APTOS 2019 (10 epochs) → DR5 desktop | 0.9442 | 0.8556 | **0.9654** | 0.9871 | 0.9545 | 0.9887 |
