#!/usr/bin/env bash
# SVM on the frozen backbone of the two-stage models (APTOS -> desktop), seeds 0-2:
#   tools/svm_dr.py (flip4 features, params by patient-grouped CV inside TRAIN) for 2 SVM training sets
#   (desktop only = Canon Nidek, and + JYE01), then eval_mtm_vt_postprocess.py (ALFA) on the SVM predictions
#   with 3 lesion-threshold settings (age 40 deployed, age 60 deployed, age 60 tuned).
# Run inside tmux (keeps running when SSH closes):
#   tmux new -s svm ; conda activate dr ; cd ~/Workspace/prj/DR-Grading ; bash scripts/run_svm.sh
# Two GPUs (one script per GPU, different seeds):
#   GPU=2 SEEDS="0 1" bash scripts/run_svm.sh & GPU=3 SEEDS=2 bash scripts/run_svm.sh & wait
# Finished steps are skipped when the script is run again (their output file exists).
set -u
cd "$(dirname "$0")/.." || exit 1
GPU=${GPU:-2}
SEEDS=${SEEDS:-0 1 2}   # e.g. two GPUs: GPU=2 SEEDS="0 1" bash ... & GPU=3 SEEDS=2 bash ...
DATA=${DATA:-/mnt/10TB_2024/Katsu/data/DR/DR5_512}
OUT=${OUT:-/mnt/10TB_2024/Katsu/Result/DR}
DET=${DET:-/mnt/10TB_2024/Katsu/Result/deployment/DR-Grading/Det_Raw/det_raw_0JCN}
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/svm_queue.log"; }

for s in $SEEDS; do   # s0 features are already cached (svm_features/), so its results come first
  R=$OUT/DR5_swinMS_desktop_384_aptosft_s$s
  for devs in "Canon Nidek" "Canon Nidek JYE01"; do
    tag=${devs// /}
    D=$R/svm_flip4_cv_$tag
    if [ -f "$D/predictions_val.csv" ]; then
      log "SKIP svm s$s $tag (done)"
    else
      log "START svm s$s $tag (GPU $GPU)"
      # shellcheck disable=SC2086
      python tools/svm_dr.py --resume "$R/checkpoint-best.pth" --data_path "$DATA" --tta flip4 \
        --svm_train_devices $devs > "$R/svm_flip4_cv_$tag.out" 2>&1 \
        || { log "FAIL svm s$s $tag, see $R/svm_flip4_cv_$tag.out"; continue; }
    fi
    for cfg in "age40_deployed|--age 40" "age60_deployed|--age 60" \
               "age60_tuned|--age 60 --threshold_config postprocess/threshold_config_tuned.json"; do
      name=${cfg%%|*}; opts=${cfg#*|}
      [ -f "$D/postprocess_${name}.out" ] && continue
      # shellcheck disable=SC2086
      python eval_mtm_vt_postprocess.py --pred_csv "$D/predictions_val.csv" --det_root "$DET" $opts \
        --out_dir "$D/postprocess_det_raw_0JCN_$name" > "$D/postprocess_${name}.out" 2>&1 \
        || log "FAIL alfa s$s $tag $name"
    done
    log "DONE svm s$s $tag"
  done
done
log "svm queue finished"
