#!/usr/bin/env bash
# Reproduce the best model from scratch, bit for bit (--deterministic):
#   stage 1: ImageNet -> APTOS 2019            (APTOS_swinMS_384_det_s0)
#   stage 2: APTOS checkpoint -> DR5 desktop   (DR5_swinMS_desktop_384_aptosft_det_s0, Canon + Nidek, old labels)
#   eval on the 466 prospective test images: argmax, TTA flip4, ALFA (flip4; age 40 / 60 deployed, age 60 tuned)
#   then compare with the reference run (log.txt, weights, predictions): should be IDENTICAL.
#
# Bit-identical only with the same: arguments (incl. --num_workers 4), data files, code, GPU model (RTX A5000),
# driver / CUDA / cuDNN, torch 2.5.1, timm 0.9.16. On another machine the results are close, not identical.
#
# Usage (conda env dr, inside tmux; ~3 h):
#   bash scripts/reproduce_best.sh
#   GPU=3 OUT=/some/dir bash scripts/reproduce_best.sh
set -eu
cd "$(dirname "$0")/.." || exit 1   # repo root

GPU=${GPU:-2}
OUT=${OUT:-/mnt/10TB_2024/Katsu/Result/DR/reproduce}
APTOS=${APTOS:-/mnt/10TB_2024/Katsu/data/DR/APTOS2019_512}
DATA=${DATA:-/mnt/10TB_2024/Katsu/data/DR/DR5_512}
DET=${DET:-/mnt/10TB_2024/Katsu/Result/deployment/DR-Grading/Det_Raw/det_raw_0JCN}
REF=${REF:-/mnt/10TB_2024/Katsu/Result/DR}   # reference run: $REF/<task>/ (empty = no comparison)
S1=APTOS_swinMS_384_det_s0
S2=DR5_swinMS_desktop_384_aptosft_det_s0
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1

if [ "$(realpath -m "$OUT")" = "$(realpath -m "$REF")" ]; then
  echo "OUT = REF ($OUT): would overwrite the reference run"; exit 1
fi
for t in $S1 $S2; do
  if [ -e "$OUT/$t" ]; then echo "$OUT/$t exists: delete it or use another OUT"; exit 1; fi
done
mkdir -p "$OUT"
LOG=$OUT/reproduce.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- environment (warn when it differs from the reference machine)
python - <<'EOF' 2>&1 | tee -a "$LOG"
import subprocess, torch, timm
want = {"torch": "2.5.1", "timm": "0.9.16", "gpu": "NVIDIA RTX A5000", "driver": "550.100"}
gpu = torch.cuda.get_device_name(0)  # the GPU of CUDA_VISIBLE_DEVICES (nvidia-smi ignores it)
driver = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split()[0]
have = {"torch": torch.__version__.split("+")[0], "timm": timm.__version__, "gpu": gpu, "driver": driver}
print(f"env: {have}, cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}")
for k in want:
    if have[k] != want[k]:
        print(f"WARNING: {k} {have[k]} != {want[k]} (reference): results will not be bit-identical")
EOF
log "git $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo ' + uncommitted changes'), GPU $GPU, OUT $OUT"

COMMON=(
  --model SwinMS --model_arch swinv2_large_window12to24_192to384 --ms_stages 0 1 2 3
  --nb_classes 5 --clahe --grad_ckpt --batch_size 4 --accum_iter 4
  --epochs 10 --warmup_epochs 1 --blr 1e-3 --layer_decay 0.9 --drop_path 0.2
  --num_workers 4 --adaptation finetune --no_tqdm
  --output_dir "$OUT" --log_dir "$OUT/output_logs"
  --input_size 384 --seed 0 --deterministic
)

# ---- stage 1: APTOS
log "stage 1 $S1"
mkdir -p "$OUT/$S1"
python train_dr.py "${COMMON[@]}" --data_path "$APTOS" --train_devices APTOS --val_devices APTOS \
  --task "$S1" > "$OUT/$S1/train.out" 2>&1

# ---- stage 2: DR5 desktop, from the stage-1 checkpoint
log "stage 2 $S2"
mkdir -p "$OUT/$S2"
python train_dr.py "${COMMON[@]}" --data_path "$DATA" --train_devices Canon Nidek --val_devices Canon Nidek \
  --finetune "$OUT/$S1/checkpoint-best.pth" --task "$S2" > "$OUT/$S2/train.out" 2>&1

# ---- eval on the 466 prospective test images (DR5_512 val)
D=$OUT/$S2
log "eval $S2"
python eval_mtm_vt.py --resume "$D/checkpoint-best.pth" --data_path "$DATA" --batch_size 16 --report_split val \
  > "$D/eval.out" 2>&1
python eval_mtm_vt.py --resume "$D/checkpoint-best.pth" --data_path "$DATA" --batch_size 16 --report_split val \
  --tta flip4 > "$D/eval_tta_flip4.out" 2>&1
for cfg in "age40_deployed|--age 40" "age60_deployed|--age 60" \
           "age60_tuned|--age 60 --threshold_config postprocess/threshold_config_tuned.json"; do
  name=${cfg%%|*}; opts=${cfg#*|}
  # shellcheck disable=SC2086
  python eval_mtm_vt_postprocess.py --pred_csv "$D/tta_flip4/predictions_val.csv" --det_root "$DET" $opts \
    --out_dir "$D/tta_flip4/postprocess_$(basename "$DET")_$name" > "$D/tta_flip4/postprocess_$name.out" 2>&1
done
log "results: $D/eval_tta_flip4.out, $D/tta_flip4/postprocess_*.out"

# ---- compare with the reference run
if [ -n "$REF" ] && [ -f "$REF/$S2/checkpoint-best.pth" ]; then
  log "compare with $REF"
  python - "$OUT" "$REF" "$S1" "$S2" <<'EOF' 2>&1 | tee -a "$LOG"
import json, sys
import numpy as np, pandas as pd, torch
out, ref, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
ok = True
for t in tasks:
    a, b = f"{out}/{t}", f"{ref}/{t}"
    la = [json.loads(l) for l in open(f"{a}/log.txt")]
    lb = [json.loads(l) for l in open(f"{b}/log.txt")]
    ma = torch.load(f"{a}/checkpoint-best.pth", map_location="cpu", weights_only=False)["model"]
    mb = torch.load(f"{b}/checkpoint-best.pth", map_location="cpu", weights_only=False)["model"]
    same_w = ma.keys() == mb.keys() and all(torch.equal(ma[k], mb[k]) for k in ma)
    res = {"log.txt": la == lb, "weights": same_w}
    for f in ("predictions_val.csv", "tta_flip4/predictions_val.csv"):
        try:
            pa, pb = pd.read_csv(f"{a}/{f}"), pd.read_csv(f"{b}/{f}")
            res[f] = pa.equals(pb)
        except FileNotFoundError:
            pass
    ok &= all(res.values())
    print(f"{t}: " + ", ".join(f"{k} {'same' if v else 'DIFFERENT'}" for k, v in res.items()))
print("REPRODUCED: IDENTICAL to the reference run" if ok else "NOT identical to the reference run")
EOF
fi
log "done"
