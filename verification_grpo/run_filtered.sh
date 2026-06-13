#!/bin/bash
# Train verification-filtered BC for each tau filter key produced by verify.py.
# Usage: run_filtered.sh <dataset> <output_dir> <seed> "<key1 key2 ...>"
set -u
DATASET="$1"; OUTDIR="$2"; SEED="$3"; KEYS="$4"
cd /root/robomimic
for KEY in $KEYS; do
  NAME="filtered_${KEY}_seed${SEED}"
  rm -rf "${OUTDIR:?}/${NAME}"
  echo "=== launching $NAME (filter_key=$KEY) ==="
  python verification_grpo/train_bc.py \
    --dataset "$DATASET" \
    --name "$NAME" --output_dir "$OUTDIR" \
    --filter_key "$KEY" --epochs 100 --steps_per_epoch 100 \
    --n_rollouts 30 --horizon 300 --rollout_rate 50 --seed "$SEED" --wandb 0 \
    > "${OUTDIR}/${NAME}.log" 2>&1
  echo "=== finished $NAME rc=$? ==="
done
echo "ALL_FILTERED_DONE"
