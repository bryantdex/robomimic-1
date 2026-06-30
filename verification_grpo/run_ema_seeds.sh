#!/bin/bash
# Multi-seed (2,3) for the top in-band EMA-teacher GRPO operating points, to get 3-seed means
# comparable to baseline(45.3%) and self-verification(82.0%). Pools already built by the sweep.
# Usage: run_ema_seeds.sh [LAM_TAGS...]   (default: 085 09)
set -u
source /tmp/rmenv.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"; conda activate robomimic
cd /root/robomimic
OUT=/root/rm_runs
TAGS=${@:-"085 09"}
for LAM in $TAGS; do
  DST=datasets/lift/mh/lift_ema_l${LAM}.hdf5
  for SEED in 2 3; do
    NAME=ema_l${LAM}_seed${SEED}
    rm -rf "${OUT:?}/${NAME}"
    echo "######## EVAL ${NAME} ########"
    python verification_grpo/train_bc.py --dataset "$DST" --name "$NAME" --output_dir "$OUT" \
      --filter_key ema_soft --epochs 100 --steps_per_epoch 100 \
      --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed $SEED --wandb 0 \
      > "${OUT}/${NAME}.log" 2>&1
    echo "######## DONE ${NAME} rc=$? ########"
  done
done
echo "EMA_SEEDS_DONE"
