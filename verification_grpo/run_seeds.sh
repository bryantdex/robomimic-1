#!/bin/bash
# Multi-seed robustness for the headline comparison: baseline (pool_all) vs
# verification-filtered (tau=0.3 -> verified-clean 40 demos). 50 rollouts each.
set -u
source /tmp/rmenv.sh
cd /root/robomimic
DS=datasets/lift/mh/lift_noisy_pool80.hdf5
OUT=/root/rm_runs
for SEED in 2 3; do
  for COND in "baseline:pool_all" "filtered_tau030:v80_tau030"; do
    NAME="v80_${COND%%:*}_seed${SEED}"; KEY="${COND##*:}"
    rm -rf "${OUT:?}/${NAME}"
    echo "######## TRAIN ${NAME} (key=${KEY}) ########"
    python verification_grpo/train_bc.py --dataset "$DS" --name "$NAME" --output_dir "$OUT" \
      --filter_key "$KEY" --epochs 100 --steps_per_epoch 100 \
      --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed $SEED --wandb 0 \
      > "${OUT}/${NAME}.log" 2>&1
    echo "######## DONE ${NAME} rc=$? ########"
  done
done
echo "ALL_SEEDS_DONE"
