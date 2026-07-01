#!/bin/bash
# Multi-seed (2,3) for the top in-band self-certainty GRPO operating point(s), to get 3-seed
# means comparable to baseline(45.3%) and self-verification(82.0%). Pools already built by the sweep.
# Usage: run_sccert_seeds.sh [BETA_TAGS...]   (default: 05 075)
set -u
source /tmp/rmenv.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"; conda activate robomimic
cd /root/robomimic
OUT=/root/rm_runs
TAGS=${@:-"05 075"}
for BT in $TAGS; do
  DST=datasets/lift/mh/lift_sccert_b${BT}.hdf5
  for SEED in 2 3; do
    NAME=sccert_b${BT}_seed${SEED}
    rm -rf "${OUT:?}/${NAME}"
    echo "######## EVAL ${NAME} ########"
    python verification_grpo/train_bc.py --dataset "$DST" --name "$NAME" --output_dir "$OUT" \
      --filter_key sccert_soft --epochs 100 --steps_per_epoch 100 \
      --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed $SEED --wandb 0 \
      > "${OUT}/${NAME}.log" 2>&1
    echo "######## DONE ${NAME} rc=$? ########"
    grep -h '"Success_Rate"' "${OUT}/${NAME}.log" | tail -n 2
  done
done
echo "SCCERT_SEEDS_DONE"
