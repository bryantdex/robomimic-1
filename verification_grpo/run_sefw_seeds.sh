#!/bin/bash
# Multi-seed (2,3) for the two intermediate semantic-entropy operating points
# delta_high=3.4, 3.3, to get 3-seed means comparable to baseline(45.3%) / self-verification(82.0%).
set -u
source /tmp/rmenv.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"; conda activate robomimic
cd /root/robomimic
OUT=/root/rm_runs
for DH in 340 330; do
  DS=datasets/lift/mh/lift_sefw_dh${DH}.hdf5
  for SEED in 2 3; do
    NAME=sefw_dh${DH}_seed${SEED}
    rm -rf "${OUT:?}/${NAME}"
    echo "######## TRAIN ${NAME} ########"
    python verification_grpo/train_bc.py --dataset "$DS" --name "$NAME" --output_dir "$OUT" \
      --filter_key sefw --epochs 100 --steps_per_epoch 100 \
      --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed $SEED --wandb 0 \
      > "${OUT}/${NAME}.log" 2>&1
    echo "######## DONE ${NAME} rc=$? ########"
  done
done
echo "SEFW_SEEDS_DONE"
