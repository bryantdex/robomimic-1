#!/bin/bash
# Frequency-weighted GRPO (self-consistency pseudo-labels) -- lambda sweep, seed 1.
# Same headline regime as baseline(45.3%)/self-verification(82.0%): sigma=1.0 strong
# pool, 100 epochs, 50 rollouts, horizon 300, rate 50.
set -u
source /tmp/rmenv.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"; conda activate robomimic
cd /root/robomimic
OUT=/root/rm_runs
for LAM in 05 07 085; do
  DS=datasets/lift/mh/lift_scfw_l${LAM}.hdf5
  NAME=scfw_l${LAM}_seed1
  rm -rf "${OUT:?}/${NAME}"
  echo "######## TRAIN ${NAME} ########"
  python verification_grpo/train_bc.py --dataset "$DS" --name "$NAME" --output_dir "$OUT" \
    --filter_key scfw --epochs 100 --steps_per_epoch 100 \
    --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed 1 --wandb 0 \
    > "${OUT}/${NAME}.log" 2>&1
  echo "######## DONE ${NAME} rc=$? ########"
done
echo "SCFW_SWEEP_DONE"
