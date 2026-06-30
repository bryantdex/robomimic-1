#!/bin/bash
# Semantic-entropy-filtered GRPO -- delta_high sweep, seed 1.
# Same headline regime as baseline(45.3%)/self-verification(82.0%): sigma=1.0 strong
# pool, 100 epochs, 50 rollouts, horizon 300, rate 50. Single tuned knob = delta_high.
set -u
source /tmp/rmenv.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"; conda activate robomimic
cd /root/robomimic
OUT=/root/rm_runs
for DH in 350 340 330 320 300 270; do
  DS=datasets/lift/mh/lift_sefw_dh${DH}.hdf5
  NAME=sefw_dh${DH}_seed1
  rm -rf "${OUT:?}/${NAME}"
  echo "######## TRAIN ${NAME} ########"
  python verification_grpo/train_bc.py --dataset "$DS" --name "$NAME" --output_dir "$OUT" \
    --filter_key sefw --epochs 100 --steps_per_epoch 100 \
    --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed 1 --wandb 0 \
    > "${OUT}/${NAME}.log" 2>&1
  echo "######## DONE ${NAME} rc=$? ########"
done
echo "SEFW_SWEEP_DONE"
