#!/bin/bash
# Downstream BC training + 30-rollout eval for the iterative fresh-twin experiment.
# One run per filter key; same recipe as the diff-instantiation downstream runs
# (100 epochs, horizon 300, 30 rollouts, rate 50) so results are directly comparable.
set -u
source /tmp/rmenv.sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate robomimic
cd /root/robomimic
DS=datasets/lift/mh/lift_adv_b045.hdf5
OUT=/root/rm_runs_itertwin
SEED=${SEED:-1}

run_bc () {  # name key
  local name="$1" key="$2"
  rm -rf "${OUT:?}/${name}"
  echo "######## TRAIN ${name} (key=${key} seed=${SEED}) ########"
  python verification_grpo/train_bc.py --dataset "$DS" --name "$name" --output_dir "$OUT" \
    --filter_key "$key" --epochs 100 --steps_per_epoch 100 \
    --n_rollouts 30 --horizon 300 --rollout_rate 50 --seed "$SEED" --wandb 0 \
    > "${OUT}/${name}.log" 2>&1
  echo "######## DONE ${name} rc=$? ########"
}

for r in 1 2 3 4; do
  K="iter_R${r}_tau030"
  python -c "import h5py;f=h5py.File('$DS','r');import sys;sys.exit(0 if 'mask/$K' in f else 1)" \
    && run_bc "it_R${r}" "$K" || echo "skip it_R${r}: no key $K"
done

# oracle ceiling: train on the 40 truly-clean demos
run_bc "it_oracle" "clean"

echo "ALL_DOWNSTREAM_DONE"
