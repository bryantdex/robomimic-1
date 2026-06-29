#!/bin/bash
# Downstream BC + 30-rollout eval on the correlated-bias pool (rho=0.6).
# Compares: no-filter baseline, best single-round diff-inst configs (M,tau),
# iterative fresh-twin R-sweep, and the clean-only oracle. Same recipe throughout
# (100 epochs, horizon 300, 30 rollouts, rate 50).
set -u
source /tmp/rmenv.sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate robomimic
cd /root/robomimic
DS=datasets/lift/mh/lift_corr_r060.hdf5
OUT=/root/rm_runs_itertwin
SEED=${SEED:-1}
TAG=${TAG:-}

run_bc () {  # name key
  local name="$1${TAG}" key="$2"
  rm -rf "${OUT:?}/${name}"
  echo "######## TRAIN ${name} (key=${key} seed=${SEED}) ########"
  python verification_grpo/train_bc.py --dataset "$DS" --name "$name" --output_dir "$OUT" \
    --filter_key "$key" --epochs 100 --steps_per_epoch 100 \
    --n_rollouts 30 --horizon 300 --rollout_rate 50 --seed "$SEED" --wandb 0 \
    > "${OUT}/${name}.log" 2>&1
  echo "######## DONE ${name} rc=$? ########"
}

run_bc "cd_baseline"   "pool_all"        # no filter (40 clean + 40 corrupt)
run_bc "cd_srM8_t30"   "ens_M8_tau030"   # single-round diff-inst, high-purity config
run_bc "cd_srM6_t30"   "ens_M6_tau030"   # single-round diff-inst, balanced
run_bc "cd_srM6_t15"   "ens_M6_tau015"   # single-round diff-inst, high-recall config
run_bc "cd_itR1"       "iter_R1_tau015"  # iterative, 1 round
run_bc "cd_itR2"       "iter_R2_tau015"  # iterative, 2 rounds
run_bc "cd_itR3"       "iter_R3_tau015"  # iterative, 3 rounds (converged)
run_bc "cd_oracle"     "clean"           # ceiling
echo "ALL_CORR_DOWNSTREAM_DONE"
