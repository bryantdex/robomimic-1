#!/bin/bash
# EMA mean-teacher GRPO -- lambda sweep, seed 1.
# Step A (once): run the EMA mean-teacher loop -> converged teacher + per-demo reward.
# Step B (per lambda): build the lambda-weighted pool, train+eval a fresh BC on it.
# Regime matches baseline(45.3%)/self-verification(82.0%): sigma=1.0 strong pool,
# 100 epochs, 50 rollouts, horizon 300, rate 50.
set -u
source /tmp/rmenv.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"; conda activate robomimic
cd /root/robomimic
OUT=/root/rm_runs
BASE=/root/rm_runs/s80_baseline_seed1/20260613123643/last.pth
SRC=datasets/lift/mh/lift_noisy_strong.hdf5
SUMM=${OUT}/ema_lab/ema_summary.json

echo "######## EMA MEAN-TEACHER LOOP ########"
python verification_grpo/ema_teacher_run.py --dataset "$SRC" --base_ckpt "$BASE" \
  --out "${OUT}/ema_lab" --gtilde 32 --R 3 --eps 0.30 --lam_ref 0.85 --Rep 10 \
  --alpha0 0.99 --alpha1 0.9999 --round_epochs 25 > "${OUT}/ema_lab.log" 2>&1
echo "######## DONE EMA LOOP rc=$? ########"

for L in 0.5 0.7 0.85 0.9; do
  LAM=${L/./}   # 0.5->05, 0.85->085
  DST=datasets/lift/mh/lift_ema_l${LAM}.hdf5
  echo "######## POOL lambda=${L} ########"
  python verification_grpo/ema_pool.py --summary "$SUMM" --src "$SRC" --dst "$DST" --lam $L --R 10 \
    > "${OUT}/ema_pool_l${LAM}.log" 2>&1
  cat "${OUT}/ema_pool_l${LAM}.log"
  NAME=ema_l${LAM}_seed1
  rm -rf "${OUT:?}/${NAME}"
  echo "######## EVAL ${NAME} ########"
  python verification_grpo/train_bc.py --dataset "$DST" --name "$NAME" --output_dir "$OUT" \
    --filter_key ema_soft --epochs 100 --steps_per_epoch 100 \
    --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed 1 --wandb 0 \
    > "${OUT}/${NAME}.log" 2>&1
  echo "######## DONE EVAL ${NAME} rc=$? ########"
done
echo "EMA_SWEEP_DONE"
