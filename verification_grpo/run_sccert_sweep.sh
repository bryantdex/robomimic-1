#!/bin/bash
# Self-certainty GRPO -- beta sweep, seed 1.
# Step A (once): run the ONLINE self-certainty loop -> converged policy + final per-demo
#                self-certainty u_i (= mean_t log pi_online(a|s)).
# Step B (per beta): build the beta-weighted (exp(beta*A), A=group-norm self-certainty) pool,
#                train+eval a fresh BC on it.
# Regime matches baseline(45.3%)/self-verification(82.0%): sigma=1.0 strong pool,
# 100 epochs, 50 rollouts, horizon 300, rate 50.
set -u
source /tmp/rmenv.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"; conda activate robomimic
cd /root/robomimic
OUT=/root/rm_runs
BASE=/root/rm_runs/s80_baseline_seed1/20260613123643/last.pth
SRC=datasets/lift/mh/lift_noisy_strong.hdf5
SUMM=${OUT}/sccert_lab/sccert_summary.json

echo "######## ONLINE SELF-CERTAINTY LOOP ########"
python verification_grpo/sccert_run.py --dataset "$SRC" --base_ckpt "$BASE" \
  --out "${OUT}/sccert_lab" --R 3 --beta_ref 1.0 --Rep 10 --round_epochs 25 \
  > "${OUT}/sccert_lab.log" 2>&1
echo "######## DONE ONLINE LOOP rc=$? ########"
tail -n 8 "${OUT}/sccert_lab.log"

for B in 0.25 0.5 0.75 1.0 1.5; do
  BT=${B/./}    # 0.25->025, 1.0->10
  DST=datasets/lift/mh/lift_sccert_b${BT}.hdf5
  echo "######## POOL beta=${B} ########"
  python verification_grpo/sccert_pool.py --summary "$SUMM" --src "$SRC" --dst "$DST" --beta $B --R 10 \
    > "${OUT}/sccert_pool_b${BT}.log" 2>&1
  cat "${OUT}/sccert_pool_b${BT}.log"
  NAME=sccert_b${BT}_seed1
  rm -rf "${OUT:?}/${NAME}"
  echo "######## EVAL ${NAME} ########"
  python verification_grpo/train_bc.py --dataset "$DST" --name "$NAME" --output_dir "$OUT" \
    --filter_key sccert_soft --epochs 100 --steps_per_epoch 100 \
    --n_rollouts 50 --horizon 300 --rollout_rate 50 --seed 1 --wandb 0 \
    > "${OUT}/${NAME}.log" 2>&1
  echo "######## DONE EVAL ${NAME} rc=$? ########"
  grep -h '"Success_Rate"' "${OUT}/${NAME}.log" | tail -n 2
done
echo "SCCERT_SWEEP_DONE"
