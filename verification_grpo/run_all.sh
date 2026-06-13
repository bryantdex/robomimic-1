#!/bin/bash
# End-to-end pipeline: baseline -> verify -> filtered tau sweep on the noisy pool.
# No pattern-based pkill anywhere (avoids self-termination).
set -u
source /tmp/rmenv.sh
cd /root/robomimic

DS=datasets/lift/mh/lift_noisy_pool80.hdf5
OUT=/root/rm_runs
SEED=1
EPOCHS=100
NROLL=30
HORIZON=300
PREFIX=v80

run_bc () {   # name  filter_key
  local name="$1" key="$2"
  rm -rf "${OUT:?}/${name}"
  echo "######## TRAIN ${name} (filter_key=${key}) ########"
  python verification_grpo/train_bc.py --dataset "$DS" --name "$name" --output_dir "$OUT" \
    --filter_key "$key" --epochs $EPOCHS --steps_per_epoch 100 \
    --n_rollouts $NROLL --horizon $HORIZON --rollout_rate 50 --seed $SEED --wandb 0 \
    > "${OUT}/${name}.log" 2>&1
  echo "######## DONE ${name} rc=$? ########"
}

# 1) baseline on full noisy pool (= the verifier model)
run_bc "${PREFIX}_baseline_seed${SEED}" "pool_all"

# 2) verification with the baseline policy (self-verification)
CKPT=$(ls ${OUT}/${PREFIX}_baseline_seed${SEED}/*/models/model_epoch_*success*.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls ${OUT}/${PREFIX}_baseline_seed${SEED}/*/models/model_epoch_${EPOCHS}.pth | head -1)
echo "######## VERIFY with $CKPT ########"
python verification_grpo/verify.py --ckpt "$CKPT" --dataset "$DS" --out ${OUT}/verify_${PREFIX} \
  --pool_filter_key pool_all --key_prefix $PREFIX --K 32 --eps 0.30 \
  --abs_taus 0.1 0.3 0.5 0.7 0.85 > ${OUT}/verify_${PREFIX}.log 2>&1
grep -iE "mean v_bar|filter key|ordering" ${OUT}/verify_${PREFIX}.log

# 3) filtered BC sweep over tau
for TAU in 010 030 050 070 085; do
  run_bc "${PREFIX}_filtered_tau${TAU}_seed${SEED}" "${PREFIX}_tau${TAU}"
done

echo "ALL_DONE_PIPELINE"
