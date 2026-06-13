#!/bin/bash
# Stronger-corruption headline experiment for a robust large gap.
# Pool: 40 clean + 40 corrupted with sigma=1.0 (corrupted actions ~ maximally invalid).
# baseline(seed1) doubles as the verifier model; then 3 seeds of baseline vs filtered(tau=0.3).
set -u
source /tmp/rmenv.sh
cd /root/robomimic
SRC=datasets/lift/mh/low_dim_v15.hdf5
DS=datasets/lift/mh/lift_noisy_strong.hdf5
OUT=/root/rm_runs
PREFIX=s80

python verification_grpo/make_noisy_dataset.py --src "$SRC" --dst "$DS" \
  --n_demos 80 --corrupt_frac 0.5 --sigma 1.0 --seed 0 2>&1 | grep -iE "wrote|demos:"

run_bc () {  # name key seed nroll
  local name="$1" key="$2" seed="$3" nroll="$4"
  rm -rf "${OUT:?}/${name}"
  echo "######## TRAIN ${name} (key=${key} seed=${seed}) ########"
  python verification_grpo/train_bc.py --dataset "$DS" --name "$name" --output_dir "$OUT" \
    --filter_key "$key" --epochs 100 --steps_per_epoch 100 \
    --n_rollouts "$nroll" --horizon 300 --rollout_rate 50 --seed "$seed" --wandb 0 \
    > "${OUT}/${name}.log" 2>&1
  echo "######## DONE ${name} rc=$? ########"
}

# 1) verifier = baseline seed 1
run_bc "${PREFIX}_baseline_seed1" "pool_all" 1 50

# 2) verify -> filter keys (tau=0.3)
CKPT=$(ls ${OUT}/${PREFIX}_baseline_seed1/*/models/model_epoch_*success*.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls ${OUT}/${PREFIX}_baseline_seed1/*/models/model_epoch_100.pth | head -1)
echo "######## VERIFY $CKPT ########"
python verification_grpo/verify.py --ckpt "$CKPT" --dataset "$DS" --out ${OUT}/verify_${PREFIX} \
  --pool_filter_key pool_all --key_prefix $PREFIX --K 32 --eps 0.30 --abs_taus 0.3 \
  > ${OUT}/verify_${PREFIX}.log 2>&1
grep -iE "mean v_bar|filter key" ${OUT}/verify_${PREFIX}.log

# 3) filtered seed1 + both conditions seeds 2,3
run_bc "${PREFIX}_filtered_seed1" "${PREFIX}_tau030" 1 50
for SEED in 2 3; do
  run_bc "${PREFIX}_baseline_seed${SEED}" "pool_all"        "$SEED" 50
  run_bc "${PREFIX}_filtered_seed${SEED}" "${PREFIX}_tau030" "$SEED" 50
done
echo "ALL_STRONG_DONE"
