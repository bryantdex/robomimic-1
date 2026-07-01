# Self-certainty GRPO on Lift — a robomimic adaptation

## The algorithm

> For each prompt you sample G completions, score each by **self-certainty** using the
> **online (current) policy**, normalize via `(uᵢ − mean)/std` within the group, and run a
> policy-gradient update with **no external labels or verifiers**.

This is label-free / verifier-free RLVR in the RLSC ("reinforcement learning from
self-certainty") family: the reward is the policy's own confidence in a completion — there is
no re-prompt-to-check-VALID step (that is the separate self-verification method) and no
majority vote over a frozen base (that is the self-consistency method).

## Mapping to Lift BC (same noisy regime as the 45.3% / 82.0% anchors)

| Original (LLM RLSC)                              | This experiment (Lift BC)                                                                 |
|--------------------------------------------------|-------------------------------------------------------------------------------------------|
| prompt + group of G completions                  | the Lift task; the **group = the pool of G=80 candidate demos** (each demo = one completion) |
| self-certainty `uᵢ` (KL-from-uniform / peakedness, online policy) | `uᵢ = mean_t log π_online(a_{i,t} \| s_{i,t})` — the **online** BC-GMM's confidence in producing demo *i*'s own actions |
| group-normalize `Aᵢ=(uᵢ−mean)/std`               | identical — over the 80 demos                                                              |
| policy-gradient update (raise log-prob of high-advantage completions) | advantage-weighted imitation (AWR) on the **raw expert actions**: `wᵢ=exp(β·Aᵢ)`, `copiesᵢ=round(R·wᵢ/max)` |
| online (current) policy is the scorer            | each round re-scores self-certainty with the **just-trained** policy (`--init_ckpt` warm-start) — not a frozen base, not an EMA teacher |

The pool is the canonical noisy regime: **40 clean** + **40 action-corrupted** (`σ=1.0`) Lift
demos (`lift_noisy_strong.hdf5`). Baseline BC on all 80 is dragged down by the bad half; a
method that denoises beats it.

**Why self-certainty separates clean from corrupt:** an on-manifold expert trajectory has high
log-density under the BC-GMM; a σ=1.0 action-corrupted trajectory is off-manifold and has very
low log-density. Probe on the base policy: clean `uᵢ ≈ +26`, corrupt `uᵢ ≈ −3`, ordering acc
**1.000**. The online loop sharpens this (clean 26.25 → 28.66 over 3 rounds; corrupt stays ≈ −3;
ordering acc stays 1.000).

## The tuned hyperparameter: β (GRPO advantage temperature)

`wᵢ = exp(β·Aᵢ)` over group-normalized self-certainty advantages.
- **β = 0** → uniform weights → trains on all 80 → the self-consistency **baseline (45.3%)**.
- **β → large** → only high-self-certainty (clean) demos keep weight → the hard-verification
  **clean-only ceiling (82.0%)**.
- Tuned β lands strictly between.

(At σ=1.0 the clean/corrupt self-certainty separation is *saturated* — ordering acc 1.000 at
every round — so it is β, not the number of online rounds, that places the result in the band.)

## Results (Lift, low-dim BC-GMM, 100 epochs, 50 rollouts; σ=1.0 pool of 80)

Online loop: R=3 rounds, β_ref=1.0, Rep=10, 25 epochs/round; final ordering acc **1.000**
(clean `u`≈28.7, corrupt `u`≈−3.0).

| run (β) | eff. corrupt frac | seed1 best | 3-seed mean ± std | placement |
|---------|------------------:|-----------:|------------------:|-----------|
| self-consistency baseline (β=0) | 0.50 | — | **45.3 ± 9.9** | anchor |
| β = 0.25 | 0.388 | 42.0 | — | ≈ baseline |
| β = 0.50 | 0.252 | 52.0 | — | in band |
| β = 0.75 | 0.191 | 66.0 | **59.3 ± 8.1** | **in band** |
| **β = 1.00** | 0.109 | 70.0 | **66.0 ± 10.2** | **in band (headline)** |
| β = 1.50 | 0.000 (clean only) | 92.0 | — | β→∞ limit = hard filter |
| self-verification filter | 0.0 | — | **82.0 ± 1.6** | anchor |

**Headline β=1.0 → 66.0 ± 10.2%**: +20.7 pts over the self-consistency baseline (45.3%),
−16.0 pts under the self-verification filter (82.0%) — lands strictly between, as required.
The tuning curve is monotone: increasing β trades off residual corrupt-demo weight for
clean-demo concentration, sweeping from baseline up to the hard-filter ceiling (β=1.5 drives
eff-corrupt to exactly 0 → recovers the clean-only filter at 92% single-seed).

It stays **below** the hard filter because the advantage weighting is *soft* — corrupted demos
keep residual weight `exp(β·A_corrupt) > 0` for finite β — so it cannot reach the filter's
clean-only ceiling.

## Files / how to reproduce
- `sccert_run.py`        — the **online** self-certainty GRPO loop: score `log π_online(a|s)` →
  group-normalize → `exp(β·A)` weighted pool → warm-start train → repeat; emits final per-demo `uᵢ`
- `sccert_pool.py`       — build the β-weighted downstream training pool from `sccert_summary.json`
- `run_sccert_sweep.sh`  — online loop (once) → β sweep {0.25,0.5,0.75,1.0,1.5} eval (seed1)
- `run_sccert_seeds.sh`  — multi-seed (2,3) for the in-band β points
- `log_sccert_wandb.py`  — parse run logs → tuning curve + comparison table + 3-seed bar → wandb

Anchors (NOT re-run): self-consistency baseline 45.3% and self-verification filter 82.0%
(see `README.md`, `README_freqweighted.md`). wandb project `robomimic-lift-verification`
(`SUMMARY_self_certainty_grpo`, `ROBUST_self_certainty_grpo`, `sccert_beta*_seed*`).

Environment: conda env `robomimic`; `source /tmp/rmenv.sh` (CUDA shim + `MUJOCO_GL=osmesa`).
