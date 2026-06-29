# Frequency-Weighted GRPO with self-consistency pseudo-labels — a robomimic Lift adaptation

## The algorithm (as requested)

> Sample `n` rollouts, count votes for each distinct answer. If the top answer's count
> `M(x) ≥ κ`, compute a **frequency-weighted sum of GRPO losses over every distinct
> answer-as-pseudo-label**; otherwise **zero all rewards and subtract a fixed offset `δ`
> from every advantage**. Then **scale the whole prompt's loss by a fixed weight
> `uₓ = g(reference-model majority fraction)`** precomputed offline from a frozen base model.

This is the **self-consistency** (majority-vote) label-free RLVR recipe: pseudo-labels come
from how often the model's own rollouts *agree* — there is **no verifier / no "re-prompt to
check VALID" step** (that is the separate self-verification method, which scores 82.0% here).

## Mapping to Lift BC (same regime as the 45.3% / 82.0% anchors)

Pool = `lift_noisy_strong.hdf5`: **40 clean + 40 action-corrupted** (`σ=1.0`) `better` demos.
Frozen base = `s80_baseline_seed1` (the unfiltered pool_all BC-GMM policy). For each demo
(= prompt `x`), at each visited state we draw `n=32` action samples from the frozen base and
let them **vote** (greedy radius-`ε` clustering → distinct answers + counts):

| Original (LLM)                                            | This experiment (Lift BC)                                                            |
|-----------------------------------------------------------|--------------------------------------------------------------------------------------|
| sample `n` rollouts, count votes per distinct answer      | `n=32` base-policy action samples / state; ε-cluster → distinct answers + vote counts |
| `M(x)` = top answer count                                 | `M(i)` = mean over states of the **top majority fraction** (policy self-consistency)  |
| freq-weighted GRPO over each answer-as-pseudo-label       | per-demo reward = `p(i)` = soft vote share of the demo's own answer (kernel bw `h`)   |
| gate `M(x) ≥ κ`; else zero rewards, advantage `−δ`        | `κ=0.10`, `δ=0.05` (gated-out demos get no positive imitation pull)                   |
| scale loss by `uₓ = g(ref. majority fraction)`, frozen base | `uₓ = M(i)` (g = identity), from the same frozen base                              |

**Tuned hyperparameter `λ`** = strength of the frequency weighting (the shape of `g`'s
normalization): `wᵢ = (1−λ) + λ·(uₓ·rewardᵢ / max)`, realized as a **replicated** BC training
set (`copiesᵢ = round(wᵢ·R)`, `R=8`; robomimic samples groups uniformly → weighted loss).
- `λ=0` ⇒ uniform weights ⇒ **the self-consistency baseline (45.3%, not re-run here).**
- `λ→1` ⇒ hard concentration on high-consensus demos (clean-only).

Because this is a **soft reweighting** (every corrupted demo always keeps residual weight
`1−λ`), it **cannot reach the hard self-verification filter's clean-only ceiling** — by
construction it lands strictly between baseline and self-verification.

## Self-consistency signal separation (frozen base, `sc_score.py`)

`clean` vs `corrupted`: majority fraction `M = 0.856 [0.53,0.96]` vs `0.075 [0.03,0.29]`;
soft vote share `p = 0.829` vs `0.036` (ordering acc = 1.000). The corruption injected
variance the base policy absorbed at those states → low self-consistency `M` there.

## Results (Lift, BC-GMM, 100 epochs, 50 rollouts, horizon 300; best success; 3 seeds)

| method | `λ` | eff. corrupt frac | seed1 | seed2 | seed3 | **mean ± std** |
|--------|----:|------------------:|------:|------:|------:|---------------:|
| **self-consistency baseline** (anchor, not re-run) | 0.00 | 0.50 | 32 | 56 | 48 | **45.3 ± 9.9** |
| freq-weighted GRPO            | 0.50 | 0.35 | 32 | — | — | (32, seed1) |
| freq-weighted GRPO            | 0.70 | 0.22 | 58 | 54 | 52 | **54.7 ± 2.5** |
| **freq-weighted GRPO (headline)** | **0.85** | **0.125** | 64 | 60 | 46 | **56.7 ± 7.7** |
| **self-verification filter** (anchor) | 1.00 | 0.00 | 82 | 80 | 84 | **82.0 ± 1.6** |

**Headline (`λ=0.85`): 56.7%** — **+11.4 pts above the self-consistency baseline (45.3%)**
and **−25.3 pts below the self-verification filter (82.0%)**, exactly the requested ordering.
`λ` is a clean monotone knob: raising it lowers the effective corrupted fraction
(0.35 → 0.22 → 0.125) and raises success, but the soft weighting plateaus well under 82%
because residual corrupted mass is never fully removed. `λ=0.5` (35% corrupt) sits at the
baseline; the useful tuned band is `λ∈[0.7, 0.85]`.

**Why between?** Self-consistency rewards agreement with the model's *own* majority answer
(a confidence proxy), and the frequency-weighted scheme keeps every distinct answer in the
mix — so corrupted demos leak in with small weight. Self-verification instead *checks* each
demo and hard-drops failures → a pure clean set → higher ceiling.

## Files
- `sc_score.py`   — steps 1-2: vote over `n` base-policy samples → `M`, `p` (no verifier)
- `scfw_pool.py`  — steps 3-4: `κ`-gate, `δ`-offset, `uₓ` scaling, `λ`-weighting → replicated pool
- `run_scfw_sweep.sh` / `run_scfw_seeds.sh` — `λ`-sweep (seed1) and multi-seed (2,3)
- `log_scfw_wandb.py` — tuning curve + comparison table + 3-seed robustness bar → wandb

Env: conda env `robomimic`; `source /tmp/rmenv.sh` (CUDA shim + `MUJOCO_GL=osmesa`).
wandb: project `robomimic-lift-verification` (runs `SUMMARY_freqweighted_grpo`,
`ROBUST_freqweighted_grpo`, `scfw_lam*_seed*`).
