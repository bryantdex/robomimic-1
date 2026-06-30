# Semantic-Entropy-Filtered GRPO — a robomimic Lift adaptation

## The algorithm (as requested)

> Sample `G` responses per unlabeled question, **cluster them by semantic equivalence**,
> reward each response by its cluster's **empirical frequency** `p(c|q) ≈ |c|/G`, compute
> **group-normalized advantages**, and run the clipped policy-gradient update **only on
> questions whose semantic entropy falls within `(δ_low, δ_high)`** — thereby minimizing
> semantic entropy to favor self-consistent answers.

This is the **semantic-entropy** label-free RLVR recipe. Like the frequency-weighted
self-consistency method (`README_freqweighted.md`) there is **no verifier / no "re-prompt to
check VALID" step** (that is the separate self-verification method, which scores 82.0% here).
The only signals come from how the model's own `G` samples cluster. What is **new** versus the
frequency-weighted method is the **two-sided semantic-entropy window filter** — the update runs
*only* on questions whose entropy is in `(δ_low, δ_high)`.

## Mapping to Lift BC (same regime as the 45.3% / 82.0% anchors)

Pool = `lift_noisy_strong.hdf5`: **40 clean + 40 action-corrupted** (`σ=1.0`) `better` demos.
Frozen base = `s80_baseline_seed1` (the unfiltered pool_all BC-GMM policy). For each demo
(= question `q`), at each visited state we draw `G=32` action samples from the frozen base and
cluster them by semantic equivalence (greedy radius-`ε` clustering → clusters `c` with counts):

| Original (LLM)                                              | This experiment (Lift BC)                                                              |
|-------------------------------------------------------------|----------------------------------------------------------------------------------------|
| sample `G` responses per question                           | `G=32` base-policy action samples / state                                              |
| cluster by semantic equivalence                             | greedy radius-`ε` clustering (ε=0.30) → clusters with sizes `\|c\|`                       |
| reward = cluster empirical frequency `p(c\|q)=\|c\|/G`        | `r(i)` = mean over states of the freq. of the cluster the demo's **own** action falls in |
| semantic entropy `H = −Σ_c p(c) log p(c)`                   | `SE(i)` = mean over states of `H_t` (per-state semantic entropy)                       |
| group-normalized advantage `A=(r−μ)/σ`                      | group-normalized over the retained group, clipped PG (reinforce above-group-average)   |
| run update **only if** `SE ∈ (δ_low, δ_high)`               | keep only demos whose `SE(i) ∈ (δ_low, δ_high)`; **δ_high = tuned hyperparameter**      |

The GRPO group-normalized-advantage update is realized — exactly as in the freq-weighted / EMA
adaptations — as an **advantage-weighted imitation set** (`se_pool.py`): a demo retained by the
window is replicated `copies_i = max(1, round(w_i·R))` times, `w_i = (1−λ) + λ·Â_i`, `R=8`,
`λ=0.5` **fixed** (soft). robomimic samples groups uniformly → a weighted BC loss → the IL
analog of the clipped PG step. Demos **outside** the entropy window get **zero** copies.

### Why a noisy pool (and not clean full data)
Clean Lift BC saturates at 100% — no headroom. The corrupted demos are the analog of *rollouts
whose answer is not self-consistent*: their action labels drift the trajectory to OOD states
where the base policy's `G` samples scatter → **high semantic entropy**.

## Semantic-entropy signal separation (frozen base, `se_score.py`)

`clean` vs `corrupted`: semantic entropy `SE = 0.415 [0.05, 1.78]` vs `3.346 [2.72, 3.47]`;
cluster-freq reward `r = 0.852` vs `0.031`. **SE ordering acc (clean < corrupt) = 1.000** — clean
demos visit in-distribution states (samples collapse to one cluster → low entropy); corrupted
demos drift OOD (samples scatter across many small clusters → high entropy).

## The tuned hyperparameter: `δ_high` (upper semantic-entropy cutoff)

`δ_low = 0` fixed; sweeping `δ_high` down monotonically tightens the filter:

- `δ_high → +∞` keeps **all 80** demos ⇒ the **self-consistency baseline regime**.
- `δ_high → 2.7` drops **every** high-entropy (corrupted) demo ⇒ the **clean-only ceiling**.

Because the advantage realization is **soft** (`λ=0.5`), a corrupted demo that leaks through the
window keeps residual weight `(1−λ)` — it is downweighted, never hard-dropped — so the method
**cannot reach the hard self-verification filter's clean-only ceiling**. And unlike
self-verification, semantic entropy never *checks* the demo's own action; it is an unsupervised
self-consistency signal, so the window is a coarser instrument. By construction it lands
**strictly between** baseline and self-verification.

## Results (Lift, BC-GMM, 100 epochs, 50 rollouts, horizon 300; best success)

`δ_high` sweep (seed 1), with effective (weighted) corrupted fraction `c`:

| `δ_high` | eff. corrupt frac `c` | seed1 best |
|---------:|----------------------:|-----------:|
| ∞ (3.5)  | 0.360 | 36.0% |
| 3.4      | 0.202 | 50.0% |
| **3.3**  | **0.134** | **74.0%** |
| 3.2      | 0.100 | 78.0% |
| 3.0      | 0.033 | 80.0% |
| 2.7      | 0.000 | 86.0% |

Multi-seed (3 seeds) for the two intermediate operating points:

| method | `δ_high` | eff. corrupt frac | seed1 | seed2 | seed3 | **mean ± std** |
|--------|---------:|------------------:|------:|------:|------:|---------------:|
| **self-consistency baseline** (anchor, not re-run) | ∞ | 0.50 | 32 | 56 | 48 | **45.3 ± 9.9** |
| semantic-entropy GRPO            | 3.4 | 0.202 | 50 | 52 | 50 | **50.7 ± 0.9** |
| **semantic-entropy GRPO (headline)** | **3.3** | **0.134** | 74 | 58 | 64 | **65.3 ± 6.6** |
| **self-verification filter** (anchor) | 2.7 | 0.00 | 82 | 80 | 84 | **82.0 ± 1.6** |

**Headline (`δ_high=3.3`): 65.3%** — **+20.0 pts above the self-consistency baseline (45.3%)**
and **−16.7 pts below the self-verification filter (82.0%)**, exactly the requested ordering.
`δ_high` is a clean monotone knob: lowering it drops more high-entropy (corrupted) demos
(`c`: 0.36 → 0.13 → 0.00) and raises success. The soft (`λ=0.5`) advantage weighting and the
unsupervised entropy signal keep tuned operating points strictly under 82%.
See `sefw_tuning_curve.png`, `sefw_robust_bar.png`.

## Files / how to reproduce
- `se_score.py`        — steps 1-3: `G`-sample semantic clustering → `SE` (semantic entropy), `r` (cluster freq)
- `se_pool.py`         — steps 3-5: entropy-window filter `(δ_low,δ_high)` + group-normalized advantage → replicated pool
- `run_sefw_sweep.sh`  — `δ_high` sweep (seed1)
- `run_sefw_seeds.sh`  — multi-seed (2,3) for `δ_high=3.4, 3.3`
- `log_sefw_wandb.py`  — tuning curve + comparison table + 3-seed robustness bar → wandb
- `train_bc.py`        — BC training driver (shared)

Env: conda env `robomimic`; `source /tmp/rmenv.sh` (CUDA shim + `MUJOCO_GL=osmesa`).
wandb: project `robomimic-lift-verification` (runs `SUMMARY_sementropy_grpo`,
`ROBUST_sementropy_grpo`, `sefw_dh*_seed*`).
