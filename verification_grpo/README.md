# Verification-Filtered BC on Lift — a robomimic adaptation of the self-verification GRPO algorithm

## What the original algorithm is, and how it maps here

The requested algorithm is an **LLM self-training (label-free RLVR / GRPO)** method:

1. For each rollout `y_i` with answer `a_i`, re-prompt the model `K` times asking whether
   `a_i` satisfies the problem's constraints; `v(a_i)` = fraction returning VALID.
2. Pseudo-label `ŷ* = argmax_a [ n(a) · v̄(a) ]`, with `n(a)` the answer's rollout count.
3. Run GRPO with binary reward `r_i = 𝟙[answer(y_i) = ŷ*]`, **skipping prompts** where
   `max_a [n(a)·v̄(a)]` is below a threshold.

robomimic is an **imitation-learning** framework (no LLM / no GRPO), so we keep the algorithm's
*mechanism* — confidence-weighted pseudo-labeling + threshold filtering driven by **model
self-verification** — and realize it on the **Lift** task:

| Original (LLM)                                   | This experiment (Lift BC)                                                                 |
|--------------------------------------------------|-------------------------------------------------------------------------------------------|
| rollout `y_i` with answer `a_i`                  | a demonstration trajectory                                                                |
| re-prompt model `K` times → `v̄(a)` (frac VALID) | sample `K=32` actions from the base BC-GMM policy at each state; VALID = within `ε` of the demo action |
| `n(a)` rollout count                             | per-demo weight (= 1; each demo is one rollout)                                            |
| pseudo-label `ŷ* = argmax n·v̄`                  | the model's action consensus (validated demos)                                            |
| binary reward `𝟙[answer = ŷ*]`, skip < threshold | retrain BC keeping only demos with `v̄ ≥ τ` (**τ = tuned hyperparameter**)                |

## Why a noisy pool (and not clean full data)

Standard BC on the clean Lift datasets **saturates at 100%** (we measured baseline BC on all
300 multi-human demos = 100% best / 93% final). There is no headroom for *any* method to show a
gap. The algorithm's value only appears when some rollouts are **invalid** — so we build the
canonical noisy-demonstration regime:

- Pool = **40 clean** `better`-operator Lift demos + **40 action-corrupted** copies
  (`actions += N(0, σ=0.5)`, clipped). Corrupted demos are the analog of *rollouts whose answer
  does not satisfy the constraints*: they look like trajectories but their action labels are wrong.
- Baseline BC trains on all 80 → dragged down by the bad half.
- Self-verification (re-prompting the model) flags the corrupted demos; filtering removes them.

(We also checked the *natural* `worse`-operator MH demos: there, action-agreement verification
does **not** separate quality — `worse` demos differ by inefficiency/length, not by taking
locally-wrong actions — an honest negative result. Action-label corruption is the setting where
the algorithm's "re-prompt to check the answer" mechanism genuinely applies.)

## Results (Lift, low-dim BC-GMM, 100 epochs, 30 eval rollouts; pool of 80)

Self-verification separation: **clean v̄ ∈ [0.53, 0.96], corrupted v̄ = 0.000** (ordering acc = 1.000).

| run (τ) | demos kept | best success | gap vs baseline |
|---------|-----------:|-------------:|----------------:|
| baseline (τ=0, all 80) | 80 | **43.3%** | — |
| filtered τ=0.10 | 40 | **93.3%** | **+50.0 pts** |
| filtered τ=0.30 | 40 | **93.3%** | **+50.0 pts** |
| filtered τ=0.50 | 40 | 76.7% | +33.3 pts |
| filtered τ=0.70 | 37 | 90.0% | +46.7 pts |
| filtered τ=0.85 | 27 | 70.0% | +26.7 pts |

The hyperparameter **τ** has a clear optimum (τ≈0.1–0.3): keep everything that passes verification
(all 40 clean demos), drop everything that fails (all 40 corrupted). Too-high τ over-filters and
discards good data (n drops to 27 → 70%). **Every** τ beats the baseline by a large margin.
See `tuning_curve.png`.

### Multi-seed robustness (σ=1.0 corruption, 3 seeds, 50 rollouts)

To rule out the gap being a single-seed fluke we repeated the headline comparison (baseline vs
filtered τ=0.3) with stronger corruption (σ=1.0 ⇒ corrupted actions ≈ maximally invalid):

| condition | seed1 | seed2 | seed3 | **mean ± std** |
|-----------|------:|------:|------:|---------------:|
| baseline (no filtering) | 32% | 56% | 48% | **45.3 ± 9.9%** |
| verification-filtered (τ=0.3) | 82% | 80% | 84% | **82.0 ± 1.6%** |

**Robust gap = +36.7 points** (per-seed +50/+24/+36, all positive). The filtered policy is also far
more *stable* (±1.6 vs ±9.9). Self-verification separation stayed perfect at σ=1.0
(clean v̄ ∈ [0.35, 0.92], corrupted v̄ = 0.000, ordering acc = 1.000). See `robustness_bar.png`.

## Files / how to reproduce
- `train_bc.py`           — BC training driver (config + rollout eval + optional wandb)
- `make_noisy_dataset.py` — builds the clean+corrupted pool hdf5 (+ ground-truth filter keys)
- `verify.py`             — step 1-2: K-sample self-verification → v̄ → τ filter keys
- `run_all.sh`            — end-to-end: baseline → verify → τ sweep (pool of 80)
- `run_seeds.sh`          — multi-seed robustness for baseline vs filtered(τ=0.3)
- `collect_and_log.py`    — parse run logs → results table + tuning_curve.png + wandb replay
- `diag_signals.py`       — diagnostic: which per-demo signal separates quality

Environment: conda env `robomimic`; runs need `LD_LIBRARY_PATH=/root/cudashim MUJOCO_GL=osmesa`
(see `/tmp/rmenv.sh`).
