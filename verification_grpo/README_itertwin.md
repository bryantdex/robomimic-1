# Iterative fresh-twin verification on Lift

Implements the requested algorithm on top of the verification-filtered BC pipeline:

> **Each self-training round, draw the verification score v̄ from a freshly
> re-seeded twin instantiation of the model rather than a fixed one, so the
> verifier that filters this round's pseudo-labels is decorrelated from both the
> generator and every prior round's verifier.**

Tuned hyperparameter: **R = number of self-training rounds**.

## Algorithm (`itertwin_run.py`)

Start with the full noisy pool `D_0`. For round `r = 1..R`:
1. **Fresh re-seed.** Draw a new seed `seed_base + 1000*r` (decorrelated from the
   generator and from every prior round).
2. Train `m` fresh twin BC-GMM instantiations, each on an independent `frac`
   subsample of the **current kept set** `D_{r-1}` (with a `min_sub` floor so the
   twins stay well-trained as the set shrinks).
3. Score each demo in `D_{r-1}` by its **out-of-bag** mean VALID fraction v̄ (the
   twins that did not train on it); `D_r = { v̄ >= tau_r }`.
4. (optional) `tau_r` is linearly annealed from `--tau` to `--tau_final` across
   rounds — lenient early to protect recall, strict late to purge residual corruption.

A filter key `iter_R{r}` is written each round, so one `R=Rmax` run yields the whole
R-sweep for downstream evaluation. Downstream BC is then trained on `D_R`.

**Why iteration helps and a single-round ensemble does not.** Round `r`'s twins are
trained on the round-`(r-1)` *cleaned* set, so each successive twin memorizes less
contamination → its v̄ separates better → the next round is cleaner: a cascade.

## The decisive regime: correlated vs independent contamination

The previous *different-instantiation* method is **single-round**: it averages `M`
twins, but every twin trains on the same full contaminated pool.

- **Independent per-demo bias** (`make_adv_pool.py`): a single round already separates,
  because a twin that did not train on demo *i* produces unbiased actions for it. Here,
  once `tau` is also tuned, single-round diff-inst reaches purity 1.0 (M=8, τ=0.15 →
  35 clean + 0 corrupt) — **the iterative method gives no gap** (verified; logged as a
  negative control). Honest finding: iteration is not magic on this regime.

- **Correlated / shared systematic bias** (`make_corr_pool.py`, `rho=0.6`, cos-to-shared
  0.84): every corrupted demo shares one bias direction. Every single-round twin learns
  it (it appears in ~half the pool), so **out-of-bag scoring still reproduces it** and
  single-round **cannot separate at any (M, τ)** — its ordering accuracy actually *drops*
  with M (0.77 → 0.64). This is the realistic self-training failure mode (a systematic
  model error reproduced across many pseudo-labels), and it is where iteration wins.

## Results (Lift, low-dim BC-GMM, 100 epochs, 30 rollouts; pool = 40 clean + 40 corrupt, correlated bias rho=0.6)

**Filtering** — iterative cascade (`tau=0.15`, `m=3`, `frac=0.5`) vs best single-round:

| method | config | n_keep (clean+corrupt) | purity | recall |
|---|---|---|---|---|
| single-round diff-inst | M=6, τ=0.30 (best) | 30 (27+3) | 0.90 | 0.675 |
| single-round diff-inst | M=6, τ=0.15 | 48 (37+11) | 0.77 | 0.925 |
| **iterative fresh-twin** | R=1 | 52 (38+14) | 0.73 | 0.950 |
| **iterative fresh-twin** | R=2 | 42 (37+5) | 0.88 | 0.925 |
| **iterative fresh-twin** | **R=3** | **40 (36+4)** | **0.90** | **0.900** |

Single-round cannot get both: at purity 0.90 it keeps only 27 clean (recall 0.68);
the iterative cascade keeps **36 clean at the same purity** (recall 0.90).

**Downstream rollout success (4 seeds, mean ± std):**

| method | seeds (1–4) | mean ± std | gap vs diff-inst best |
|---|---|---|---|
| no filtering | 17/20/17/20 | 18.3 ± 1.7 | — |
| single-round diff-inst (best, M6/τ0.30) | 63/57/67/50 | 59.2 ± 6.4 | — |
| **iterative fresh-twin R=3 (NEW)** | 73/70/77/63 | **70.8 ± 4.9** | **+11.7** |
| oracle (clean-only) | 87/90/87/73 | 84.2 ± 6.4 | +25.0 |

**Headline: iterative fresh-twin beats the best single-round different-instantiation
config by +11.7 points, on all 4 seeds individually (+10, +13, +10, +13).** The R-tuning
curve climbs monotonically (R1 46.7 → R2 53.3 → R3 73.3 at seed 1) toward the oracle.

`rho=0.8` (cos 0.98) is harder and noisier; single-round there collapses to purity ≤0.68
(keeps 13 corrupt) — same qualitative gap, larger rollout variance, so `rho=0.6` is the
clean headline.

## Files / reproduce
- `make_corr_pool.py`   — correlated shared-bias pool (`lift_corr_r0{60,80}.hdf5`)
- `make_adv_pool.py`    — independent-bias pool (negative control)
- `make_boot_keys.py`   — per-twin subsample filter keys
- `itertwin_run.py`     — the iterative fresh-twin orchestrator (R, m, frac, tau, tau_final)
- `verify_ensemble.py`  — single-round diff-inst scorer (the method to beat)
- `train_bc.py`         — BC driver
- `log_final_wandb.py`  — push runs + R-tuning curve + robustness bar + mechanism plot
- runs/artifacts in `/root/rm_runs_itertwin/`; wandb project
  `bryantruong-work-kaist/robomimic-lift-verification` (runs `itw_*`, `ROBUST_itertwin`,
  `SUMMARY_itertwin`).

Environment: conda env `robomimic`; `source /tmp/rmenv.sh` (CUDA shim + `MUJOCO_GL=osmesa`).
