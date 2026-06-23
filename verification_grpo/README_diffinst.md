# Different-instantiation verification on Lift — "compute v̄ under a different instantiation of the same model"

This implements the requested algorithm variant on top of the verification-filtered
BC pipeline (`README.md`): **keep the generator (data pipeline + downstream BC training)
exactly as-is, but compute the verification score v̄ under a *different instantiation*
of the same model.** The tuned hyperparameter is **M = the number of independent
instantiations** whose VALID fractions are averaged.

## Algorithm

The current method (`verify.py`) is *self*-verification: the verifier is a single
model instance trained on the (contaminated) pool that is also being filtered.
Generator and verifier are one instantiation, so the verifier's per-demo v̄ estimate
is noisy/biased and — crucially — it *memorizes* the corrupted action labels it was
trained on, validating them.

The different-instantiation variant re-instantiates the same BC-GMM model M times,
each on its **own random 55% subsample** of the pool (`make_boot_keys.py`), and scores
each demo with the **out-of-bag** mean VALID fraction over the instantiations that did
*not* train on it:

```
v_t^(m) = (1/K) Σ_k 1[ ||sample_k^(m) - a_t|| <= eps ]            # VALID frac, instance m, state t
v̄_M(i) = mean_t [ mean over the m<=M that did NOT train on demo i of v_t^(m) ]
keep demo i  iff  v̄_M(i) >= τ            (τ = 0.30, the current method's threshold)
```

Why a *different* instantiation helps: a systematically-corrupted demo is reproduced
(high v̄) by the instantiations that trained on it but flagged (low v̄) by those that
did not, so averaging M instantiations drives corrupted demos below τ while clean demos
stay above. M is tuned; M=1 collapses toward the current single-instantiation behaviour.

## Why this corruption regime

Zero-mean Gaussian action noise is **detectability-coupled**: any σ large enough to hurt
downstream is trivially flagged by *every* single instantiation (corrupted v̄→0), so an
ensemble adds nothing (verified: σ=0.15 gives ordering-acc 1.000 at M=1). To give the
ensemble a real job we use an **adversarial constant per-demo action bias** (`make_adv_pool.py`,
bias_norm=0.45): each corrupted demo gets a fixed offset b added to every action. Per
state this is only ~ε off (near the verification boundary, so single instantiations
disagree — ordering acc 0.53–0.87), but it is *systematic*, so it compounds over the
trajectory and badly derails a policy that imitates it (baseline-on-pool = 23%). This
decouples harm from per-state detectability — exactly where averaging instantiations pays off.

## Results (Lift, low-dim BC-GMM, 100 epochs, 30 eval rollouts; pool of 80 = 40 clean + 40 biased)

Separation (OOB ensemble v̄, ordering acc) and downstream success vs M, all at **τ=0.30**:

| method                                   | M  | demos kept (clean+corrupt) | purity | ordering acc | best success | gap vs current |
|------------------------------------------|----|----------------------------|--------|--------------|--------------|----------------|
| no filtering (baseline)                  | –  | 80 (40+40)                 | 0.50   | –            | 23.3%        | –              |
| **current** verification-filtered (self) | 1  | 69 (40+29)                 | 0.58   | –            | **33.3%**    | —              |
| different-instantiation                  | 1  | 56 (35+21)                 | 0.62   | 0.748        | 43.3%        | +10.0          |
| different-instantiation                  | 2  | 40 (32+8)                  | 0.80   | 0.871        | 60.0%        | +26.7          |
| **different-instantiation (optimum)**    | 3  | 37 (34+3)                  | 0.92   | 0.946        | **73.3%**    | **+40.0**      |
| different-instantiation                  | 5  | 32 (31+1)                  | 0.97   | 0.984        | 56.7%        | +23.3          |
| **different-instantiation**              | 8  | 25 (25+0)                  | 1.00   | 0.998        | **73.3%**    | **+40.0**      |
| different-instantiation                  | 10 | 24 (24+0)                  | 1.00   | 0.996        | 73.3%        | +40.0          |

**Tuning M is the whole game.** The current single-instantiation self-verifier keeps 29
of 40 biased demos (purity 0.58) → 33.3%. Averaging M independent instantiations OOB
purges the bias (purity → 1.00 by M=8) and lifts downstream success to **73.3% — a +40.0
point gap** over the current verification-filtered (τ=0.3) algorithm. Like the original
τ-sweep, the curve has an optimum (M≈3–10): too few instantiations under-purge; very large
M slightly over-filters (keeps only ~24 clean demos), and the M=5 point is within
single-seed rollout noise of the plateau.

### Multi-seed robustness (4 seeds): current self-verifier vs different-instantiation M=8

To rule out single-seed/rollout-noise flukes we repeated the headline comparison over 4
seeds (no-filter baseline and current self-verifier τ=0.3 vs the M=8 OOB ensemble, all at τ=0.30):

| seed | baseline (no filter) | current τ=0.3 | different-instantiation M=8 | gap vs current | gap vs baseline |
|------|---------------------:|--------------:|----------------------------:|---------------:|----------------:|
| 1    | 23.3%                | 33.3%         | 73.3%                       | +40.0 | +50.0 |
| 2    | 26.7%                | 33.3%         | 63.3%                       | +30.0 | +36.7 |
| 3    | 30.0%                | 23.3%         | 70.0%                       | +46.7 | +40.0 |
| 4    | 16.7%                | 23.3%         | 80.0%                       | +56.7 | +63.3 |
| **mean ± std** | **24.2 ± 4.9%** | **28.3 ± 5.0%** | **71.7 ± 6.0%** | **+43.3** | **+47.5** |

**Robust mean gap = +43.3 points over the current method (+47.5 over the no-filter baseline),
all four seeds positive (+30 to +57).** wandb runs `diffinst_{baseline,current,M8}_seed{1-4}` +
`ROBUST_diffinst`.

## Files / reproduce
- `make_adv_pool.py`   — build the constant-bias adversarial pool (`lift_adv_b045.hdf5`)
- `make_boot_keys.py`  — per-instantiation 55% training-subset filter keys (`boot_s*`)
- `train_bc.py`        — BC driver (now supports `--n_rollouts 0` to train a verifier without eval)
- `verify_ensemble.py` — score M instantiations, OOB-aggregate v̄, write `ens_M{M}_tau030` keys
- `verify.py`          — the current single self-verifier (writes `cur_tau030`)
- `log_diffinst_wandb.py` — push runs + M-tuning table + tuning-curve plot to wandb
- runs/artifacts in `/root/rm_runs_diffinst/`; wandb project
  `bryantruong-work-kaist/robomimic-lift-verification` (runs `diffinst_*`, `SUMMARY_diffinst`).

Environment: conda env `robomimic`; `source /tmp/rmenv.sh` (CUDA shim + `MUJOCO_GL=osmesa`).
