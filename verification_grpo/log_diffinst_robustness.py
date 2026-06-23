"""
Log the multi-seed robustness check for the different-instantiation experiment:
current self-verification (tau=0.3) vs the M=8 different-instantiation ensemble
(purity 1.0), over seeds 1-4. Writes per-seed runs + a ROBUST_diffinst summary
run with mean/std and the seed-wise gap.
"""
import argparse, os, re
import numpy as np

RUNS = "/root/rm_runs_diffinst"


def best(name):
    p = f"{RUNS}/{name}.log"
    if not os.path.exists(p):
        return None
    v = [float(x) for x in re.findall(r'"Success_Rate":\s*([0-9.]+)', open(p, errors="ignore").read())]
    return max(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="robomimic-lift-verification")
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()
    os.environ.pop("WANDB_MODE", None)
    import wandb

    # seed 1 came from the main sweep (di_baseline / di_current_tau030 / di_M8); seeds 2-4 from rob_*.
    base = {1: best("di_baseline"), 2: best("rob_baseline_seed2"),
            3: best("rob_baseline_seed3"), 4: best("rob_baseline_seed4")}
    cur = {1: best("di_current_tau030"), 2: best("rob_current_seed2"),
           3: best("rob_current_seed3"), 4: best("rob_current_seed4")}
    m8 = {1: best("di_M8"), 2: best("rob_M8_seed2"),
          3: best("rob_M8_seed3"), 4: best("rob_M8_seed4")}
    seeds = [s for s in [1, 2, 3, 4] if cur[s] is not None and m8[s] is not None]

    for s in seeds:
        for name, val, meth, M in [(f"diffinst_baseline_seed{s}", base.get(s), "no filtering (all 80)", None),
                                   (f"diffinst_current_seed{s}", cur[s], "current self-verif τ=0.3", 1),
                                   (f"diffinst_M8_seed{s}", m8[s], "diff-instantiation ensemble", 8)]:
            if val is None:
                continue
            r = wandb.init(project=args.project, entity=args.entity, name=name, reinit=True,
                           config={"regime": "adv_bias0.45", "tau": 0.3, "method": meth, "M": M, "seed": s})
            r.summary["best_success_rate"] = val
            r.finish()
            print("  logged %-26s best=%.3f" % (name, val))

    bb = np.array([base[s] for s in seeds if base.get(s) is not None])
    cb = np.array([cur[s] for s in seeds]); mb = np.array([m8[s] for s in seeds])
    have_base = len(bb) == len(seeds)
    srun = wandb.init(project=args.project, entity=args.entity, name="ROBUST_diffinst", reinit=True,
                      config={"regime": "adv_bias0.45", "tau": 0.3, "seeds": seeds})
    tbl = wandb.Table(columns=["seed", "baseline_no_filter", "current_tau0.3", "diffinst_M8",
                               "gap_vs_current_pts", "gap_vs_baseline_pts"])
    for s in seeds:
        tbl.add_data(s, base.get(s), cur[s], m8[s], 100 * (m8[s] - cur[s]),
                     (100 * (m8[s] - base[s])) if base.get(s) is not None else None)
    wandb.log({"robustness_table": tbl})
    srun.summary["n_seeds"] = len(seeds)
    if have_base:
        srun.summary["baseline_mean"] = float(bb.mean()); srun.summary["baseline_std"] = float(bb.std())
        srun.summary["mean_gap_vs_baseline_pts"] = float(100 * (mb.mean() - bb.mean()))
    srun.summary["current_mean"] = float(cb.mean()); srun.summary["current_std"] = float(cb.std())
    srun.summary["M8_mean"] = float(mb.mean()); srun.summary["M8_std"] = float(mb.std())
    srun.summary["mean_gap_pts"] = float(100 * (mb.mean() - cb.mean()))
    srun.summary["min_seed_gap_pts"] = float(100 * (mb - cb).min())
    srun.finish()
    print("\n=== ROBUSTNESS (%d seeds) ===" % len(seeds))
    if have_base:
        print("  baseline no filter       : %.1f ± %.1f %%" % (100 * bb.mean(), 100 * bb.std()))
    print("  current self-verif τ=0.3 : %.1f ± %.1f %%" % (100 * cb.mean(), 100 * cb.std()))
    print("  diff-instantiation M=8   : %.1f ± %.1f %%" % (100 * mb.mean(), 100 * mb.std()))
    print("  mean gap vs current = %+.1f pts  (per-seed min %+.1f)" % (100 * (mb.mean() - cb.mean()), 100 * (mb - cb).min()))
    if have_base:
        print("  mean gap vs baseline = %+.1f pts" % (100 * (mb.mean() - bb.mean())))


if __name__ == "__main__":
    main()
