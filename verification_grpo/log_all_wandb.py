"""
Push ALL experiment results to wandb: every run's rollout success curve, a summary
run with the tuning table + both plots, and the multi-seed mean/std.

Run inside conda env robomimic:
  WANDB_API_KEY=<key> python verification_grpo/log_all_wandb.py --entity <ENTITY> [--project NAME]
(or pass --api_key <key>)
"""
import argparse, glob, json, os, re
import numpy as np

RUNS_DIR = "/root/rm_runs"


def parse(path):
    if not os.path.exists(path):
        return [], None
    txt = open(path, errors="ignore").read()
    pairs = []
    for m in re.finditer(r"Epoch (\d+) Rollouts took", txt):
        tail = txt[m.end():m.end() + 2000]
        sm = re.search(r'"Success_Rate":\s*([0-9.]+)', tail)
        if sm:
            pairs.append((int(m.group(1)), float(sm.group(1))))
    pairs = sorted(set(pairs))
    return pairs, (max(p[1] for p in pairs) if pairs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=None, help="omit to use the API key's default entity")
    ap.add_argument("--project", default="robomimic-lift-verification")
    ap.add_argument("--api_key", default=None)
    args = ap.parse_args()
    if args.api_key:
        os.environ["WANDB_API_KEY"] = args.api_key
    os.environ.pop("WANDB_MODE", None)  # force online
    import wandb

    # config tables
    tau_keys = json.load(open(f"{RUNS_DIR}/verify_v80/filter_keys.json"))
    tau_meta = {0.0: 80}
    for v in tau_keys.values():
        tau_meta[v["tau"]] = v["n_keep"]

    # define all runs: (wandb_name, logpath, config)
    runs = []
    # reference: clean full-data baseline (saturates)
    runs.append(("ref_baseline_all300_clean", f"{RUNS_DIR}/baseline_seed1.log",
                 {"regime": "clean_full_data", "n_demos": 300}))
    # tuning sweep (sigma=0.5 pool of 80)
    runs.append(("v80_baseline_tau0.0", f"{RUNS_DIR}/v80_baseline_seed1.log",
                 {"regime": "noisy_pool80", "tau": 0.0, "n_keep": 80, "sigma": 0.5}))
    for tau, key in [(0.1, "010"), (0.3, "030"), (0.5, "050"), (0.7, "070"), (0.85, "085")]:
        runs.append((f"v80_filtered_tau{tau}", f"{RUNS_DIR}/v80_filtered_tau{key}_seed1.log",
                     {"regime": "noisy_pool80", "tau": tau, "n_keep": tau_meta.get(tau), "sigma": 0.5}))
    # multi-seed robustness (sigma=1.0)
    for s in [1, 2, 3]:
        runs.append((f"s80_baseline_seed{s}", f"{RUNS_DIR}/s80_baseline_seed{s}.log",
                     {"regime": "robustness_sigma1.0", "tau": 0.0, "seed": s, "sigma": 1.0}))
        runs.append((f"s80_filtered_tau0.3_seed{s}", f"{RUNS_DIR}/s80_filtered_seed{s}.log",
                     {"regime": "robustness_sigma1.0", "tau": 0.3, "seed": s, "sigma": 1.0}))

    logged = 0
    for name, path, cfg in runs:
        curve, best = parse(path)
        if best is None:
            print("  skip (no data):", name)
            continue
        run = wandb.init(project=args.project, entity=args.entity, name=name, config=cfg, reinit=True)
        for (ep, sr) in curve:
            wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
        run.summary["best_success_rate"] = best
        run.summary["final_success_rate"] = curve[-1][1]
        run.finish()
        logged += 1
        print("  logged %-30s best=%.3f" % (name, best))

    # summary run
    srun = wandb.init(project=args.project, entity=args.entity, name="SUMMARY", reinit=True)
    # tuning table
    tbl = wandb.Table(columns=["tau", "n_keep", "best_success_rate"])
    base80 = parse(f"{RUNS_DIR}/v80_baseline_seed1.log")[1]
    for tau, key in [(0.0, None), (0.1, "010"), (0.3, "030"), (0.5, "050"), (0.7, "070"), (0.85, "085")]:
        p = f"{RUNS_DIR}/v80_baseline_seed1.log" if key is None else f"{RUNS_DIR}/v80_filtered_tau{key}_seed1.log"
        b = parse(p)[1]
        tbl.add_data(tau, tau_meta.get(tau), b)
    wandb.log({"tuning_table": tbl})
    # multi-seed stats
    bb = [parse(f"{RUNS_DIR}/s80_baseline_seed{s}.log")[1] for s in [1, 2, 3]]
    ff = [parse(f"{RUNS_DIR}/s80_filtered_seed{s}.log")[1] for s in [1, 2, 3]]
    srun.summary["robust_baseline_mean"] = float(np.mean(bb))
    srun.summary["robust_baseline_std"] = float(np.std(bb))
    srun.summary["robust_filtered_mean"] = float(np.mean(ff))
    srun.summary["robust_filtered_std"] = float(np.std(ff))
    srun.summary["robust_gap_pts"] = float(100 * (np.mean(ff) - np.mean(bb)))
    for img in ["tuning_curve.png", "robustness_bar.png"]:
        p = f"{RUNS_DIR}/{img}"
        if os.path.exists(p):
            wandb.log({img.replace(".png", ""): wandb.Image(p)})
    srun.finish()
    print("\nLogged %d runs + SUMMARY to %s/%s" % (logged, args.entity, args.project))


if __name__ == "__main__":
    main()
