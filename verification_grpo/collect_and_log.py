"""
Collect rollout success-rate curves from robomimic run logs and (optionally)
replay them into wandb as proper runs, plus a summary comparison.

Each robomimic eval prints a block like:
    Epoch 50 Rollouts took ...
    { ... "Success_Rate": 0.83, ... }
We parse (epoch, Success_Rate) pairs from each run's stdout log.

Usage:
  # just print/save the table (no wandb):
  python collect_and_log.py --runs name1:log1.log name2:log2.log --out results.json
  # also push to wandb (requires WANDB_API_KEY + --entity):
  python collect_and_log.py --runs ... --out results.json --wandb --entity ENTITY --project PROJ
"""
import argparse
import json
import re
import os


def parse_log(path):
    """Return list of (epoch, success_rate) and the best success rate."""
    with open(path, "r", errors="ignore") as f:
        txt = f.read()
    pairs = []
    # find "Epoch N Rollouts" headers, then the next "Success_Rate": x after it
    for m in re.finditer(r"Epoch (\d+) Rollouts took", txt):
        epoch = int(m.group(1))
        tail = txt[m.end():m.end() + 2000]
        sm = re.search(r'"Success_Rate":\s*([0-9.]+)', tail)
        if sm:
            pairs.append((epoch, float(sm.group(1))))
    # also capture final-epoch training loss curve if desired (skip)
    pairs = sorted(set(pairs))
    best = max([p[1] for p in pairs], default=None)
    return pairs, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="name:logpath pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default="robomimic-lift-verification")
    p.add_argument("--meta", default=None, help="optional json mapping run name -> extra config (tau, n_keep, etc.)")
    p.add_argument("--plot", default=None, help="path to save tuning-curve png (success vs tau)")
    p.add_argument("--baseline_name", default=None, help="run name to treat as baseline for gap calc")
    args = p.parse_args()

    meta = {}
    if args.meta and os.path.exists(args.meta):
        meta = json.load(open(args.meta))

    results = {}
    for spec in args.runs:
        name, path = spec.split(":", 1)
        pairs, best = parse_log(path)
        results[name] = {"curve": pairs, "best_success_rate": best,
                         "final_success_rate": pairs[-1][1] if pairs else None,
                         "config": meta.get(name, {})}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY: best rollout success rate ===")
    base_name = args.baseline_name or "baseline_all300_seed1"
    base = results.get(base_name, {}).get("best_success_rate")
    for name, r in results.items():
        gap = ""
        if base is not None and r["best_success_rate"] is not None:
            gap = "  (gap vs baseline: %+.1f pts)" % (100 * (r["best_success_rate"] - base))
        cfg = r["config"]
        cfgs = (" [tau=%.3f keep=%d]" % (cfg["tau"], cfg["n_keep"])) if "tau" in cfg else ""
        print("  %-26s best=%s final=%s%s%s" % (
            name,
            ("%.3f" % r["best_success_rate"]) if r["best_success_rate"] is not None else "NA",
            ("%.3f" % r["final_success_rate"]) if r["final_success_rate"] is not None else "NA",
            cfgs, gap))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = []
        for name, r in results.items():
            cfg = r["config"]
            if "tau" in cfg and r["best_success_rate"] is not None:
                pts.append((cfg["tau"], r["best_success_rate"], cfg.get("n_keep")))
        pts.sort()
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        if pts:
            xs = [p[0] for p in pts]; ys = [100 * p[1] for p in pts]
            ax.plot(xs, ys, "o-", color="#1f77b4", lw=2, ms=7, label="verification-filtered BC")
            for (t, s, nk) in pts:
                if nk is not None:
                    ax.annotate("n=%d" % nk, (t, 100 * s), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
        if base is not None:
            ax.axhline(100 * base, ls="--", color="#d62728", lw=2, label="baseline (no filtering, τ=0)")
        ax.set_xlabel("verification threshold τ  (skip demos with v̄ < τ)")
        ax.set_ylabel("Lift rollout success rate (%)")
        ax.set_title("Verification-filtered BC vs baseline on noisy Lift pool")
        ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print("saved plot ->", args.plot)

    if args.wandb:
        import wandb
        for name, r in results.items():
            run = wandb.init(project=args.project, entity=args.entity, name=name,
                             config=r["config"], reinit=True)
            for (epoch, sr) in r["curve"]:
                wandb.log({"rollout/success_rate": sr, "epoch": epoch}, step=epoch)
            wandb.summary["best_success_rate"] = r["best_success_rate"]
            wandb.summary["final_success_rate"] = r["final_success_rate"]
            run.finish()
        # summary run: tuning curve (success vs tau) + plot image
        srun = wandb.init(project=args.project, entity=args.entity, name="SUMMARY_tuning_curve", reinit=True)
        tbl = wandb.Table(columns=["tau", "n_keep", "best_success_rate", "gap_vs_baseline"])
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["config"].get("tau", -1)):
            cfg = r["config"]
            if "tau" in cfg and r["best_success_rate"] is not None:
                gap = (r["best_success_rate"] - base) if base is not None else None
                tbl.add_data(cfg["tau"], cfg.get("n_keep"), r["best_success_rate"], gap)
                wandb.log({"tuning/best_success_rate": r["best_success_rate"],
                           "tuning/n_keep": cfg.get("n_keep"), "tuning/tau": cfg["tau"]})
        srun.summary["baseline_best_success_rate"] = base
        wandb.log({"tuning_table": tbl})
        if args.plot and os.path.exists(args.plot):
            wandb.log({"tuning_curve": wandb.Image(args.plot)})
        srun.finish()
        print("\nLogged %d runs + summary to wandb (%s/%s)" % (len(results), args.entity, args.project))


if __name__ == "__main__":
    main()
