"""
Push the different-instantiation verification experiment to wandb.

Algorithm under test: "Keep the generator as-is, but compute v̄ under a DIFFERENT
INSTANTIATION of the same model." Each verifier instantiation is the same BC-GMM
model re-instantiated on a different 55% subsample of the noisy pool; the
verification score v̄ for a demo is the out-of-bag mean VALID fraction over the M
instantiations that did NOT train on it. M (number of instantiations) is the tuned
hyperparameter; M=1 collapses to a single-instantiation verifier.

Logs to project robomimic-lift-verification (entity = key default):
  - one run per downstream BC policy (rollout success curve):
      diffinst_baseline            (no filter, all 80 demos)
      diffinst_current_tau0.3      (current method: single full-data self-verifier, tau=0.3)
      diffinst_M{M}_tau0.3         (new method, M instantiations, tau=0.3)
  - SUMMARY_diffinst: M-tuning table (M, ordering_acc, purity, n_keep, best_success,
      gap_vs_current) + tuning-curve image.
"""
import argparse, json, os, re
import numpy as np
import h5py

RUNS = "/root/rm_runs_diffinst"
DS = "datasets/lift/mh/lift_adv_b045.hdf5"
VOUT = f"{RUNS}/verify_sub_b045"


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


def key_stats(ds, key):
    f = h5py.File(ds, "r")
    clean = set(d.decode() for d in f["mask/clean"][:])
    corr = set(d.decode() for d in f["mask/corrupted"][:])
    if "mask/%s" % key not in f:
        f.close(); return None
    keep = [d.decode() for d in f["mask/%s" % key][:]]
    f.close()
    nc = sum(d in clean for d in keep); nk = sum(d in corr for d in keep)
    return {"n_keep": len(keep), "n_clean": nc, "n_corrupt": nk,
            "purity": nc / max(len(keep), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="robomimic-lift-verification")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--bias", type=float, default=0.45)
    ap.add_argument("--eps", type=float, default=0.30)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--plot", default=f"{RUNS}/diffinst_tuning.png")
    args = ap.parse_args()
    os.environ.pop("WANDB_MODE", None)

    sep = json.load(open(f"{VOUT}/ensemble_scores.json"))["per_M"]
    base_cfg = {"regime": f"adv_bias{args.bias}", "task": "Lift", "tau": args.tau, "eps": args.eps}

    # ---- collect rows ----
    rows = []  # dict per policy
    # baseline
    c, b = parse(f"{RUNS}/di_baseline.log"); st = key_stats(DS, "pool_all")
    rows.append({"name": "diffinst_baseline", "method": "no_filter", "M": None,
                 "curve": c, "best": b, "ordering_acc": None, **(st or {})})
    # current method
    c, b = parse(f"{RUNS}/di_current_tau030.log"); st = key_stats(DS, "cur_tau030")
    rows.append({"name": "diffinst_current_tau0.3",
                 "method": "current: single full-data self-verifier", "M": 1,
                 "curve": c, "best": b, "ordering_acc": None, **(st or {})})
    cur_best = b
    # new method M-sweep
    for M in args.Ms:
        c, b = parse(f"{RUNS}/di_M{M}.log")
        s = sep.get(str(M), {})
        rows.append({"name": f"diffinst_M{M}_tau0.3",
                     "method": "diff-instantiation OOB ensemble", "M": M,
                     "curve": c, "best": b, "ordering_acc": s.get("ordering_acc"),
                     "n_keep": s.get("n_keep"), "n_clean": s.get("n_clean_kept"),
                     "n_corrupt": s.get("n_corrupt_kept"), "purity": s.get("purity")})

    import wandb
    logged = 0
    for r in rows:
        if r["best"] is None:
            print("  skip (no log yet):", r["name"]); continue
        cfg = {**base_cfg, "method": r["method"], "M": r["M"],
               "n_keep": r.get("n_keep"), "purity": r.get("purity"),
               "ordering_acc": r.get("ordering_acc")}
        run = wandb.init(project=args.project, entity=args.entity, name=r["name"],
                         config=cfg, reinit=True)
        for ep, sr in r["curve"]:
            wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
        run.summary["best_success_rate"] = r["best"]
        run.summary["final_success_rate"] = r["curve"][-1][1]
        for k in ["ordering_acc", "purity", "n_keep", "n_clean", "n_corrupt"]:
            if r.get(k) is not None:
                run.summary[k] = r[k]
        run.finish(); logged += 1
        print("  logged %-26s best=%.3f  purity=%s  ord_acc=%s"
              % (r["name"], r["best"], r.get("purity"), r.get("ordering_acc")))

    # ---- tuning-curve plot ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    msweep = [r for r in rows if r["method"].startswith("diff-inst") and r["best"] is not None]
    msweep.sort(key=lambda r: r["M"])
    base_best = next((r["best"] for r in rows if r["method"] == "no_filter"), None)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    if msweep:
        xs = [r["M"] for r in msweep]; ys = [100 * r["best"] for r in msweep]
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=2.2, ms=8,
                label="different-instantiation ensemble (τ=0.3)")
        for r in msweep:
            ax.annotate("p=%.2f" % (r["purity"] or 0), (r["M"], 100 * r["best"]),
                        textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")
    if cur_best is not None:
        ax.axhline(100 * cur_best, ls="--", color="#d62728", lw=2,
                   label="current verification-filtered (τ=0.3, single self-verifier)")
    if base_best is not None:
        ax.axhline(100 * base_best, ls=":", color="#7f7f7f", lw=1.8, label="no filtering (all 80)")
    ax.set_xlabel("M = number of independent model instantiations averaged")
    ax.set_ylabel("Lift rollout success rate (%)")
    ax.set_title("Different-instantiation verification vs current self-verification (Lift, bias=%.2f)" % args.bias)
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout(); fig.savefig(args.plot, dpi=130)
    print("saved plot ->", args.plot)

    # ---- SUMMARY ----
    srun = wandb.init(project=args.project, entity=args.entity, name="SUMMARY_diffinst",
                      config=base_cfg, reinit=True)
    tbl = wandb.Table(columns=["M", "method", "ordering_acc", "purity", "n_keep",
                               "best_success_rate", "gap_vs_current_pts"])
    for r in rows:
        gap = (100 * (r["best"] - cur_best)) if (r["best"] is not None and cur_best is not None) else None
        tbl.add_data(r["M"], r["method"], r.get("ordering_acc"), r.get("purity"),
                     r.get("n_keep"), r["best"], gap)
        if r["best"] is not None and r["M"] is not None and r["method"].startswith("diff-inst"):
            wandb.log({"tuning/M": r["M"], "tuning/best_success_rate": r["best"],
                       "tuning/ordering_acc": r.get("ordering_acc"), "tuning/gap_vs_current": gap})
    wandb.log({"diffinst_tuning_table": tbl})
    srun.summary["baseline_best"] = base_best
    srun.summary["current_tau0.3_best"] = cur_best
    best_new = max([r for r in msweep], key=lambda r: r["best"], default=None)
    if best_new is not None:
        srun.summary["best_M"] = best_new["M"]
        srun.summary["best_M_success"] = best_new["best"]
        if cur_best is not None:
            srun.summary["best_gap_vs_current_pts"] = 100 * (best_new["best"] - cur_best)
    if os.path.exists(args.plot):
        wandb.log({"diffinst_tuning_curve": wandb.Image(args.plot)})
    srun.finish()
    print("\nLogged %d runs + SUMMARY_diffinst to %s" % (logged, args.project))


if __name__ == "__main__":
    main()
