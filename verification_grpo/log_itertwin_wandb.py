"""
Push the ITERATIVE FRESH-TWIN verification experiment to wandb.

Algorithm under test: each self-training round draws v̄ from a freshly re-seeded
twin instantiation trained on the round's (progressively cleaned) kept set, so the
verifier is decorrelated from the generator and from every prior round's verifier.
Tuned hyperparameter: R = number of rounds.

Baselines for comparison (reused from the diff-instantiation experiment on the SAME
b045 pool / recipe): no-filter, current single self-verifier (tau=0.3), and the
single-round different-instantiation M-ensemble sweep (its best = 73.3%).

Project robomimic-lift-verification. Runs:
  itertwin_baseline, itertwin_current_tau0.3,
  itertwin_diffinst_M{M}      (single-round ensemble, reused)
  itertwin_R{r}_tau0.3        (new method, r rounds)
  itertwin_oracle             (train on the 40 truly-clean demos = ceiling)
  SUMMARY_itertwin            (R-tuning table vs diff-inst best + tuning-curve plot)
"""
import argparse, json, os, re
import numpy as np
import h5py

ITER = "/root/rm_runs_itertwin"
DI = "/root/rm_runs_diffinst"
DS = "datasets/lift/mh/lift_adv_b045.hdf5"


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
    return {"n_keep": len(keep), "n_clean": nc, "n_corrupt": nk, "purity": nc / max(len(keep), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="robomimic-lift-verification")
    ap.add_argument("--entity", default="bryantruong-work-kaist")
    ap.add_argument("--Rs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--di_Ms", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--plot", default=f"{ITER}/itertwin_tuning.png")
    args = ap.parse_args()
    os.environ.pop("WANDB_MODE", None)

    isum = json.load(open(f"{ITER}/main/itertwin_summary.json"))
    rounds = {r["round"]: r for r in isum["rounds"]}
    di_sep = {}
    di_sep_path = f"{DI}/verify_sub_b045/ensemble_scores.json"
    if os.path.exists(di_sep_path):
        di_sep = json.load(open(di_sep_path))["per_M"]

    base_cfg = {"regime": "adv_bias0.45", "task": "Lift", "tau": args.tau, "eps": isum["eps"],
                "method_family": "iterative-fresh-twin"}

    rows = []
    # --- baselines (reused diff-inst downstream logs, same pool/recipe) ---
    c, b = parse(f"{DI}/di_baseline.log"); st = key_stats(DS, "pool_all")
    rows.append({"name": "itertwin_baseline", "method": "no_filter", "x": None, "kind": "base",
                 "curve": c, "best": b, **(st or {})})
    c, b = parse(f"{DI}/di_current_tau030.log"); st = key_stats(DS, "cur_tau030")
    rows.append({"name": "itertwin_current_tau0.3", "method": "current single self-verifier", "x": None,
                 "kind": "base", "curve": c, "best": b, **(st or {})})
    cur_best = b
    base_best = rows[0]["best"]
    # --- single-round diff-instantiation sweep (reused) ---
    di_best = None
    for M in args.di_Ms:
        c, b = parse(f"{DI}/di_M{M}.log")
        s = di_sep.get(str(M), {})
        rows.append({"name": f"itertwin_diffinst_M{M}", "method": "diff-instantiation (single-round ensemble)",
                     "x": M, "kind": "diffinst", "curve": c, "best": b,
                     "ordering_acc": s.get("ordering_acc"), "n_keep": s.get("n_keep"),
                     "n_clean": s.get("n_clean_kept"), "n_corrupt": s.get("n_corrupt_kept"),
                     "purity": s.get("purity")})
        if b is not None:
            di_best = b if di_best is None else max(di_best, b)
    # --- NEW: iterative fresh-twin R-sweep ---
    for r in args.Rs:
        c, b = parse(f"{ITER}/it_R{r}.log")
        rs = rounds.get(r, {})
        rows.append({"name": f"itertwin_R{r}_tau0.3", "method": "iterative fresh-twin (NEW)",
                     "x": r, "kind": "iter", "curve": c, "best": b,
                     "ordering_acc": rs.get("ordering_acc"), "n_keep": rs.get("n_keep"),
                     "n_clean": rs.get("n_clean_kept"), "n_corrupt": rs.get("n_corrupt_kept"),
                     "purity": rs.get("purity"), "recall_clean": rs.get("recall_clean")})
    # --- oracle ceiling ---
    c, b = parse(f"{ITER}/it_oracle.log"); st = key_stats(DS, "clean")
    rows.append({"name": "itertwin_oracle", "method": "oracle (clean-only)", "x": None, "kind": "oracle",
                 "curve": c, "best": b, **(st or {})})

    import wandb
    logged = 0
    for r in rows:
        if r["best"] is None:
            print("  skip (no log yet):", r["name"]); continue
        cfg = {**base_cfg, "method": r["method"], "kind": r["kind"],
               "R_or_M": r.get("x"), "n_keep": r.get("n_keep"), "purity": r.get("purity"),
               "ordering_acc": r.get("ordering_acc"), "recall_clean": r.get("recall_clean")}
        run = wandb.init(project=args.project, entity=args.entity, name=r["name"], config=cfg, reinit=True)
        for ep, sr in r["curve"]:
            wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
        run.summary["best_success_rate"] = r["best"]
        run.summary["final_success_rate"] = r["curve"][-1][1]
        for k in ["ordering_acc", "purity", "n_keep", "n_clean", "n_corrupt", "recall_clean"]:
            if r.get(k) is not None:
                run.summary[k] = r[k]
        run.finish(); logged += 1
        print("  logged %-26s best=%.3f purity=%s recall=%s" %
              (r["name"], r["best"], r.get("purity"), r.get("recall_clean")))

    # ---- tuning-curve plot: iterative-R vs diff-inst-M, with baselines/oracle/ceiling lines ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    itr = sorted([r for r in rows if r["kind"] == "iter" and r["best"] is not None], key=lambda r: r["x"])
    dir_ = sorted([r for r in rows if r["kind"] == "diffinst" and r["best"] is not None], key=lambda r: r["x"])
    orc = next((r["best"] for r in rows if r["kind"] == "oracle"), None)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if dir_:
        ax.plot([r["x"] for r in dir_], [100 * r["best"] for r in dir_], "s--", color="#1f77b4",
                lw=2, ms=7, label="diff-instantiation single-round (vs M)")
    if itr:
        xs = [r["x"] for r in itr]; ys = [100 * r["best"] for r in itr]
        ax.plot(xs, ys, "o-", color="#2ca02c", lw=2.6, ms=9, label="iterative fresh-twin (vs R) [NEW]")
        for r in itr:
            ax.annotate("p=%.2f\nr=%.2f" % (r.get("purity") or 0, r.get("recall_clean") or 0),
                        (r["x"], 100 * r["best"]), textcoords="offset points", xytext=(0, 9),
                        fontsize=7.5, ha="center")
    if di_best is not None:
        ax.axhline(100 * di_best, ls="--", color="#1f77b4", lw=1.2, alpha=0.6,
                   label="diff-inst best (%.0f%%)" % (100 * di_best))
    if cur_best is not None:
        ax.axhline(100 * cur_best, ls="--", color="#d62728", lw=1.8, label="current self-verifier (%.0f%%)" % (100 * cur_best))
    if base_best is not None:
        ax.axhline(100 * base_best, ls=":", color="#7f7f7f", lw=1.6, label="no filtering (%.0f%%)" % (100 * base_best))
    if orc is not None:
        ax.axhline(100 * orc, ls="-.", color="#9467bd", lw=1.6, label="oracle clean-only (%.0f%%)" % (100 * orc))
    ax.set_xlabel("rounds R (iterative) / instantiations M (single-round)")
    ax.set_ylabel("Lift rollout success rate (%)")
    ax.set_title("Iterative fresh-twin verification vs single-round diff-instantiation (Lift, bias=0.45)")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(args.plot, dpi=130)
    print("saved plot ->", args.plot)

    # ---- SUMMARY ----
    srun = wandb.init(project=args.project, entity=args.entity, name="SUMMARY_itertwin",
                      config=base_cfg, reinit=True)
    tbl = wandb.Table(columns=["method", "R_or_M", "ordering_acc", "purity", "recall_clean",
                               "n_keep", "best_success_rate", "gap_vs_diffinst_best_pts", "gap_vs_current_pts"])
    for r in rows:
        gdi = (100 * (r["best"] - di_best)) if (r["best"] is not None and di_best is not None) else None
        gcur = (100 * (r["best"] - cur_best)) if (r["best"] is not None and cur_best is not None) else None
        tbl.add_data(r["method"], r.get("x"), r.get("ordering_acc"), r.get("purity"),
                     r.get("recall_clean"), r.get("n_keep"), r["best"], gdi, gcur)
        if r["kind"] == "iter" and r["best"] is not None:
            wandb.log({"tuning/R": r["x"], "tuning/best_success_rate": r["best"],
                       "tuning/ordering_acc": r.get("ordering_acc"),
                       "tuning/gap_vs_diffinst_best": gdi})
    wandb.log({"itertwin_tuning_table": tbl})
    srun.summary["baseline_best"] = base_best
    srun.summary["current_best"] = cur_best
    srun.summary["diffinst_best"] = di_best
    srun.summary["oracle_best"] = orc
    best_iter = max([r for r in itr], key=lambda r: r["best"], default=None)
    if best_iter is not None:
        srun.summary["best_R"] = best_iter["x"]
        srun.summary["best_R_success"] = best_iter["best"]
        if di_best is not None:
            srun.summary["best_gap_vs_diffinst_pts"] = 100 * (best_iter["best"] - di_best)
        if cur_best is not None:
            srun.summary["best_gap_vs_current_pts"] = 100 * (best_iter["best"] - cur_best)
    if os.path.exists(args.plot):
        wandb.log({"itertwin_tuning_curve": wandb.Image(args.plot)})
    srun.finish()
    print("\nLogged %d runs + SUMMARY_itertwin to %s" % (logged, args.project))


if __name__ == "__main__":
    main()
