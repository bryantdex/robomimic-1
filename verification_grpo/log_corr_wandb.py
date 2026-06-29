"""
Log the ITERATIVE FRESH-TWIN vs single-round different-instantiation experiment
on the CORRELATED-bias Lift pool to wandb.

Headline: when pseudo-label corruption is a shared/systematic bias (the realistic
self-training failure mode), the single-round different-instantiation ensemble
cannot separate it at ANY (M, tau) -- every twin trains on the full contaminated
pool and learns the shared bias, so out-of-bag scoring still reproduces it. The
iterative fresh-twin method (each round re-seed a twin, train it on the round's
progressively-cleaned kept set, filter, repeat) cascades the contamination out.
Tuned hyperparameter: R = number of rounds.

Project robomimic-lift-verification. Runs:
  itertwin_corr_baseline                no filter (40 clean + 40 corrupt)
  itertwin_corr_diffinst_M{M}_tau{t}    single-round diff-inst configs (the method to beat)
  itertwin_corr_R{r}                    iterative fresh-twin, r rounds (NEW)
  itertwin_corr_oracle                  clean-only ceiling
  SUMMARY_itertwin_corr                 comparison table + R-tuning curve
"""
import argparse, json, os, re
import numpy as np
import h5py

ROOT = "/root/rm_runs_itertwin"
DS = "datasets/lift/mh/lift_corr_r060.hdf5"


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


def kstats(key):
    f = h5py.File(DS, "r")
    clean = set(d.decode() for d in f["mask/clean"][:]); corr = set(d.decode() for d in f["mask/corrupted"][:])
    if "mask/%s" % key not in f:
        f.close(); return {}
    keep = [d.decode() for d in f["mask/%s" % key][:]]; f.close()
    nc = sum(d in clean for d in keep); nk = sum(d in corr for d in keep)
    return {"n_keep": len(keep), "n_clean": nc, "n_corrupt": nk,
            "purity": nc / max(len(keep), 1), "recall_clean": nc / max(len(clean), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="robomimic-lift-verification")
    ap.add_argument("--entity", default="bryantruong-work-kaist")
    ap.add_argument("--tag", default="", help="suffix on run dir names (e.g. _s2) and wandb names")
    ap.add_argument("--plot", default=f"{ROOT}/itertwin_corr_tuning.png")
    args = ap.parse_args()
    os.environ.pop("WANDB_MODE", None)
    T = args.tag

    isum = json.load(open(f"{ROOT}/corr060/itertwin_summary.json"))
    rounds = {r["round"]: r for r in isum["rounds"]}
    sep015 = json.load(open(f"{ROOT}/sr060_tau015/ensemble_scores.json"))["per_M"]
    sep030 = json.load(open(f"{ROOT}/sr060_tau030/ensemble_scores.json"))["per_M"]

    base_cfg = {"regime": "correlated_bias_rho0.6", "task": "Lift", "eps": isum["eps"],
                "pool": "40 clean + 40 corrupt (shared systematic bias)"}

    rows = []
    c, b = parse(f"{ROOT}/cd_baseline{T}.log")
    rows.append({"name": "itertwin_corr_baseline", "method": "no_filter", "kind": "base",
                 "x": None, "curve": c, "best": b, **kstats("pool_all")})
    # single-round diff-inst configs (the method to beat)
    for (M, tau, key, sep) in [(8, 0.30, "ens_M8_tau030", sep030), (6, 0.30, "ens_M6_tau030", sep030),
                               (6, 0.15, "ens_M6_tau015", sep015)]:
        tag = {"ens_M8_tau030": "srM8_t30", "ens_M6_tau030": "srM6_t30", "ens_M6_tau015": "srM6_t15"}[key]
        c, b = parse(f"{ROOT}/cd_{tag}{T}.log")
        s = sep.get(str(M), {})
        rows.append({"name": f"itertwin_corr_diffinst_M{M}_tau{tau}",
                     "method": "single-round diff-instantiation", "kind": "diffinst",
                     "x": M, "tau": tau, "curve": c, "best": b,
                     "ordering_acc": s.get("ordering_acc"), **kstats(key)})
    # iterative fresh-twin (NEW), R-sweep
    for r in [1, 2, 3]:
        c, b = parse(f"{ROOT}/cd_itR{r}{T}.log")
        rs = rounds.get(r, {})
        rows.append({"name": f"itertwin_corr_R{r}", "method": "iterative fresh-twin (NEW)",
                     "kind": "iter", "x": r, "tau": isum["tau"], "curve": c, "best": b,
                     "ordering_acc": rs.get("ordering_acc"), **kstats("iter_R%d_tau%03d" % (r, int(round(isum["tau"] * 100))))})
    c, b = parse(f"{ROOT}/cd_oracle{T}.log")
    rows.append({"name": "itertwin_corr_oracle", "method": "oracle (clean-only)", "kind": "oracle",
                 "x": None, "curve": c, "best": b, **kstats("clean")})

    di_best = max([r["best"] for r in rows if r["kind"] == "diffinst" and r["best"] is not None], default=None)
    base_best = next((r["best"] for r in rows if r["kind"] == "base"), None)
    orc = next((r["best"] for r in rows if r["kind"] == "oracle"), None)

    import wandb
    logged = 0
    for r in rows:
        if r["best"] is None:
            print("  skip (no log):", r["name"]); continue
        cfg = {**base_cfg, "method": r["method"], "kind": r["kind"], "R_or_M": r.get("x"),
               "tau": r.get("tau"), "purity": r.get("purity"), "recall_clean": r.get("recall_clean"),
               "n_keep": r.get("n_keep"), "ordering_acc": r.get("ordering_acc")}
        run = wandb.init(project=args.project, entity=args.entity, name=r["name"] + T, config=cfg, reinit=True)
        for ep, sr in r["curve"]:
            wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
        run.summary["best_success_rate"] = r["best"]
        run.summary["final_success_rate"] = r["curve"][-1][1]
        for k in ["ordering_acc", "purity", "recall_clean", "n_keep", "n_clean", "n_corrupt"]:
            if r.get(k) is not None:
                run.summary[k] = r[k]
        if di_best is not None:
            run.summary["gap_vs_diffinst_best_pts"] = 100 * (r["best"] - di_best)
        run.finish(); logged += 1
        print("  logged %-34s best=%.3f purity=%s recall=%s" %
              (r["name"] + T, r["best"], r.get("purity"), r.get("recall_clean")))

    # tuning curve
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    itr = sorted([r for r in rows if r["kind"] == "iter" and r["best"] is not None], key=lambda r: r["x"])
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    if itr:
        xs = [r["x"] for r in itr]; ys = [100 * r["best"] for r in itr]
        ax.plot(xs, ys, "o-", color="#2ca02c", lw=2.6, ms=10, label="iterative fresh-twin (vs R) [NEW]")
        for r in itr:
            ax.annotate("p=%.2f r=%.2f" % (r.get("purity") or 0, r.get("recall_clean") or 0),
                        (r["x"], 100 * r["best"]), textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha="center")
    if di_best is not None:
        ax.axhline(100 * di_best, ls="--", color="#1f77b4", lw=2,
                   label="best single-round diff-inst (%.0f%%)" % (100 * di_best))
    if base_best is not None:
        ax.axhline(100 * base_best, ls=":", color="#7f7f7f", lw=1.8, label="no filtering (%.0f%%)" % (100 * base_best))
    if orc is not None:
        ax.axhline(100 * orc, ls="-.", color="#9467bd", lw=1.8, label="oracle clean-only (%.0f%%)" % (100 * orc))
    ax.set_xlabel("R = number of self-training rounds (fresh re-seeded twin each round)")
    ax.set_ylabel("Lift rollout success rate (%)")
    ax.set_title("Iterative fresh-twin vs single-round diff-instantiation\n(Lift, CORRELATED systematic bias)")
    ax.set_xticks([1, 2, 3]); ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout(); fig.savefig(args.plot, dpi=130)
    print("saved plot ->", args.plot)

    srun = wandb.init(project=args.project, entity=args.entity, name="SUMMARY_itertwin_corr" + T,
                      config=base_cfg, reinit=True)
    tbl = wandb.Table(columns=["method", "R_or_M", "tau", "ordering_acc", "purity", "recall_clean",
                               "n_keep", "best_success_rate", "gap_vs_diffinst_best_pts"])
    for r in rows:
        gap = (100 * (r["best"] - di_best)) if (r["best"] is not None and di_best is not None) else None
        tbl.add_data(r["method"], r.get("x"), r.get("tau"), r.get("ordering_acc"), r.get("purity"),
                     r.get("recall_clean"), r.get("n_keep"), r["best"], gap)
        if r["kind"] == "iter" and r["best"] is not None:
            wandb.log({"tuning/R": r["x"], "tuning/best_success_rate": r["best"],
                       "tuning/purity": r.get("purity"), "tuning/gap_vs_diffinst_best": gap})
    wandb.log({"itertwin_corr_table": tbl})
    srun.summary["baseline_best"] = base_best
    srun.summary["diffinst_best"] = di_best
    srun.summary["oracle_best"] = orc
    best_iter = max([r for r in itr], key=lambda r: r["best"], default=None)
    if best_iter is not None:
        srun.summary["best_R"] = best_iter["x"]
        srun.summary["best_R_success"] = best_iter["best"]
        if di_best is not None:
            srun.summary["best_gap_vs_diffinst_pts"] = 100 * (best_iter["best"] - di_best)
    if os.path.exists(args.plot):
        wandb.log({"itertwin_corr_tuning_curve": wandb.Image(args.plot)})
    srun.finish()
    print("\nLogged %d runs + SUMMARY_itertwin_corr%s" % (logged, T))


if __name__ == "__main__":
    main()
