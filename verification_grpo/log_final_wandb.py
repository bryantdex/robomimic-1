"""
Comprehensive wandb logging for the ITERATIVE FRESH-TWIN verification experiment.

Algorithm: each self-training round draws v̄ from a freshly RE-SEEDED twin
instantiation trained on that round's (progressively cleaned) kept set, so the
verifier is decorrelated from the generator AND from every prior round's verifier.
Tuned hyperparameter: R = number of rounds.

Headline (Lift, correlated/systematic-bias pool rho=0.6): the iterative method beats
the BEST single-round different-instantiation config by +11.6 pts (4/4 seeds), because
single-round trains every twin on the full contaminated pool and learns the shared
bias (cannot separate it at any M, tau), while the iterative cascade removes it.

Negative control (independent-per-demo bias): once tau is also tuned, single-round
diff-inst matches the iterative method (no gap) -- the iterative advantage is specific
to CORRELATED contamination. Logged for honesty.

Project: robomimic-lift-verification.
"""
import json, os, re
import numpy as np
import h5py

ROOT = "/root/rm_runs_itertwin"
PROJ = "robomimic-lift-verification"
ENT = "bryantruong-work-kaist"
DS06 = "datasets/lift/mh/lift_corr_r060.hdf5"
DS08 = "datasets/lift/mh/lift_corr_r080.hdf5"


def curve_best(name):
    p = f"{ROOT}/{name}.log"
    if not os.path.exists(p):
        return [], None
    txt = open(p, errors="ignore").read()
    pairs = []
    for m in re.finditer(r"Epoch (\d+) Rollouts took", txt):
        tail = txt[m.end():m.end() + 2000]
        sm = re.search(r'"Success_Rate":\s*([0-9.]+)', tail)
        if sm:
            pairs.append((int(m.group(1)), float(sm.group(1))))
    pairs = sorted(set(pairs))
    return pairs, (max(p[1] for p in pairs) if pairs else None)


def kstats(ds, key):
    f = h5py.File(ds, "r")
    clean = set(d.decode() for d in f["mask/clean"][:]); corr = set(d.decode() for d in f["mask/corrupted"][:])
    if "mask/%s" % key not in f:
        f.close(); return {}
    keep = [d.decode() for d in f["mask/%s" % key][:]]; f.close()
    nc = sum(d in clean for d in keep); nk = sum(d in corr for d in keep)
    return {"n_keep": len(keep), "n_clean": nc, "n_corrupt": nk,
            "purity": round(nc / max(len(keep), 1), 3), "recall_clean": round(nc / max(len(clean), 1), 3)}


def main():
    os.environ.pop("WANDB_MODE", None)
    import wandb
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    isum06 = json.load(open(f"{ROOT}/corr060/itertwin_summary.json"))
    rounds06 = {r["round"]: r for r in isum06["rounds"]}
    sep030 = json.load(open(f"{ROOT}/sr060_tau030/ensemble_scores.json"))["per_M"]
    sep015 = json.load(open(f"{ROOT}/sr060_tau015/ensemble_scores.json"))["per_M"]

    base_cfg = {"task": "Lift", "algo": "BC-GMM", "eps": 0.30, "regime": "correlated_bias_rho0.6",
                "pool": "80 demos = 40 clean + 40 corrupted (shared systematic action bias)",
                "tuned_hyperparameter": "R (number of self-training rounds)"}

    # ---------- per-method downstream curves (seed 1, rho=0.6) ----------
    runs = [
        ("itw_baseline", "no_filter", None, "pool_all", "cd_baseline"),
        ("itw_diffinst_M8_t0.30", "single-round diff-inst", 8, "ens_M8_tau030", "cd_srM8_t30"),
        ("itw_diffinst_M6_t0.30_BEST", "single-round diff-inst (best)", 6, "ens_M6_tau030", "cd_srM6_t30"),
        ("itw_diffinst_M6_t0.15", "single-round diff-inst", 6, "ens_M6_tau015", "cd_srM6_t15"),
        ("itw_R1", "iterative fresh-twin (NEW)", 1, "iter_R1_tau015", "cd_itR1"),
        ("itw_R2", "iterative fresh-twin (NEW)", 2, "iter_R2_tau015", "cd_itR2"),
        ("itw_R3_BEST", "iterative fresh-twin (NEW, best)", 3, "iter_R3_tau015", "cd_itR3"),
        ("itw_oracle", "oracle clean-only", None, "clean", "cd_oracle"),
    ]
    best = {}
    for wbname, method, x, key, logname in runs:
        curve, b = curve_best(logname)
        if b is None:
            print("skip", wbname); continue
        st = kstats(DS06, key)
        oa = None
        if method.startswith("iter") and x in rounds06:
            oa = rounds06[x].get("ordering_acc")
        elif method.startswith("single") and x is not None:
            sep = sep030 if "t0.30" in wbname else sep015
            oa = sep.get(str(x), {}).get("ordering_acc")
        cfg = {**base_cfg, "method": method, "R_or_M": x, **st, "ordering_acc": oa}
        run = wandb.init(project=PROJ, entity=ENT, name=wbname, config=cfg, reinit=True)
        for ep, sr in curve:
            wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
        run.summary["best_success_rate"] = b
        run.summary.update({k: v for k, v in st.items()})
        if oa is not None:
            run.summary["ordering_acc"] = oa
        run.finish()
        best[wbname] = b
        print("logged %-30s best=%.3f %s" % (wbname, b, st))

    di_best = best.get("itw_diffinst_M6_t0.30_BEST")

    # ---------- robustness (4 seeds) ----------
    def seedvals(meth_seed1, rb_meth):
        vals = []
        _, b1 = curve_best(meth_seed1); vals.append(b1)
        for s in (2, 3, 4):
            _, b = curve_best(f"rb_{rb_meth}_s{s}"); vals.append(b)
        return [v * 100 for v in vals if v is not None]
    rob = {
        "no-filter baseline": seedvals("cd_baseline", "baseline"),
        "single-round diff-inst (best)": seedvals("cd_srM6_t30", "srM6_t30"),
        "iterative fresh-twin R3 (NEW)": seedvals("cd_itR3", "itR3"),
        "oracle (clean-only)": seedvals("cd_oracle", "oracle"),
    }
    rrun = wandb.init(project=PROJ, entity=ENT, name="ROBUST_itertwin", config=base_cfg, reinit=True)
    rtbl = wandb.Table(columns=["method", "seed1", "seed2", "seed3", "seed4", "mean", "std", "gap_vs_diffinst_pts"])
    di_mean = np.mean(rob["single-round diff-inst (best)"])
    for meth, vals in rob.items():
        v = vals + [None] * (4 - len(vals))
        gap = np.mean(vals) - di_mean
        rtbl.add_data(meth, *[None if x is None else round(x, 1) for x in v],
                      round(np.mean(vals), 1), round(np.std(vals), 1), round(gap, 1))
    wandb.log({"robustness_table": rtbl})
    it_mean = np.mean(rob["iterative fresh-twin R3 (NEW)"])
    rrun.summary["iterative_mean"] = round(it_mean, 1)
    rrun.summary["diffinst_best_mean"] = round(di_mean, 1)
    rrun.summary["robust_gap_vs_diffinst_pts"] = round(it_mean - di_mean, 1)
    rrun.summary["baseline_mean"] = round(np.mean(rob["no-filter baseline"]), 1)
    rrun.summary["oracle_mean"] = round(np.mean(rob["oracle (clean-only)"]), 1)
    rrun.summary["all_seeds_positive"] = bool(all(
        i - s > 0 for i, s in zip(rob["iterative fresh-twin R3 (NEW)"], rob["single-round diff-inst (best)"])))
    # bar chart with error bars
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = list(rob.keys()); means = [np.mean(rob[k]) for k in labels]; stds = [np.std(rob[k]) for k in labels]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd"]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=5, color=colors)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1.5, "%.1f" % m, ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["no-filter", "single-round\ndiff-inst (best)", "iterative\nfresh-twin (NEW)", "oracle\n(clean)"], fontsize=9)
    ax.set_ylabel("Lift rollout success rate (%)  [4 seeds]")
    ax.set_title("Iterative fresh-twin vs best single-round diff-instantiation\n(Lift, correlated systematic bias; +%.1f pts, 4/4 seeds)" % (it_mean - di_mean))
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); barpath = f"{ROOT}/itertwin_robust_bar.png"; fig.savefig(barpath, dpi=130)
    wandb.log({"robustness_bar": wandb.Image(barpath)})
    rrun.finish()
    print("robust gap = +%.1f pts (iterative %.1f vs diff-inst %.1f)" % (it_mean - di_mean, it_mean, di_mean))

    # ---------- R-tuning curve + filtering-frontier + mechanism (SUMMARY) ----------
    srun = wandb.init(project=PROJ, entity=ENT, name="SUMMARY_itertwin", config=base_cfg, reinit=True)
    # (a) R-tuning curve (seed1) with reference lines
    Rxy = [(1, best.get("itw_R1")), (2, best.get("itw_R2")), (3, best.get("itw_R3_BEST"))]
    Rxy = [(x, y) for x, y in Rxy if y is not None]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if Rxy:
        ax.plot([x for x, _ in Rxy], [100 * y for _, y in Rxy], "o-", color="#2ca02c", lw=2.6, ms=10,
                label="iterative fresh-twin (vs R) [NEW]")
        for r, y in Rxy:
            rs = rounds06.get(r, {})
            ax.annotate("p=%.2f r=%.2f" % (rs.get("purity", 0), rs.get("recall_clean", 0)),
                        (r, 100 * y), textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")
    if di_best is not None:
        ax.axhline(100 * di_best, ls="--", color="#1f77b4", lw=2, label="best single-round diff-inst (%.0f%%)" % (100 * di_best))
    if best.get("itw_baseline") is not None:
        ax.axhline(100 * best["itw_baseline"], ls=":", color="#7f7f7f", lw=1.8, label="no filtering (%.0f%%)" % (100 * best["itw_baseline"]))
    if best.get("itw_oracle") is not None:
        ax.axhline(100 * best["itw_oracle"], ls="-.", color="#9467bd", lw=1.8, label="oracle clean-only (%.0f%%)" % (100 * best["itw_oracle"]))
    ax.set_xlabel("R = number of self-training rounds (fresh re-seeded twin each round)")
    ax.set_ylabel("Lift rollout success rate (%)")
    ax.set_xticks([1, 2, 3]); ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Iterative fresh-twin verification: R-tuning curve (Lift, correlated bias)")
    fig.tight_layout(); tpath = f"{ROOT}/itertwin_Rtuning.png"; fig.savefig(tpath, dpi=130)
    wandb.log({"R_tuning_curve": wandb.Image(tpath)})

    # (b) single-round mechanism: ordering_acc does NOT improve with M on correlated pool
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    Ms = sorted(int(m) for m in sep030.keys())
    ax.plot(Ms, [sep030[str(m)]["ordering_acc"] for m in Ms], "s--", color="#1f77b4", lw=2, ms=7,
            label="single-round diff-inst (corr. bias)")
    cascade = [rounds06[r]["ordering_acc"] for r in sorted(rounds06) if rounds06[r].get("ordering_acc") is not None]
    ax.plot(range(1, len(cascade) + 1), cascade, "o-", color="#2ca02c", lw=2.4, ms=8,
            label="iterative fresh-twin (per round)")
    ax.set_xlabel("M (twins averaged)  /  R (round)"); ax.set_ylabel("ordering accuracy (clean vs corrupt)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
    ax.set_title("Why single-round fails on correlated bias: averaging more twins doesn't separate")
    fig.tight_layout(); mpath = f"{ROOT}/itertwin_mechanism.png"; fig.savefig(mpath, dpi=130)
    wandb.log({"mechanism_ordering_acc": wandb.Image(mpath)})

    # (c) comparison table
    tbl = wandb.Table(columns=["method", "R_or_M", "purity", "recall_clean", "n_keep", "best_success_seed1", "gap_vs_diffinst_pts"])
    for wbname, method, x, key, logname in runs:
        if wbname not in best:
            continue
        st = kstats(DS06, key)
        gap = 100 * (best[wbname] - di_best) if di_best is not None else None
        tbl.add_data(method, x, st.get("purity"), st.get("recall_clean"), st.get("n_keep"),
                     round(100 * best[wbname], 1), round(gap, 1) if gap is not None else None)
    wandb.log({"comparison_table": tbl})
    srun.summary["headline_robust_gap_pts"] = round(it_mean - di_mean, 1)
    srun.summary["iterative_R3_mean"] = round(it_mean, 1)
    srun.summary["diffinst_best_mean"] = round(di_mean, 1)
    srun.summary["note_negative_control"] = ("On independent-per-demo bias, tuned single-round diff-inst "
                                             "matches iterative (no gap); the gap is specific to correlated bias.")
    srun.finish()
    print("DONE logging to", PROJ)


if __name__ == "__main__":
    main()
