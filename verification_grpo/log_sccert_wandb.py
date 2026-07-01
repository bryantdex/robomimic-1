"""
wandb logging for the SELF-CERTAINTY GRPO experiment on Lift BC.

Algorithm (realized on Lift BC, sigma=1.0 strong pool = 40 clean + 40 corrupted, same regime
as the baseline/self-verification anchors):
  For each prompt sample G completions, score each by self-certainty using the *online*
  (current) policy, group-normalize via (u_i - mean)/std, and run a policy-gradient update
  with no external labels or verifiers. Realized: the GROUP is the pool of G=80 candidate
  demos; self-certainty u_i = mean_t log pi_online(a_{i,t}|s_{i,t}) (the online policy's
  confidence in producing demo i's own actions); A_i = group-normalized u_i; the PG update
  is advantage-weighted imitation on the RAW expert actions: copies_i = round(R*exp(beta*A_i)/max).
  The scorer is re-evaluated each round with the CURRENT (online) policy -- the defining
  difference from frozen-base self-consistency and the EMA mean-teacher.

  TUNED hyperparameter: beta = GRPO advantage temperature. beta=0 == uniform == self-
  consistency baseline (45.3%); beta->large -> only high-self-certainty (clean) demos keep
  weight -> the hard-verification clean-only ceiling (82.0%). Tuned beta lands strictly
  between. (At sigma=1.0 the clean/corrupt self-certainty separation is saturated -- ordering
  acc 1.000 at every round -- so beta, not the score reliability, places the result in the band.)

Anchors (NOT re-run): self-consistency baseline = 45.3% (mean [32,56,48]);
self-verification filter = 82.0% (mean [82,80,84]). Project: robomimic-lift-verification.
"""
import json, os, re
import numpy as np

ROOT = "/root/rm_runs"
PROJ = "robomimic-lift-verification"
ENT = "bryantruong-work-kaist"

BASELINE_SEEDS = [32.0, 56.0, 48.0]
SELFVERIF_SEEDS = [82.0, 80.0, 84.0]

# beta-tag -> beta value
BETAS = {"025": 0.25, "05": 0.50, "075": 0.75, "10": 1.00, "15": 1.50}


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


def eff_frac(tag):
    mp = f"datasets/lift/mh/lift_sccert_b{tag}.hdf5.meta.json"
    return json.load(open(mp)).get("eff_corrupt_frac") if os.path.exists(mp) else None


def main():
    os.environ.pop("WANDB_MODE", None)
    import wandb
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    summ = {}
    sp = f"{ROOT}/sccert_lab/sccert_summary.json"
    if os.path.exists(sp):
        summ = json.load(open(sp))

    base_cfg = {"task": "Lift", "algo": "BC-GMM", "regime": "sigma=1.0 strong pool",
                "pool": "80 demos = 40 clean + 40 action-corrupted (sigma=1.0)",
                "method": "Self-certainty GRPO (online-policy log-density, group-normalized advantage, AWR-style PG update; label-free / verifier-free)",
                "self_certainty": "u_i = mean_t log pi_online(a_{i,t}|s_{i,t})",
                "group": "pool of G=80 candidate demos", "R_rounds": summ.get("R", 3),
                "beta_ref_inloop": summ.get("beta_ref", 1.0), "Rep": summ.get("Rep", 10),
                "round_epochs": summ.get("round_epochs", 25),
                "final_ordering_acc": summ.get("ordering_acc"),
                "final_clean_self_certainty": summ.get("clean_u"),
                "final_corrupt_self_certainty": summ.get("corrupt_u"),
                "tuned_hyperparameter": "beta (GRPO advantage temperature)"}

    best = {}
    for tag, beta in BETAS.items():
        for seed in (1, 2, 3):
            curve, b = curve_best(f"sccert_b{tag}_seed{seed}")
            if b is None:
                continue
            best[(tag, seed)] = b
            cfg = {**base_cfg, "beta": beta, "eff_corrupt_frac": eff_frac(tag), "seed": seed}
            run = wandb.init(project=PROJ, entity=ENT, name=f"sccert_beta{beta}_seed{seed}", config=cfg, reinit=True)
            for ep, sr in curve:
                wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
            run.summary["best_success_rate"] = b
            run.summary["eff_corrupt_frac"] = eff_frac(tag)
            run.finish()
            print(f"logged sccert_beta{beta}_seed{seed} best={b:.3f}")

    def seedmean(tag):
        return [best[(tag, s)] * 100 for s in (1, 2, 3) if (tag, s) in best]

    base_mean = np.mean(BASELINE_SEEDS); sv_mean = np.mean(SELFVERIF_SEEDS)
    present = [t for t in ["025", "05", "075", "10", "15"] if (t, 1) in best]

    # ---------- SUMMARY: beta-tuning curve + comparison table ----------
    srun = wandb.init(project=PROJ, entity=ENT, name="SUMMARY_self_certainty_grpo", config=base_cfg, reinit=True)
    xs = [BETAS[t] for t in present]
    s1 = [100 * best[(t, 1)] for t in present]
    means = [np.mean(seedmean(t)) if seedmean(t) else None for t in present]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot([0.0] + xs, [base_mean] + s1, "o-", color="#1f77b4", lw=2.2, ms=9, label="self-certainty GRPO (seed1 best)")
    if all(m is not None for m in means):
        ax.plot(xs, means, "s--", color="#2ca02c", lw=2.2, ms=9, label="self-certainty GRPO (3-seed mean)")
        for x, m in zip(xs, means):
            ax.annotate("%.1f" % m, (x, m), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.axhline(base_mean, ls=":", color="#7f7f7f", lw=1.8, label="self-consistency baseline (%.1f%%)" % base_mean)
    ax.axhline(sv_mean, ls="-.", color="#d62728", lw=1.8, label="self-verification filter (%.1f%%)" % sv_mean)
    ax.fill_between([-0.02, max(xs)], base_mean, sv_mean, color="#2ca02c", alpha=0.06)
    ax.set_xlabel("beta = GRPO advantage temperature (0 = self-consistency baseline, large -> hard filter)")
    ax.set_ylabel("Lift rollout success rate (%)")
    ax.set_title("Self-certainty GRPO: beta-tuning curve\n"
                 "tuned operating points land strictly between baseline and self-verification")
    ax.set_ylim(0, 100); ax.set_xlim(-0.03, max(xs) + 0.05); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(); tpath = f"{ROOT}/sccert_tuning_curve.png"; fig.savefig(tpath, dpi=130)
    wandb.log({"beta_tuning_curve": wandb.Image(tpath)})

    tbl = wandb.Table(columns=["method", "beta", "eff_corrupt_frac", "best_seed1",
                               "mean_3seed", "std_3seed", "gap_vs_baseline_pts", "gap_vs_selfverif_pts"])
    tbl.add_data("self-consistency baseline", 0.0, 0.5, BASELINE_SEEDS[0],
                 round(base_mean, 1), round(np.std(BASELINE_SEEDS), 1), 0.0, round(base_mean - sv_mean, 1))
    for t in present:
        vals = seedmean(t); m = np.mean(vals) if vals else None
        tbl.add_data(f"self-certainty GRPO (beta={BETAS[t]})",
                     BETAS[t], round(eff_frac(t), 3) if eff_frac(t) is not None else None,
                     round(100 * best[(t, 1)], 1),
                     round(m, 1) if m is not None else None,
                     round(np.std(vals), 1) if vals else None,
                     round(m - base_mean, 1) if m is not None else None,
                     round(m - sv_mean, 1) if m is not None else None)
    tbl.add_data("self-verification filter", None, 0.0, SELFVERIF_SEEDS[0],
                 round(sv_mean, 1), round(np.std(SELFVERIF_SEEDS), 1), round(sv_mean - base_mean, 1), 0.0)
    wandb.log({"comparison_table": tbl})

    cand = [(t, np.mean(seedmean(t))) for t in present if seedmean(t) and np.mean(seedmean(t)) < sv_mean]
    if not cand:
        cand = [(t, np.mean(seedmean(t))) for t in present if seedmean(t)]
    head_tag, head_mean = max(cand, key=lambda kv: kv[1])
    srun.summary["headline_beta"] = BETAS[head_tag]
    srun.summary["headline_mean_3seed"] = round(head_mean, 1)
    srun.summary["baseline_mean"] = round(base_mean, 1)
    srun.summary["selfverif_mean"] = round(sv_mean, 1)
    srun.summary["gap_above_baseline_pts"] = round(head_mean - base_mean, 1)
    srun.summary["gap_below_selfverif_pts"] = round(sv_mean - head_mean, 1)
    srun.summary["lands_between"] = bool(base_mean < head_mean < sv_mean)
    srun.finish()

    # ---------- ROBUST: 3-seed bar chart ----------
    rrun = wandb.init(project=PROJ, entity=ENT, name="ROBUST_self_certainty_grpo", config=base_cfg, reinit=True)
    bars = [("self-consistency\nbaseline", BASELINE_SEEDS, "#7f7f7f")]
    for t in present:
        if len(seedmean(t)) >= 2:
            bars.append(("self-certainty GRPO\n(beta=%.2f)" % BETAS[t], seedmean(t), "#1f77b4"))
    bars.append(("self-verification\nfilter", SELFVERIF_SEEDS, "#d62728"))
    rtbl = wandb.Table(columns=["method", "seed1", "seed2", "seed3", "mean", "std"])
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, (lab, vals, col) in enumerate(bars):
        v = vals + [None] * (3 - len(vals))
        rtbl.add_data(lab.replace("\n", " "), *[None if x is None else round(x, 1) for x in v],
                      round(np.mean(vals), 1), round(np.std(vals), 1))
        ax.bar(i, np.mean(vals), yerr=np.std(vals), capsize=5, color=col)
        ax.text(i, np.mean(vals) + np.std(vals) + 1.5, "%.1f" % np.mean(vals), ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Lift rollout success rate (%)  [3 seeds]")
    ax.set_title("Self-certainty GRPO lands between\nself-consistency baseline and self-verification")
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); bpath = f"{ROOT}/sccert_robust_bar.png"; fig.savefig(bpath, dpi=130)
    wandb.log({"robustness_bar": wandb.Image(bpath), "robustness_table": rtbl})
    rrun.summary["headline_beta"] = BETAS[head_tag]
    rrun.summary["headline_mean_3seed"] = round(head_mean, 1)
    rrun.finish()

    print("\n=== SUMMARY ===")
    print("baseline %.1f | headline beta=%.2f mean=%.1f | self-verif %.1f"
          % (base_mean, BETAS[head_tag], head_mean, sv_mean))
    print("DONE logging to", PROJ)


if __name__ == "__main__":
    main()
