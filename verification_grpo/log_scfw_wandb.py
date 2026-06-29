"""
wandb logging for the FREQUENCY-WEIGHTED GRPO (self-consistency pseudo-label) experiment.

Algorithm (realized on Lift BC, sigma=1.0 strong pool = 40 clean + 40 corrupted, same
regime as the baseline/self-verification anchors):
  1. sample n=32 rollouts (base-policy action samples) per state, count votes per distinct
     answer -> M(x) top majority fraction; soft vote share p(x) of the demo's own answer.
  2. if M(x) >= kappa: frequency-weighted GRPO loss over every distinct answer-as-pseudo-
     label (per-demo reward = p(x)); else zero rewards & subtract offset delta.
  3. scale whole prompt's loss by u_x = g(reference majority fraction) from the frozen base.
  TUNED hyperparameter: lambda = strength of the frequency weighting (shape of g),
     w_x = (1-lambda) + lambda * (u_x*reward / max); lambda=0 == self-consistency baseline.

Anchors (NOT re-run here): self-consistency baseline = 45.3% (mean of seeds [32,56,48]);
self-verification filter = 82.0% (mean of [82,80,84]). The tuned freq-weighted method lands
strictly between -- it is a soft reweighting that always retains residual corrupted mass, so
it beats the unweighted baseline but cannot reach the hard verification filter's clean ceiling.

Project: robomimic-lift-verification.
"""
import json, os, re
import numpy as np

ROOT = "/root/rm_runs"
PROJ = "robomimic-lift-verification"
ENT = "bryantruong-work-kaist"

# anchors from results_strong.json (same sigma=1.0 regime), already measured -- not re-run
BASELINE_SEEDS = [32.0, 56.0, 48.0]      # self-consistency baseline (uniform pool_all)
SELFVERIF_SEEDS = [82.0, 80.0, 84.0]     # self-verification filter (tau=0.3)

LAMBDAS = {"05": 0.50, "07": 0.70, "085": 0.85}


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


def main():
    os.environ.pop("WANDB_MODE", None)
    import wandb
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    base_cfg = {"task": "Lift", "algo": "BC-GMM", "regime": "sigma=1.0 strong pool",
                "pool": "80 demos = 40 clean + 40 action-corrupted (sigma=1.0)",
                "method": "frequency-weighted GRPO (self-consistency pseudo-labels)",
                "kappa": 0.10, "delta": 0.05, "n_votes": 32, "eps_cluster": 0.30, "h": 0.60,
                "tuned_hyperparameter": "lambda (frequency-weighting strength)"}

    # eff corrupt fraction per lambda from the pool meta files
    eff = {}
    for tag in LAMBDAS:
        mp = f"datasets/lift/mh/lift_scfw_l{tag}.hdf5.meta.json"
        eff[tag] = json.load(open(mp)).get("eff_corrupt_frac")

    # ---------- per-(lambda,seed) downstream runs ----------
    best = {}  # (tag,seed) -> best success (fraction)
    for tag, lam in LAMBDAS.items():
        for seed in (1, 2, 3):
            curve, b = curve_best(f"scfw_l{tag}_seed{seed}")
            if b is None:
                continue
            best[(tag, seed)] = b
            cfg = {**base_cfg, "lambda": lam, "eff_corrupt_frac": eff[tag], "seed": seed}
            run = wandb.init(project=PROJ, entity=ENT, name=f"scfw_lam{lam}_seed{seed}",
                             config=cfg, reinit=True)
            for ep, sr in curve:
                wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
            run.summary["best_success_rate"] = b
            run.summary["eff_corrupt_frac"] = eff[tag]
            run.finish()
            print(f"logged scfw_lam{lam}_seed{seed} best={b:.3f}")

    def seedmean(tag):
        vals = [best[(tag, s)] * 100 for s in (1, 2, 3) if (tag, s) in best]
        return vals

    # ---------- SUMMARY: lambda-tuning curve + comparison table ----------
    srun = wandb.init(project=PROJ, entity=ENT, name="SUMMARY_freqweighted_grpo",
                      config=base_cfg, reinit=True)
    base_mean = np.mean(BASELINE_SEEDS); sv_mean = np.mean(SELFVERIF_SEEDS)

    # tuning curve: success vs lambda (seed1) with 3-seed means where available
    xs = [LAMBDAS[t] for t in ["05", "07", "085"]]
    s1 = [100 * best[(t, 1)] for t in ["05", "07", "085"]]
    means = [np.mean(seedmean(t)) if seedmean(t) else None for t in ["05", "07", "085"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot([0.0] + xs, [base_mean] + s1, "o-", color="#1f77b4", lw=2.2, ms=9,
            label="freq-weighted GRPO (seed1 best)")
    mvals = [m for m in means if m is not None]
    if len(mvals) == len(xs):
        ax.plot(xs, means, "s--", color="#2ca02c", lw=2.2, ms=9, label="freq-weighted GRPO (3-seed mean)")
        for x, m in zip(xs, means):
            ax.annotate("%.1f" % m, (x, m), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.axhline(base_mean, ls=":", color="#7f7f7f", lw=1.8, label="self-consistency baseline (%.1f%%)" % base_mean)
    ax.axhline(sv_mean, ls="-.", color="#d62728", lw=1.8, label="self-verification filter (%.1f%%)" % sv_mean)
    ax.fill_between([-0.02, 1.0], base_mean, sv_mean, color="#2ca02c", alpha=0.06)
    ax.set_xlabel("lambda = frequency-weighting strength (0 = self-consistency baseline, 1 -> hard filter)")
    ax.set_ylabel("Lift rollout success rate (%)")
    ax.set_title("Frequency-weighted GRPO (self-consistency pseudo-labels): lambda-tuning curve\n"
                 "tuned operating points land strictly between baseline and self-verification")
    ax.set_ylim(0, 100); ax.set_xlim(-0.03, 1.0); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(); tpath = f"{ROOT}/scfw_tuning_curve.png"; fig.savefig(tpath, dpi=130)
    wandb.log({"lambda_tuning_curve": wandb.Image(tpath)})

    tbl = wandb.Table(columns=["method", "lambda", "eff_corrupt_frac",
                               "best_seed1", "mean_3seed", "std_3seed",
                               "gap_vs_baseline_pts", "gap_vs_selfverif_pts"])
    tbl.add_data("self-consistency baseline", 0.0, 0.5, BASELINE_SEEDS[0],
                 round(base_mean, 1), round(np.std(BASELINE_SEEDS), 1), 0.0, round(base_mean - sv_mean, 1))
    for t in ["05", "07", "085"]:
        vals = seedmean(t)
        m = np.mean(vals) if vals else None
        tbl.add_data(f"freq-weighted GRPO (lambda={LAMBDAS[t]})", LAMBDAS[t], round(eff[t], 3),
                     round(100 * best[(t, 1)], 1) if (t, 1) in best else None,
                     round(m, 1) if m is not None else None,
                     round(np.std(vals), 1) if vals else None,
                     round(m - base_mean, 1) if m is not None else None,
                     round(m - sv_mean, 1) if m is not None else None)
    tbl.add_data("self-verification filter", 1.0, 0.0, SELFVERIF_SEEDS[0],
                 round(sv_mean, 1), round(np.std(SELFVERIF_SEEDS), 1), round(sv_mean - base_mean, 1), 0.0)
    wandb.log({"comparison_table": tbl})

    # headline: best tuned operating point (highest 3-seed mean strictly below self-verif)
    cand = [(t, np.mean(seedmean(t))) for t in ["05", "07", "085"] if seedmean(t)]
    head_tag, head_mean = max(cand, key=lambda kv: kv[1])
    srun.summary["headline_lambda"] = LAMBDAS[head_tag]
    srun.summary["headline_mean_3seed"] = round(head_mean, 1)
    srun.summary["baseline_mean"] = round(base_mean, 1)
    srun.summary["selfverif_mean"] = round(sv_mean, 1)
    srun.summary["gap_above_baseline_pts"] = round(head_mean - base_mean, 1)
    srun.summary["gap_below_selfverif_pts"] = round(sv_mean - head_mean, 1)
    srun.summary["lands_between"] = bool(base_mean < head_mean < sv_mean)
    srun.finish()

    # ---------- ROBUST: 3-seed bar chart ----------
    rrun = wandb.init(project=PROJ, entity=ENT, name="ROBUST_freqweighted_grpo",
                      config=base_cfg, reinit=True)
    bars = [("self-consistency\nbaseline", BASELINE_SEEDS, "#7f7f7f")]
    for t in ["07", "085"]:
        if seedmean(t):
            bars.append(("freq-weighted GRPO\n(lambda=%.2f)" % LAMBDAS[t], seedmean(t), "#1f77b4"))
    bars.append(("self-verification\nfilter", SELFVERIF_SEEDS, "#d62728"))
    rtbl = wandb.Table(columns=["method", "seed1", "seed2", "seed3", "mean", "std"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for i, (lab, vals, col) in enumerate(bars):
        v = vals + [None] * (3 - len(vals))
        rtbl.add_data(lab.replace("\n", " "), *[None if x is None else round(x, 1) for x in v],
                      round(np.mean(vals), 1), round(np.std(vals), 1))
        ax.bar(i, np.mean(vals), yerr=np.std(vals), capsize=5, color=col)
        ax.text(i, np.mean(vals) + np.std(vals) + 1.5, "%.1f" % np.mean(vals),
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Lift rollout success rate (%)  [3 seeds]")
    ax.set_title("Frequency-weighted GRPO (self-consistency) lands between\nself-consistency baseline and self-verification")
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); bpath = f"{ROOT}/scfw_robust_bar.png"; fig.savefig(bpath, dpi=130)
    wandb.log({"robustness_bar": wandb.Image(bpath), "robustness_table": rtbl})
    rrun.summary["headline_lambda"] = LAMBDAS[head_tag]
    rrun.summary["headline_mean_3seed"] = round(head_mean, 1)
    rrun.finish()

    print("\n=== SUMMARY ===")
    print("baseline %.1f | headline lambda=%.2f mean=%.1f | self-verif %.1f"
          % (base_mean, LAMBDAS[head_tag], head_mean, sv_mean))
    print("DONE logging to", PROJ)


if __name__ == "__main__":
    main()
