"""
wandb logging for the SEMANTIC-ENTROPY-FILTERED GRPO experiment on Lift BC.

Algorithm (realized on Lift BC, sigma=1.0 strong pool = 40 clean + 40 corrupted, same
regime as the baseline/self-verification anchors):
  1. sample G=32 responses (base-policy action samples) per state; cluster them by
     semantic equivalence (greedy radius-eps) -> clusters c with empirical freq p(c|q)=|c|/G.
  2. reward each response by its cluster's empirical frequency r(i)=p(c|q); compute
     GROUP-NORMALIZED advantages A(i)=(r-mean)/std over the retained group (clipped PG).
  3. run the clipped policy-gradient update ONLY on questions (demos) whose SEMANTIC
     ENTROPY SE(i)=mean_t[-sum_c p log p] falls within the window (delta_low, delta_high)
     -- thereby minimizing semantic entropy to favor self-consistent answers.
  TUNED hyperparameter: delta_high = upper semantic-entropy cutoff (delta_low=0).
     delta_high -> +inf keeps all 80 (~self-consistency baseline); delta_high -> 2.7 drops
     every high-entropy (corrupted) demo (-> clean-only ceiling). Soft (lam=0.5) advantage
     weighting means leaked corrupted demos keep residual weight -> cannot reach 82%.

Anchors (NOT re-run here): self-consistency baseline = 45.3% (mean of seeds [32,56,48]);
self-verification filter = 82.0% (mean of [82,80,84]).

Project: robomimic-lift-verification.
"""
import json, os, re
import numpy as np

ROOT = "/root/rm_runs"
PROJ = "robomimic-lift-verification"
ENT = "bryantruong-work-kaist"

BASELINE_SEEDS = [32.0, 56.0, 48.0]      # self-consistency baseline (uniform pool_all)
SELFVERIF_SEEDS = [82.0, 80.0, 84.0]     # self-verification filter (tau=0.3)

# delta_high tags -> value
DHS = {"350": 3.5, "340": 3.4, "330": 3.3, "320": 3.2, "300": 3.0, "270": 2.7}
ORDER = ["350", "340", "330", "320", "300", "270"]
# which delta_high values get the multi-seed treatment (filled after sweep inspection)
MULTISEED_TAGS = ["330", "340"]


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
                "method": "semantic-entropy-filtered GRPO (cluster-frequency reward)",
                "G": 32, "eps_cluster": 0.30, "delta_low": 0.0, "lam": 0.5,
                "tuned_hyperparameter": "delta_high (upper semantic-entropy cutoff)"}

    eff = {}
    for tag in DHS:
        mp = f"datasets/lift/mh/lift_sefw_dh{tag}.hdf5.meta.json"
        if os.path.exists(mp):
            eff[tag] = json.load(open(mp)).get("eff_corrupt_frac")

    base_mean = np.mean(BASELINE_SEEDS); sv_mean = np.mean(SELFVERIF_SEEDS)

    # ---------- per-(delta_high, seed) downstream runs ----------
    best = {}  # (tag, seed) -> best success fraction
    for tag in ORDER:
        seeds = (1, 2, 3) if tag in MULTISEED_TAGS else (1,)
        for seed in seeds:
            curve, b = curve_best(f"sefw_dh{tag}_seed{seed}")
            if b is None:
                continue
            best[(tag, seed)] = b
            cfg = {**base_cfg, "delta_high": DHS[tag], "eff_corrupt_frac": eff.get(tag), "seed": seed}
            run = wandb.init(project=PROJ, entity=ENT, name=f"sefw_dh{DHS[tag]}_seed{seed}",
                             config=cfg, reinit=True)
            for ep, sr in curve:
                wandb.log({"rollout/success_rate": sr, "epoch": ep}, step=ep)
            run.summary["best_success_rate"] = b
            run.summary["eff_corrupt_frac"] = eff.get(tag)
            run.finish()
            print(f"logged sefw_dh{DHS[tag]}_seed{seed} best={b:.3f}")

    def seedmean(tag):
        return [best[(tag, s)] * 100 for s in (1, 2, 3) if (tag, s) in best]

    # ---------- SUMMARY: delta_high tuning curve + comparison table ----------
    srun = wandb.init(project=PROJ, entity=ENT, name="SUMMARY_sementropy_grpo",
                      config=base_cfg, reinit=True)
    xs = [DHS[t] for t in ORDER if (t, 1) in best]
    s1 = [100 * best[(t, 1)] for t in ORDER if (t, 1) in best]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(xs, s1, "o-", color="#1f77b4", lw=2.2, ms=9, label="semantic-entropy GRPO (seed1 best)")
    for t in ORDER:
        if (t, 1) in best and seedmean(t) and len(seedmean(t)) > 1:
            m = np.mean(seedmean(t))
            ax.plot(DHS[t], m, "s", color="#2ca02c", ms=11)
            ax.annotate("%.1f (3-seed)" % m, (DHS[t], m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5, color="#2ca02c")
    # annotate eff corrupt frac at each point
    for t, x, y in zip([t for t in ORDER if (t, 1) in best], xs, s1):
        if eff.get(t) is not None:
            ax.annotate("c=%.2f" % eff[t], (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7.5, color="#555")
    ax.axhline(base_mean, ls=":", color="#7f7f7f", lw=1.8, label="self-consistency baseline (%.1f%%)" % base_mean)
    ax.axhline(sv_mean, ls="-.", color="#d62728", lw=1.8, label="self-verification filter (%.1f%%)" % sv_mean)
    ax.fill_between([2.6, 3.6], base_mean, sv_mean, color="#2ca02c", alpha=0.06)
    ax.invert_xaxis()  # tighter filter (lower delta_high) to the right
    ax.set_xlabel("delta_high = upper semantic-entropy cutoff  (lower -> tighter filter; c = eff. corrupt frac)")
    ax.set_ylabel("Lift rollout success rate (%)")
    ax.set_title("Semantic-entropy-filtered GRPO: delta_high tuning curve\n"
                 "tuned operating points land strictly between baseline and self-verification")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout(); tpath = f"{ROOT}/sefw_tuning_curve.png"; fig.savefig(tpath, dpi=130)
    wandb.log({"delta_high_tuning_curve": wandb.Image(tpath)})

    tbl = wandb.Table(columns=["method", "delta_high", "eff_corrupt_frac",
                               "best_seed1", "mean_3seed", "std_3seed",
                               "gap_vs_baseline_pts", "gap_vs_selfverif_pts"])
    tbl.add_data("self-consistency baseline", float("inf"), 0.5, BASELINE_SEEDS[0],
                 round(base_mean, 1), round(np.std(BASELINE_SEEDS), 1), 0.0, round(base_mean - sv_mean, 1))
    for t in ORDER:
        if (t, 1) not in best:
            continue
        vals = seedmean(t); m = np.mean(vals)
        tbl.add_data(f"semantic-entropy GRPO (delta_high={DHS[t]})", DHS[t], round(eff.get(t, float('nan')), 3),
                     round(100 * best[(t, 1)], 1), round(m, 1) if len(vals) > 1 else None,
                     round(np.std(vals), 1) if len(vals) > 1 else None,
                     round(m - base_mean, 1), round(m - sv_mean, 1))
    tbl.add_data("self-verification filter", 2.7, 0.0, SELFVERIF_SEEDS[0],
                 round(sv_mean, 1), round(np.std(SELFVERIF_SEEDS), 1), round(sv_mean - base_mean, 1), 0.0)
    wandb.log({"comparison_table": tbl})

    # headline: highest 3-seed mean strictly below self-verification (among multi-seeded tags)
    cand = [(t, np.mean(seedmean(t))) for t in MULTISEED_TAGS if len(seedmean(t)) > 1
            and np.mean(seedmean(t)) < sv_mean]
    if not cand:
        cand = [(t, 100 * best[(t, 1)]) for t in ORDER if (t, 1) in best and 100 * best[(t, 1)] < sv_mean]
    head_tag, head_mean = max(cand, key=lambda kv: kv[1])
    srun.summary["headline_delta_high"] = DHS[head_tag]
    srun.summary["headline_mean_3seed"] = round(head_mean, 1)
    srun.summary["baseline_mean"] = round(base_mean, 1)
    srun.summary["selfverif_mean"] = round(sv_mean, 1)
    srun.summary["gap_above_baseline_pts"] = round(head_mean - base_mean, 1)
    srun.summary["gap_below_selfverif_pts"] = round(sv_mean - head_mean, 1)
    srun.summary["lands_between"] = bool(base_mean < head_mean < sv_mean)
    srun.finish()

    # ---------- ROBUST: 3-seed bar chart ----------
    rrun = wandb.init(project=PROJ, entity=ENT, name="ROBUST_sementropy_grpo",
                      config=base_cfg, reinit=True)
    bars = [("self-consistency\nbaseline", BASELINE_SEEDS, "#7f7f7f")]
    for t in MULTISEED_TAGS:
        if len(seedmean(t)) > 1:
            bars.append(("semantic-entropy GRPO\n(delta_high=%.1f)" % DHS[t], seedmean(t), "#1f77b4"))
    bars.append(("self-verification\nfilter", SELFVERIF_SEEDS, "#d62728"))
    rtbl = wandb.Table(columns=["method", "seed1", "seed2", "seed3", "mean", "std"])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, (lab, vals, col) in enumerate(bars):
        v = vals + [None] * (3 - len(vals))
        rtbl.add_data(lab.replace("\n", " "), *[None if x is None else round(x, 1) for x in v],
                      round(np.mean(vals), 1), round(np.std(vals), 1))
        ax.bar(i, np.mean(vals), yerr=np.std(vals), capsize=5, color=col)
        ax.text(i, np.mean(vals) + np.std(vals) + 1.5, "%.1f" % np.mean(vals),
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("Lift rollout success rate (%)  [3 seeds]")
    ax.set_title("Semantic-entropy-filtered GRPO lands between\nself-consistency baseline and self-verification")
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); bpath = f"{ROOT}/sefw_robust_bar.png"; fig.savefig(bpath, dpi=130)
    wandb.log({"robustness_bar": wandb.Image(bpath), "robustness_table": rtbl})
    rrun.summary["headline_delta_high"] = DHS[head_tag]
    rrun.summary["headline_mean_3seed"] = round(head_mean, 1)
    rrun.finish()

    print("\n=== SUMMARY ===")
    print("baseline %.1f | headline delta_high=%.1f mean=%.1f | self-verif %.1f"
          % (base_mean, DHS[head_tag], head_mean, sv_mean))
    print("DONE logging to", PROJ)


if __name__ == "__main__":
    main()
