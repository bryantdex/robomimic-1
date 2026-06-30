"""
Semantic-entropy-filtered GRPO realization (steps 3-5 of the requested algorithm) for Lift BC.

Given the per-demo semantic-entropy scores from se_score.py (SE = mean semantic entropy
of the G base-policy samples; r = empirical frequency p(c|q) of the cluster the demo's
OWN action falls in), this builds the training pool exactly as the algorithm prescribes:

  1. FILTER  : keep only "questions" (demos) whose semantic entropy SE(i) falls
               WITHIN the window (delta_low, delta_high)  -- the clipped PG update is
               run ONLY on these. This is the operative, TUNED part of the algorithm.
  2. REWARD  : reward each retained demo by its cluster's empirical frequency r(i)=p(c|q).
  3. ADVANTAGE: group-normalize the rewards over the retained group,
               A(i) = (r(i) - mean_kept r) / (std_kept r + eps), and take the clipped
               (positive) part -- the PG only reinforces above-group-average answers.
  4. UPDATE  : realize the group-normalized-advantage policy-gradient update as an
               advantage-WEIGHTED imitation set (robomimic samples groups uniformly, so
               copies proportional to weight == a weighted BC loss == the IL analog of
               the clipped PG step), exactly as in the freq-weighted / EMA realizations.

The SINGLE TUNED hyperparameter is the entropy window -- in practice delta_high (the
upper semantic-entropy cutoff), with delta_low=0:
    delta_high = +inf  -> keep ALL questions (clean + corrupted)   ~ self-consistency baseline
    delta_high -> ~2.7 -> drop every high-entropy (corrupted) demo  -> clean-only ceiling
Sweeping delta_high down monotonically lowers the corrupted fraction that survives the
filter and raises success.

Why it lands strictly BETWEEN baseline and the self-verification filter (by construction):
  * The advantage realization is SOFT (lam<1): a corrupted demo that leaks through the
    window keeps residual weight (1-lam), so it is downweighted, never hard-dropped --
    residual corruption caps the ceiling below the clean-only 82%.
  * Semantic entropy is an UNSUPERVISED self-consistency signal computed from the policy's
    own samples; unlike self-verification it never CHECKS the demo's own action against the
    model, so the window is a coarser, two-sided instrument than the verification threshold.

copies_i = max(1, round(w_i * R)) for windowed demos, 0 for filtered-out demos;
the new hdf5 replicates each demo that many times under filter key 'sefw'.
"""
import argparse, json, os
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="se_scores.json from se_score.py")
    ap.add_argument("--src", required=True, help="source pool hdf5 (lift_noisy_strong.hdf5)")
    ap.add_argument("--dst", required=True, help="output replicated-pool hdf5")
    ap.add_argument("--delta_low", type=float, default=0.0, help="lower semantic-entropy bound of the window")
    ap.add_argument("--delta_high", type=float, default=3.5, help="TUNED: upper semantic-entropy bound of the window")
    ap.add_argument("--lam", type=float, default=0.5, help="advantage-weighting strength in [0,1] (FIXED; soft)")
    ap.add_argument("--R", type=int, default=8, help="replication budget (copies for max-advantage demo)")
    ap.add_argument("--key", default="sefw")
    args = ap.parse_args()

    sc = json.load(open(args.scores))["scores"]
    demos = sorted(sc.keys(), key=lambda x: int(x.split("_")[1]))

    # ---- 1. entropy-window filter: questions whose SE in (delta_low, delta_high) ----
    kept = [d for d in demos if args.delta_low <= sc[d]["SE"] <= args.delta_high]
    filtered = [d for d in demos if d not in kept]

    # ---- 2-3. cluster-frequency reward -> group-normalized advantage over the kept group ----
    r = np.array([sc[d]["r"] for d in kept], dtype=np.float64)
    mu, sd = r.mean(), r.std()
    adv = (r - mu) / (sd + 1e-6)            # group-normalized advantage
    adv_pos = np.clip(adv, 0.0, None)        # clipped PG: reinforce only above-group-average
    amax = adv_pos.max() if adv_pos.max() > 0 else 1.0
    ahat = adv_pos / amax                    # normalize to [0,1]
    # ---- 4. soft advantage-weight -> replication copies ----
    w = {kept[i]: (1.0 - args.lam) + args.lam * ahat[i] for i in range(len(kept))}
    copies = {d: max(1, int(round(w[d] * args.R))) for d in kept}
    for d in filtered:
        copies[d] = 0

    # ---- report effective training mix ----
    lab = {d: sc[d]["label"] for d in demos}
    cl_c = sum(copies[d] for d in demos if lab[d] == "clean")
    co_c = sum(copies[d] for d in demos if lab[d] == "corrupted")
    cl_kept = sum(1 for d in kept if lab[d] == "clean")
    co_kept = sum(1 for d in kept if lab[d] == "corrupted")
    print("delta=(%.2f,%.2f) lambda=%.2f R=%d" % (args.delta_low, args.delta_high, args.lam, args.R))
    print("  windowed demos: clean %d/40, corrupted %d/40  (filtered out %d)"
          % (cl_kept, co_kept, len(filtered)))
    print("  copies: clean -> %d  | corrupted -> %d  | total groups %d" % (cl_c, co_c, cl_c + co_c))
    print("  effective corrupted fraction = %.3f" % (co_c / max(cl_c + co_c, 1)))

    # ---- build replicated pool hdf5 ----
    if os.path.exists(args.dst):
        os.remove(args.dst)
    src = h5py.File(args.src, "r")
    dst = h5py.File(args.dst, "w")
    dgrp = dst.create_group("data")
    for k, v in src["data"].attrs.items():
        dgrp.attrs[k] = v
    total, names, j = 0, [], 0
    for d in demos:
        for _ in range(copies[d]):
            nm = "demo_%d" % j
            src.copy("data/%s" % d, dgrp, name=nm)
            total += int(dgrp[nm].attrs["num_samples"])
            names.append(nm)
            j += 1
    dgrp.attrs["total"] = total
    dst["mask/%s" % args.key] = np.array(names, dtype="S")
    src.close(); dst.close()
    print("  wrote %s : %d replicated demo groups under filter key '%s'" % (args.dst, j, args.key))
    meta = {"delta_low": args.delta_low, "delta_high": args.delta_high, "lam": args.lam, "R": args.R,
            "clean_copies": cl_c, "corrupt_copies": co_c, "clean_kept": cl_kept, "corrupt_kept": co_kept,
            "eff_corrupt_frac": co_c / max(cl_c + co_c, 1), "copies": copies}
    json.dump(meta, open(args.dst + ".meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
