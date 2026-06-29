"""
Frequency-weighted GRPO realization (steps 3-4 of the requested algorithm) for Lift BC.

Given the self-consistency scores from sc_score.py (votes -> M(i) top majority
fraction, p(i) soft vote share of the demo's own answer), this builds the
per-prompt loss weight exactly as the algorithm prescribes and realizes the
"frequency-weighted sum of GRPO losses" as a weighted (replicated) BC training set.

Per demo (= prompt x):
  gate_i = 1[ M(i) >= kappa ]                      # "if top answer count M(x) >= kappa"
  adv_i  = p(i)              if gate_i              # frequency-weighted reward over distinct
         = (0 - delta)       otherwise             #   answers-as-pseudo-labels; else zero
                                                    #   rewards & subtract offset delta
  u_i    = g(Mref(i))        = M(i)                 # scale whole prompt by reference (frozen
                                                    #   base) majority fraction; g = identity
  s_i    = u_i * max(adv_i, 0)                      # imitation pull (negative adv -> no pull)

The single TUNED hyperparameter is lambda (weighting strength / inverse-temperature of
the frequency weighting), entering through the shape of g's normalization:
  w_i = (1 - lambda) + lambda * (s_i / max_j s_j)   # lambda in [0,1]
    lambda = 0  -> uniform weights              == self-consistency baseline (no reweighting)
    lambda = 1  -> full frequency weighting      -> concentrates on high-consensus demos
A soft (replicated) weighting -- not a hard keep/drop filter -- so corrupted demos always
retain residual weight (1-lambda + lambda*small): by construction this cannot reach the
hard self-verification filter's clean-only ceiling.

copies_i = round(w_i * R); the new hdf5 replicates demo i that many times (distinct group
names) and a filter key 'scfw' lists them all -> robomimic samples them uniformly ->
realizes the weighted loss.
"""
import argparse, json, os
import numpy as np
import h5py
import robomimic.utils.file_utils as FileUtils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="sc_scores.json from sc_score.py")
    ap.add_argument("--src", required=True, help="source pool hdf5 (lift_noisy_strong.hdf5)")
    ap.add_argument("--dst", required=True, help="output replicated-pool hdf5")
    ap.add_argument("--kappa", type=float, default=0.10, help="gate on top majority fraction M")
    ap.add_argument("--delta", type=float, default=0.05, help="advantage offset for gated-out prompts")
    ap.add_argument("--lam", type=float, default=0.8, help="TUNED: frequency-weighting strength in [0,1]")
    ap.add_argument("--R", type=int, default=8, help="replication budget (copies for max-weight demo)")
    ap.add_argument("--key", default="scfw")
    args = ap.parse_args()

    sc = json.load(open(args.scores))["scores"]
    demos = sorted(sc.keys(), key=lambda x: int(x.split("_")[1]))

    # ---- per-demo weight exactly per the algorithm ----
    s = {}
    for d in demos:
        M, p = sc[d]["M"], sc[d]["p"]
        gate = 1.0 if M >= args.kappa else 0.0
        adv = p if gate > 0 else (0.0 - args.delta)
        u = M  # g(reference majority fraction) = identity
        s[d] = u * max(adv, 0.0)
    smax = max(s.values()) if max(s.values()) > 0 else 1.0
    w = {d: (1.0 - args.lam) + args.lam * (s[d] / smax) for d in demos}
    copies = {d: max(0, int(round(w[d] * args.R))) for d in demos}

    # ---- report effective training mix ----
    lab = {d: sc[d]["label"] for d in demos}
    cl_c = sum(copies[d] for d in demos if lab[d] == "clean")
    co_c = sum(copies[d] for d in demos if lab[d] == "corrupted")
    print("kappa=%.3f delta=%.3f lambda=%.2f R=%d" % (args.kappa, args.delta, args.lam, args.R))
    print("  copies: clean demos -> %d  | corrupted demos -> %d  | total groups %d"
          % (cl_c, co_c, cl_c + co_c))
    print("  effective corrupted fraction = %.3f" % (co_c / max(cl_c + co_c, 1)))
    cl_w = [w[d] for d in demos if lab[d] == "clean"]
    co_w = [w[d] for d in demos if lab[d] == "corrupted"]
    print("  mean w: clean=%.3f corrupted=%.3f" % (np.mean(cl_w), np.mean(co_w)))

    # ---- build replicated pool hdf5 ----
    if os.path.exists(args.dst):
        os.remove(args.dst)
    src = h5py.File(args.src, "r")
    dst = h5py.File(args.dst, "w")
    dgrp = dst.create_group("data")
    for k, v in src["data"].attrs.items():
        dgrp.attrs[k] = v
    total, names = 0, []
    j = 0
    for d in demos:
        for c in range(copies[d]):
            nm = "demo_%d" % j
            src.copy("data/%s" % d, dgrp, name=nm)
            total += int(dgrp[nm].attrs["num_samples"])
            names.append(nm)
            j += 1
    dgrp.attrs["total"] = total
    dst["mask/%s" % args.key] = np.array(names, dtype="S")
    # carry through clean/corrupted ground-truth (mapped to new replicated names) for reporting
    src.close(); dst.close()
    print("  wrote %s : %d replicated demo groups under filter key '%s'" % (args.dst, j, args.key))
    meta = {"kappa": args.kappa, "delta": args.delta, "lam": args.lam, "R": args.R,
            "copies": copies, "labels": lab, "clean_copies": cl_c, "corrupt_copies": co_c,
            "eff_corrupt_frac": co_c / max(cl_c + co_c, 1)}
    json.dump(meta, open(args.dst + ".meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
