"""
Build the beta-weighted downstream training pool from the converged online policy's
per-demo self-certainty (sccert_summary.json from sccert_run.py).

GRPO advantage = group-normalized self-certainty A_i = (u_i - mean)/std (recomputed here
from the stored u so beta is the only downstream knob). The policy-gradient update that
raises log-prob of above-mean-advantage completions is realized as advantage-weighted
imitation on the RAW expert actions:

    w_i = exp(beta * A_i),   copies_i = round(R * w_i / max_j w_j)

The pool replicates each demo that many times under filter key `sccert_soft` (robomimic
samples groups uniformly -> realizes the advantage weighting).
  beta = 0  -> uniform == self-consistency baseline (45.3%)
  beta large-> only high-self-certainty (clean) demos keep weight -> hard-verification ceiling (82.0%)
Tuned beta lands strictly between.  beta is the TUNED hyperparameter.
"""
import argparse, json, os
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="sccert_summary.json (per-demo self-certainty u + labels)")
    ap.add_argument("--src", required=True, help="source noisy pool hdf5")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--beta", type=float, required=True, help="TUNED: GRPO advantage temperature (>=0)")
    ap.add_argument("--R", type=int, default=10, help="replication budget")
    ap.add_argument("--key", default="sccert_soft")
    args = ap.parse_args()

    s = json.load(open(args.summary))
    u, lab = s["u"], s["labels"]
    demos = sorted(u.keys(), key=lambda x: int(x.split("_")[1]))

    # group-normalized advantage (recomputed from u so beta is the only downstream variable)
    vals = np.array([u[d] for d in demos], dtype=np.float64)
    mu, sd = vals.mean(), (vals.std() if vals.std() > 1e-8 else 1.0)
    adv = {d: (u[d] - mu) / sd for d in demos}
    w = {d: float(np.exp(args.beta * adv[d])) for d in demos}
    wmax = max(w.values()) if max(w.values()) > 0 else 1.0
    copies = {d: max(0, int(round(args.R * w[d] / wmax))) for d in demos}

    cc = sum(copies[d] for d in demos if lab[d] == "clean")
    kc = sum(copies[d] for d in demos if lab[d] == "corrupt")
    eff = kc / max(cc + kc, 1)
    print("beta=%.2f R=%d: copies clean=%d corrupt=%d | eff corrupt frac=%.3f" % (args.beta, args.R, cc, kc, eff))

    if os.path.exists(args.dst):
        os.remove(args.dst)
    src = h5py.File(args.src, "r"); dst = h5py.File(args.dst, "w")
    dg = dst.create_group("data")
    for k, v in src["data"].attrs.items():
        dg.attrs[k] = v
    names, tot, j = [], 0, 0
    for d in demos:
        for _ in range(copies[d]):
            nm = "demo_%d" % j
            src.copy("data/%s" % d, dg, name=nm)
            tot += int(dg[nm].attrs["num_samples"]); names.append(nm); j += 1
    dg.attrs["total"] = tot
    dst["mask/%s" % args.key] = np.array(names, dtype="S")
    src.close(); dst.close()
    print("  wrote %s : %d replicated groups under '%s'" % (args.dst, j, args.key))
    json.dump({"beta": args.beta, "R": args.R, "clean_copies": cc, "corrupt_copies": kc,
               "eff_corrupt_frac": eff, "copies": copies}, open(args.dst + ".meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
