"""
Build the lambda-weighted downstream training pool from the converged EMA teacher's
per-demo reward (ema_summary.json from ema_teacher_run.py).

copies_i = round(w_i * R),  w_i = (1-lambda) + lambda * (reward_i / max_j reward_j)
The pool replicates each demo that many times under filter key `ema_soft` (robomimic samples
groups uniformly -> realizes the GRPO advantage weighting). Trains on RAW expert actions.
lambda=0 == uniform == self-consistency baseline; lambda->1 -> corrupt demos drop out
(hard-verification clean-only ceiling). Tuned lambda lands strictly between.
"""
import argparse, json, os
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="ema_summary.json (has per-demo reward + labels)")
    ap.add_argument("--src", required=True, help="source noisy pool hdf5")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--lam", type=float, required=True, help="TUNED: GRPO advantage-weighting strength in [0,1]")
    ap.add_argument("--R", type=int, default=10, help="replication budget")
    ap.add_argument("--key", default="ema_soft")
    args = ap.parse_args()

    s = json.load(open(args.summary))
    reward, lab = s["reward"], s["labels"]
    demos = sorted(reward.keys(), key=lambda x: int(x.split("_")[1]))
    rmax = max(reward.values()) if max(reward.values()) > 0 else 1.0
    w = {d: (1.0 - args.lam) + args.lam * (reward[d] / rmax) for d in demos}
    copies = {d: max(0, int(round(w[d] * args.R))) for d in demos}

    cc = sum(copies[d] for d in demos if lab[d] == "clean")
    kc = sum(copies[d] for d in demos if lab[d] == "corrupt")
    eff = kc / max(cc + kc, 1)
    print("lambda=%.2f R=%d: copies clean=%d corrupt=%d | eff corrupt frac=%.3f" % (args.lam, args.R, cc, kc, eff))

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
    json.dump({"lam": args.lam, "R": args.R, "clean_copies": cc, "corrupt_copies": kc,
               "eff_corrupt_frac": eff, "copies": copies}, open(args.dst + ".meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
