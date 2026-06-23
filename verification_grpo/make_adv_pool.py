"""
Adversarial noisy pool for the different-instantiation verification experiment.

Unlike zero-mean Gaussian action noise (whose detectability is *coupled* to its
harm -- any sigma large enough to hurt downstream is trivially flagged by EVERY
single verifier instantiation, leaving no role for ensembling), this builds a
corruption that decouples the two:

  CONSTANT PER-DEMO ACTION BIAS: for each corrupted demo we draw a fixed random
  direction u and add b = bias_norm * u to *every* action (then clip, + small
  jitter). Per state the biased action is only ~bias_norm off -- right around the
  verification tolerance eps -- so whether a given policy instantiation's samples
  fall within eps is BORDERLINE and differs across instantiations (high inter-
  instantiation variance => a single instantiation mis-ranks demos). But the bias
  is SYSTEMATIC, so it compounds over the trajectory and badly derails a policy
  that imitates it => keeping a corrupted demo is very harmful downstream.

This is exactly the setting where "compute v̄ under a different instantiation of
the same model" -- averaging the VALID fraction over M independent instantiations
-- reduces the estimator variance and restores a clean filter.

Filter keys written: pool_all, clean, corrupted (+ .meta.json).
"""
import argparse, h5py, numpy as np, json, os
import robomimic.utils.file_utils as FileUtils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n_demos", type=int, default=80)
    ap.add_argument("--corrupt_frac", type=float, default=0.5)
    ap.add_argument("--bias_norm", type=float, default=0.30, help="L2 norm of the constant per-demo action bias")
    ap.add_argument("--jitter", type=float, default=0.03, help="small extra per-step gaussian std")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    src = h5py.File(args.src, "r")
    better = sorted([d.decode() for d in src["mask/better"][:]], key=lambda x: int(x.split("_")[1]))
    pick = sorted(rng.choice(better, args.n_demos, replace=False).tolist(), key=lambda x: int(x.split("_")[1]))
    n_corrupt = int(round(args.corrupt_frac * args.n_demos))
    corrupt_set = set(rng.choice(pick, n_corrupt, replace=False).tolist())

    if os.path.exists(args.dst):
        os.remove(args.dst)
    dst = h5py.File(args.dst, "w")
    dgrp = dst.create_group("data")
    for k, v in src["data"].attrs.items():
        dgrp.attrs[k] = v

    total = 0
    clean_ids, corrupt_ids = [], []
    for i, demo in enumerate(pick):
        new_name = "demo_%d" % i
        src.copy("data/%s" % demo, dgrp, name=new_name)
        ng = dgrp[new_name]
        if demo in corrupt_set:
            acts = ng["actions"][:].astype(np.float64)
            ad = acts.shape[1]
            u = rng.randn(ad); u = u / (np.linalg.norm(u) + 1e-9)   # random unit direction
            b = args.bias_norm * u                                  # constant bias, this demo
            acts = acts + b[None, :] + rng.randn(*acts.shape) * args.jitter
            acts = np.clip(acts, -1.0, 1.0).astype(np.float32)
            del ng["actions"]; ng["actions"] = acts
            corrupt_ids.append(new_name)
        else:
            clean_ids.append(new_name)
        total += int(ng.attrs["num_samples"])
    dgrp.attrs["total"] = total
    src.close()

    def mk(name, ids):
        dst["mask/%s" % name] = np.array(sorted(ids, key=lambda x: int(x.split("_")[1])), dtype="S")
    mk("pool_all", clean_ids + corrupt_ids)
    mk("clean", clean_ids)
    mk("corrupted", corrupt_ids)
    dst.close()

    print("wrote %s" % args.dst)
    print("  %d demos: %d clean + %d corrupted (constant bias_norm=%.2f jitter=%.2f)"
          % (args.n_demos, len(clean_ids), len(corrupt_ids), args.bias_norm, args.jitter))
    with open(args.dst + ".meta.json", "w") as f:
        json.dump({"clean": clean_ids, "corrupted": corrupt_ids,
                   "bias_norm": args.bias_norm, "jitter": args.jitter,
                   "corrupt_frac": args.corrupt_frac}, f, indent=2)


if __name__ == "__main__":
    main()
