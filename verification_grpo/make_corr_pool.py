"""
CORRELATED-bias adversarial pool -- the regime that separates the iterative
fresh-twin method from the single-round different-instantiation ensemble.

In make_adv_pool.py each corrupted demo gets its OWN independent random bias
direction. That is exactly the easy case for a single-round OOB ensemble: a twin
that did not train on demo i produces unbiased actions for it (the biases of the
OTHER corrupt demos it saw point in unrelated directions and cancel), so it flags
demo i. Averaging M such twins separates clean from corrupt at the right tau --
there is no headroom for iteration (verified: single-round M=8 tau=0.15 already
reaches purity 1.0).

Here the corruption is CORRELATED: every corrupted demo shares a common bias
direction u0, mixed with a small per-demo component:

    b_i = bias_norm * normalize( rho * u0 + (1-rho) * u_i ),   u_i ~ random unit

With rho large (e.g. 0.8) the bias is dominated by the shared u0. A twin trained
on the (heavily corrupted) full pool sees MANY demos shifted by u0 and LEARNS u0
as normal, so even out-of-bag it reproduces the u0-shifted action -> corrupt v̄
stays HIGH -> a single round cannot flag corrupted demos no matter how many twins
M are averaged (every twin is contaminated by the same shared bias).

The iterative fresh-twin method breaks this: the small per-demo component (1-rho)
gives round 1 a foothold to drop the few most-flagged corrupted demos; the next
round's freshly re-seeded twins then train on a pool with a LOWER corrupted
fraction, so the shared u0 is reinforced less, corrupt v̄ drops, more are dropped
-- a cascade that single-round averaging cannot reproduce.

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
    ap.add_argument("--bias_norm", type=float, default=0.45)
    ap.add_argument("--rho", type=float, default=0.8, help="shared-direction weight (1=fully shared, 0=independent)")
    ap.add_argument("--jitter", type=float, default=0.03, help="small extra per-step gaussian std")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    src = h5py.File(args.src, "r")
    better = sorted([d.decode() for d in src["mask/better"][:]], key=lambda x: int(x.split("_")[1]))
    pick = sorted(rng.choice(better, args.n_demos, replace=False).tolist(), key=lambda x: int(x.split("_")[1]))
    n_corrupt = int(round(args.corrupt_frac * args.n_demos))
    corrupt_set = set(rng.choice(pick, n_corrupt, replace=False).tolist())

    # one peek to get the action dim, then a SHARED bias direction u0
    ad = src["data/%s/actions" % pick[0]].shape[1]
    u0 = rng.randn(ad); u0 = u0 / (np.linalg.norm(u0) + 1e-9)

    if os.path.exists(args.dst):
        os.remove(args.dst)
    dst = h5py.File(args.dst, "w")
    dgrp = dst.create_group("data")
    for k, v in src["data"].attrs.items():
        dgrp.attrs[k] = v

    total = 0
    clean_ids, corrupt_ids = [], []
    cos_to_shared = []
    for i, demo in enumerate(pick):
        new_name = "demo_%d" % i
        src.copy("data/%s" % demo, dgrp, name=new_name)
        ng = dgrp[new_name]
        if demo in corrupt_set:
            acts = ng["actions"][:].astype(np.float64)
            u_i = rng.randn(ad); u_i = u_i / (np.linalg.norm(u_i) + 1e-9)
            mix = args.rho * u0 + (1.0 - args.rho) * u_i
            dirn = mix / (np.linalg.norm(mix) + 1e-9)
            cos_to_shared.append(float(np.dot(dirn, u0)))
            b = args.bias_norm * dirn
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
    print("  %d demos: %d clean + %d corrupted | bias_norm=%.2f rho=%.2f (mean cos to shared=%.3f) jitter=%.2f"
          % (args.n_demos, len(clean_ids), len(corrupt_ids), args.bias_norm, args.rho,
             float(np.mean(cos_to_shared)) if cos_to_shared else 0.0, args.jitter))
    with open(args.dst + ".meta.json", "w") as f:
        json.dump({"clean": clean_ids, "corrupted": corrupt_ids, "bias_norm": args.bias_norm,
                   "rho": args.rho, "jitter": args.jitter, "corrupt_frac": args.corrupt_frac,
                   "mean_cos_to_shared": float(np.mean(cos_to_shared)) if cos_to_shared else 0.0}, f, indent=2)


if __name__ == "__main__":
    main()
