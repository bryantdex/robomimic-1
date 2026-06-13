"""
Construct a controlled noisy-demonstration pool for the verification experiment.

We take N good ('better'-operator) Lift demos and CORRUPT a fraction of them by
adding Gaussian noise to their action labels (clipped to [-1,1]). Corrupted demos
are the analog of "rollouts whose answer a_i does NOT satisfy the constraints":
they look like trajectories but their action labels are wrong, so a policy that
imitates them behaves badly. A model re-prompted at each state will disagree with
these wrong actions -> low verification score -> filtered out.

Writes a new hdf5 with filter keys:
  - pool_all : all demos (clean + corrupted)              -> baseline trains on this
  - clean    : ground-truth uncorrupted demos (for reporting only)
  - corrupted: ground-truth corrupted demos (for reporting only)
"""
import argparse, h5py, numpy as np, json, os
import robomimic.utils.file_utils as FileUtils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n_demos", type=int, default=40)
    ap.add_argument("--corrupt_frac", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.5, help="std of action noise on corrupted demos")
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
    # copy data-level attrs (env_args, etc.)
    for k, v in src["data"].attrs.items():
        dgrp.attrs[k] = v

    total = 0
    clean_ids, corrupt_ids = [], []
    for i, demo in enumerate(pick):
        new_name = "demo_%d" % i
        src.copy("data/%s" % demo, dgrp, name=new_name)           # full group + attrs
        ng = dgrp[new_name]
        if demo in corrupt_set:
            acts = ng["actions"][:]
            noise = rng.randn(*acts.shape).astype(acts.dtype) * args.sigma
            acts = np.clip(acts + noise, -1.0, 1.0)
            del ng["actions"]; ng["actions"] = acts
            corrupt_ids.append(new_name)
        else:
            clean_ids.append(new_name)
        total += int(ng.attrs["num_samples"])
    dgrp.attrs["total"] = total
    src.close()

    # filter keys
    def mk(name, ids):
        dst["mask/%s" % name] = np.array(sorted(ids, key=lambda x: int(x.split("_")[1])), dtype="S")
    mk("pool_all", clean_ids + corrupt_ids)
    mk("clean", clean_ids)
    mk("corrupted", corrupt_ids)
    dst.close()

    print("wrote %s" % args.dst)
    print("  %d demos: %d clean + %d corrupted (sigma=%.2f)" % (args.n_demos, len(clean_ids), len(corrupt_ids), args.sigma))
    print("  clean    :", clean_ids)
    print("  corrupted:", corrupt_ids)
    with open(args.dst + ".meta.json", "w") as f:
        json.dump({"clean": clean_ids, "corrupted": corrupt_ids,
                   "sigma": args.sigma, "corrupt_frac": args.corrupt_frac}, f, indent=2)


if __name__ == "__main__":
    main()
