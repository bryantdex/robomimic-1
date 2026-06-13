"""
Build a limited, noise-heavy candidate pool of demonstrations (the analog of a
batch of noisy rollouts). Most demos come from the low-quality 'worse' operator
tier, with a few 'better' demos mixed in. Standard BC on this pool is dragged
down by the noisy majority; the verification filter is meant to recover the
good behavior. Writes filter key 'pool_noisy' into the hdf5.
"""
import argparse, h5py, numpy as np, json
import robomimic.utils.file_utils as FileUtils


def demos_for(f, key):
    return [d.decode("utf-8") for d in f["mask/%s" % key][:]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n_better", type=int, default=10)
    p.add_argument("--n_worse", type=int, default=30)
    p.add_argument("--key", default="pool_noisy")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    f = h5py.File(args.dataset, "r")
    better = sorted(demos_for(f, "better"), key=lambda x: int(x.split("_")[1]))
    worse = sorted(demos_for(f, "worse"), key=lambda x: int(x.split("_")[1]))
    f.close()

    rng = np.random.RandomState(args.seed)
    pick_b = sorted(rng.choice(better, args.n_better, replace=False).tolist(), key=lambda x: int(x.split("_")[1]))
    pick_w = sorted(rng.choice(worse, args.n_worse, replace=False).tolist(), key=lambda x: int(x.split("_")[1]))
    pool = sorted(pick_b + pick_w, key=lambda x: int(x.split("_")[1]))

    FileUtils.create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=pool, key_name=args.key)
    print("pool '%s': %d demos (%d better + %d worse)" % (args.key, len(pool), len(pick_b), len(pick_w)))
    print("better:", pick_b)
    print("worse :", pick_w)


if __name__ == "__main__":
    main()
