"""
Write per-instantiation training-subset filter keys ("boot_s<seed>") into a pool hdf5.

Each independent instantiation of the verifier is trained on its OWN random
subsample (fraction `frac`, without replacement) of pool_all. This is what makes
the instantiations genuinely DIFFERENT (decorrelated) rather than near-identical
same-data/different-seed copies: each one sees a different ~half of the demos.

Consequence for a systematically-corrupted (biased) demo: the instantiations that
trained on it memorize its offset and call it VALID (high v̄), but the ~half that
did NOT see it produce their own (correct) actions and flag it (low v̄). Averaging
v̄ over M instantiations therefore drives corrupted demos DOWN while clean demos
stay up -- the ensemble separates where any single instantiation cannot.
"""
import argparse, h5py, numpy as np
import robomimic.utils.file_utils as FileUtils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pool_filter_key", default="pool_all")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--frac", type=float, default=0.55)
    args = ap.parse_args()

    f = h5py.File(args.dataset, "r")
    demos = [d.decode() for d in f["mask/%s" % args.pool_filter_key][:]]
    f.close()
    demos = sorted(demos, key=lambda x: int(x.split("_")[1]))
    n = max(1, int(round(args.frac * len(demos))))

    for s in args.seeds:
        rng = np.random.RandomState(1000 + s)
        sub = sorted(rng.choice(demos, n, replace=False).tolist(), key=lambda x: int(x.split("_")[1]))
        key = "boot_s%d" % s
        FileUtils.create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=sub, key_name=key)
        print("  %s : %d/%d demos" % (key, len(sub), len(demos)))


if __name__ == "__main__":
    main()
