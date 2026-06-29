"""
Iterative self-training with a per-round FRESHLY RE-SEEDED TWIN verifier.

Requested algorithm:
  "Each self-training round, draw the verification score v̄ from a freshly
   re-seeded twin instantiation of the model rather than a fixed one, so the
   verifier that filters this round's pseudo-labels is decorrelated from both
   the generator and every prior round's verifier."

Contrast with the previous different-instantiation method (verify_ensemble.py):
that one is SINGLE-ROUND -- it averages M twins, but every twin is trained on the
SAME ~50%-contaminated pool, so each twin still memorizes a share of the bias and
the ensemble plateaus (best 73.3% on Lift bias=0.45). Here we ITERATE: round r's
twins are re-seeded AND trained on the set kept by round r-1, which is already
purer. Each successive twin therefore memorizes LESS bias -> its v̄ separates clean
from corrupted more sharply -> we purge residual corruption while RECOVERING
borderline-clean demos the single-round ensemble had to discard. The verifier each
round is decorrelated from the generator (fresh seed + subsample) and from every
prior round (fresh seed + a different, cleaner training set).

Per round we train m>=2 twins on independent `frac` subsamples of the CURRENT kept
set and score each kept demo by its OUT-OF-BAG mean VALID fraction (the twins that
did not train on it). Demos with v̄ >= tau survive to the next round.

The tuned hyperparameter is R = number of self-training rounds (parallels M in the
single-round method). A filter key `iter_R{r}_tau{tau}` is written after EACH round
so one R=Rmax run yields the whole R-sweep for downstream evaluation.
"""
import argparse, json, os, subprocess, glob
import numpy as np
import torch
import h5py
import robomimic.utils.file_utils as FileUtils
from verify_ensemble import score_one_verifier, ordering_acc, gt_sets

HERE = os.path.dirname(os.path.abspath(__file__))


def idx_of(d):
    return int(d.split("_")[1])


def train_twin(dataset, name, out_dir, filter_key, epochs, seed):
    """Train one twin BC-GMM verifier (no rollout eval); return its last.pth."""
    run_dir = os.path.join(out_dir, name)
    if os.path.exists(run_dir):
        subprocess.run(["rm", "-rf", run_dir], check=True)
    cmd = ["python", os.path.join(HERE, "train_bc.py"),
           "--dataset", dataset, "--name", name, "--output_dir", out_dir,
           "--filter_key", filter_key, "--epochs", str(epochs),
           "--steps_per_epoch", "100", "--n_rollouts", "0",
           "--seed", str(seed), "--wandb", "0"]
    log = os.path.join(out_dir, name + ".log")
    with open(log, "w") as fp:
        rc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError("twin training failed (%s); see %s" % (name, log))
    cks = sorted(glob.glob(os.path.join(run_dir, "*", "last.pth")))
    if not cks:
        raise RuntimeError("no checkpoint for twin %s" % name)
    return cks[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--R", type=int, default=4, help="number of self-training rounds (the tuned hyperparameter)")
    p.add_argument("--m", type=int, default=2, help="fresh twins per round (>=2 so every demo has an OOB score)")
    p.add_argument("--frac", type=float, default=0.55, help="subsample fraction each twin trains on")
    p.add_argument("--min_sub", type=int, default=30, help="floor on per-twin training-subset size (keeps twins well-trained as the kept set shrinks)")
    p.add_argument("--tau", type=float, default=0.30, help="round-1 keep threshold (lenient -> protect recall)")
    p.add_argument("--tau_final", type=float, default=None,
                   help="final-round threshold; linearly annealed from --tau across rounds (default: = --tau, constant)")
    p.add_argument("--key_prefix", default="iter", help="filter-key name prefix (avoid clobbering other runs)")
    p.add_argument("--eps", type=float, default=0.30)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_states", type=int, default=80)
    p.add_argument("--epochs", type=int, default=40, help="twin training epochs")
    p.add_argument("--pool_filter_key", default="pool_all")
    p.add_argument("--seed_base", type=int, default=7000, help="base seed; round r uses seed_base + 1000*r")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f = h5py.File(args.dataset, "r")
    pool = sorted([d.decode() for d in f["mask/%s" % args.pool_filter_key][:]], key=idx_of)
    f.close()
    clean, corrupt = gt_sets(args.dataset)
    tau_tag = int(round(args.tau * 100))

    tau_final = args.tau if args.tau_final is None else args.tau_final

    def tau_for_round(r):  # linear anneal from tau (round 1) to tau_final (round R)
        if args.R <= 1:
            return args.tau
        return args.tau + (tau_final - args.tau) * (r - 1) / (args.R - 1)

    current = list(pool)
    rounds = []
    print("ITER start: pool=%d (%d clean / %d corrupt)  R=%d m=%d frac=%.2f tau=%.2f->%.2f eps=%.2f"
          % (len(pool), len(clean), len(corrupt), args.R, args.m, args.frac, args.tau, tau_final, args.eps))

    for r in range(1, args.R + 1):
        tau_r = tau_for_round(r)
        round_seed = args.seed_base + 1000 * r            # fresh re-seed, decorrelated per round
        # --- build m fresh subsamples of the CURRENT kept set ---
        boot_keys, train_sets = [], []
        n_sub = max(2, int(round(args.frac * len(current))))
        n_sub = max(n_sub, min(args.min_sub, len(current)))  # floor so twins stay well-trained
        n_sub = min(n_sub, len(current))
        for j in range(args.m):
            rng = np.random.RandomState(round_seed * 10 + j)
            sub = sorted(rng.choice(current, n_sub, replace=False).tolist(), key=idx_of)
            key = "itw_r%d_s%d" % (r, j)
            FileUtils.create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=sub, key_name=key)
            boot_keys.append(key)
            train_sets.append(set(sub))
        # --- train m fresh twins on those subsamples ---
        ckpts = []
        for j, key in enumerate(boot_keys):
            name = "itw_r%d_t%d" % (r, j)
            ck = train_twin(args.dataset, name, args.out, key, args.epochs, seed=round_seed + j)
            ckpts.append(ck)
        # --- score current kept set with each twin, OOB-aggregate ---
        per_twin, holder = [], []
        for j, ck in enumerate(ckpts):
            s = score_one_verifier(ck, args.dataset, current, holder, args.K, args.eps,
                                   args.max_states, device, seed=round_seed + 500 + j)
            per_twin.append(s)
        vbar = {}
        for d in current:
            oob = [per_twin[j][d] for j in range(args.m) if d not in train_sets[j]]
            vbar[d] = float(np.mean(oob)) if oob else float(np.mean([per_twin[j][d] for j in range(args.m)]))
        keep = [d for d in current if vbar[d] >= tau_r]
        # --- round stats ---
        acc = ordering_acc(vbar, clean, corrupt)
        nck = sum(1 for d in keep if d in clean)
        nkk = sum(1 for d in keep if d in corrupt)
        mc = float(np.mean([vbar[d] for d in current if d in clean])) if any(d in clean for d in current) else None
        mk = float(np.mean([vbar[d] for d in current if d in corrupt])) if any(d in corrupt for d in current) else None
        key_R = "%s_R%d" % (args.key_prefix, r)
        FileUtils.create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=keep, key_name=key_R)
        rounds.append({
            "round": r, "filter_key": key_R, "round_seed": round_seed, "tau_r": tau_r,
            "n_in": len(current), "n_keep": len(keep),
            "n_clean_kept": nck, "n_corrupt_kept": nkk,
            "purity": nck / max(len(keep), 1), "recall_clean": nck / max(len(clean), 1),
            "ordering_acc": acc, "clean_vbar": mc, "corrupt_vbar": mk,
        })
        print("  round %d (tau=%.2f): in=%d -> keep %d (%d clean +%d corrupt) purity=%.3f recall=%.3f ord_acc=%.3f "
              "clean v̄=%.3f corrupt v̄=%s"
              % (r, tau_r, len(current), len(keep), nck, nkk, nck / max(len(keep), 1),
                 nck / max(len(clean), 1), acc if acc is not None else -1,
                 mc if mc is not None else -1, ("%.3f" % mk) if mk is not None else "n/a"))
        current = keep
        if len(current) < 2:
            print("  (kept set collapsed; stopping early)")
            break

    summary = {"dataset": args.dataset, "R": args.R, "m": args.m, "frac": args.frac,
               "tau": args.tau, "eps": args.eps, "K": args.K, "epochs": args.epochs,
               "n_clean": len(clean), "n_corrupt": len(corrupt), "rounds": rounds}
    with open(os.path.join(args.out, "itertwin_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print("DONE ->", os.path.join(args.out, "itertwin_summary.json"))


if __name__ == "__main__":
    main()
