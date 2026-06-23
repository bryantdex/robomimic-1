"""
Different-instantiation verification (the requested algorithm variant).

Algorithm: "Keep the generator as-is, but compute v̄ under a DIFFERENT
INSTANTIATION of the same model."

The current verification-filtered method is *self*-verification: the verifier is
the very same model instance trained on the (contaminated) pool that is also being
filtered. Generator and verifier are one instantiation, so the verifier's idiosyncratic
fit to the noisy actions is correlated with what it is asked to judge, and its per-demo
v̄ estimate is noisy.

Here we keep the generator/data pipeline unchanged but compute v̄ under M INDEPENDENT
instantiations of the same BC-GMM model (same architecture + training recipe, different
random seeds). For each state we sample K actions from each instantiation, count the
fraction within eps of the demonstrated action, and AVERAGE that VALID fraction across
the M instantiations:

    v_t^(m) = (1/K) Σ_k 1[ ||sample_k^(m) - a_t|| <= eps ]
    v̄_M(i) = mean_t [ (1/M) Σ_{m<=M} v_t^(m) ]

M (number of instantiations averaged) is the tuned hyperparameter. M=1 reproduces the
current single-instantiation self-verification; larger M averages out per-instance
estimation variance -> cleaner separation -> better filtering.

For each M and threshold tau we write a filter key keeping demos with v̄_M >= tau, and
report separation quality (ordering accuracy = AUC of v̄ as a clean-vs-corrupted classifier).
"""
import argparse, json, os
import numpy as np
import torch
import h5py
import robomimic.utils.file_utils as FileUtils


def gt_sets(hdf5_path):
    f = h5py.File(hdf5_path, "r")
    clean = set(d.decode() for d in f["mask/clean"][:]) if "mask/clean" in f else set()
    corrupt = set(d.decode() for d in f["mask/corrupted"][:]) if "mask/corrupted" in f else set()
    f.close()
    return clean, corrupt


def score_one_verifier(ckpt, dataset, demos, obs_keys_holder, K, eps, max_states, device, seed):
    """Return {demo: v_bar} for one verifier instantiation."""
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    algo = policy.policy
    algo.set_eval()
    net = algo.nets["policy"]
    prev_lne = getattr(net, "low_noise_eval", None)
    net.low_noise_eval = False
    obs_keys = list(config.observation.modalities.obs.low_dim)
    obs_keys_holder.append(obs_keys)

    f = h5py.File(dataset, "r")
    rng = np.random.RandomState(seed)  # state subsampling fixed per verifier
    out = {}
    for demo in demos:
        g = f["data/%s" % demo]
        T = g.attrs["num_samples"]
        idx = np.arange(T)
        if T > max_states:
            idx = np.sort(rng.choice(T, max_states, replace=False))
        obs_dict = {k: torch.from_numpy(g["obs/%s" % k][:][idx].astype(np.float32)).to(device) for k in obs_keys}
        acts = torch.from_numpy(g["actions"][:][idx].astype(np.float32)).to(device)
        with torch.no_grad():
            dist = net.forward_train(obs_dict=obs_dict, goal_dict=None)
            samp = dist.sample((K,))                      # [K, B, ac]
            d = torch.linalg.norm(samp - acts.unsqueeze(0), dim=-1)  # [K, B]
            v_t = (d <= eps).float().mean(dim=0)          # [B]
            out[demo] = float(v_t.mean().item())
    f.close()
    if prev_lne is not None:
        net.low_noise_eval = prev_lne
    del policy, algo, net
    torch.cuda.empty_cache()
    return out


def ordering_acc(scores, clean, corrupt):
    """AUC: fraction of (clean,corrupt) pairs with v_clean > v_corrupt (ties=0.5)."""
    vc = [scores[d] for d in scores if d in clean]
    vk = [scores[d] for d in scores if d in corrupt]
    if not vc or not vk:
        return None
    n = wins = 0
    for a in vc:
        for b in vk:
            n += 1
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="verifier instantiation checkpoints (.pth)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--eps", type=float, default=0.30)
    p.add_argument("--max_states", type=int, default=80)
    p.add_argument("--pool_filter_key", default="pool_all")
    p.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--tau", type=float, default=0.30)
    p.add_argument("--train_keys", nargs="+", default=None,
                   help="per-ckpt training-subset filter key (parallel to --ckpts); enables OOB aggregation")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f = h5py.File(args.dataset, "r")
    demos = [d.decode() for d in f["mask/%s" % args.pool_filter_key][:]]
    demos = sorted(demos, key=lambda x: int(x.split("_")[1]))
    # per-verifier training sets (for OOB aggregation)
    train_sets = None
    if args.train_keys:
        assert len(args.train_keys) == len(args.ckpts)
        train_sets = [set(d.decode() for d in f["mask/%s" % k][:]) for k in args.train_keys]
    f.close()
    clean, corrupt = gt_sets(args.dataset)
    print("pool=%s  %d demos  (%d clean / %d corrupt)  K=%d eps=%.2f  M up to %d"
          % (args.pool_filter_key, len(demos), len(clean), len(corrupt), args.K, args.eps, max(args.Ms)))

    # ---- score each independent instantiation ----
    per_verifier = []  # list of {demo: v_bar}
    holder = []
    for vi, ckpt in enumerate(args.ckpts):
        s = score_one_verifier(ckpt, args.dataset, demos, holder, args.K, args.eps,
                               args.max_states, device, seed=100 + vi)
        per_verifier.append(s)
        acc = ordering_acc(s, clean, corrupt)
        mc = np.mean([s[d] for d in s if d in clean]); mk = np.mean([s[d] for d in s if d in corrupt])
        print("  instantiation %d: ordering_acc=%.3f  clean v̄=%.3f  corrupt v̄=%.3f" % (vi, acc, mc, mk))

    nV = len(per_verifier)

    def ensemble_vbar(demo, M):
        """Average v̄ over the first M instantiations. With train_sets, use only the
        instantiations that did NOT train on `demo` (out-of-bag); fall back to all M
        if the demo was in every training subset."""
        if train_sets is not None:
            oob = [per_verifier[m][demo] for m in range(M) if demo not in train_sets[m]]
            if oob:
                return float(np.mean(oob))
        return float(np.mean([per_verifier[m][demo] for m in range(M)]))

    # ---- ensemble over first M instantiations for each requested M ----
    agg = "oob" if train_sets is not None else "avg"
    summary = {"eps": args.eps, "K": args.K, "tau": args.tau, "n_verifiers": nV, "agg": agg, "per_M": {}}
    created = {}
    for M in args.Ms:
        if M > nV:
            continue
        ens = {d: ensemble_vbar(d, M) for d in demos}
        acc = ordering_acc(ens, clean, corrupt)
        mc = float(np.mean([ens[d] for d in demos if d in clean]))
        mk = float(np.mean([ens[d] for d in demos if d in corrupt]))
        keep = [d for d in demos if ens[d] >= args.tau]
        n_clean_kept = sum(1 for d in keep if d in clean)
        n_corrupt_kept = sum(1 for d in keep if d in corrupt)
        key = "ens_M%d_tau%03d" % (M, int(round(args.tau * 100)))
        FileUtils.create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=keep, key_name=key)
        summary["per_M"][str(M)] = {
            "filter_key": key, "ordering_acc": acc, "clean_vbar": mc, "corrupt_vbar": mk,
            "n_keep": len(keep), "n_clean_kept": n_clean_kept, "n_corrupt_kept": n_corrupt_kept,
            "purity": n_clean_kept / max(len(keep), 1),
            "recall_clean": n_clean_kept / max(len(clean), 1),
            "ens_vbar": ens,
        }
        created[key] = M
        print("  M=%d  ordering_acc=%.3f  clean v̄=%.3f corrupt v̄=%.3f | keep %d (%d clean +%d corrupt) purity=%.2f"
              % (M, acc, mc, mk, len(keep), n_clean_kept, n_corrupt_kept, n_clean_kept / max(len(keep), 1)))

    with open(os.path.join(args.out, "ensemble_scores.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    with open(os.path.join(args.out, "filter_keys.json"), "w") as fp:
        json.dump(created, fp, indent=2)
    print("DONE ->", args.out)


if __name__ == "__main__":
    main()
