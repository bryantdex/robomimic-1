"""
Verification stage (steps 1-2 of the algorithm), realized for Lift BC.

For each demonstration trajectory y_i (the "rollout with answer a_i"), we
"re-prompt the base policy K times" at each visited state and ask whether the
demonstrated action is something the policy itself would produce (i.e. whether
a_i is consistent with the model's own consensus over the data). Concretely, at
each state s_t we draw K action samples from the base BC-GMM policy and count
the fraction that fall within epsilon of the demonstrated action a_t:

    v_t = (1/K) * sum_k 1[ || sample_k - a_t || <= eps ]      ("VALID" fraction)
    v_bar(i) = mean_t v_t                                     (verification score)

We then aggregate the pseudo-label exactly as in the algorithm:
    score(i) = n(i) * v_bar(i)   with n(i) = 1 per demo (each demo is one rollout)
and write filter keys that keep only demos whose score >= threshold tau
(step 3's "skip prompts below threshold").

Outputs:
  - <out>/verification_scores.json : per-demo v_bar, quality label, score
  - filter keys "verified_geq_q{q}" written into the hdf5 for several tau (quantiles)
"""
import argparse
import json
import os
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import h5py


def quality_label_map(hdf5_path):
    """Map each demo -> {better, okay, worse} using the dataset's built-in keys
    (used ONLY for reporting/validation that v_bar is meaningful, NOT for filtering)."""
    f = h5py.File(hdf5_path, "r")
    lab = {}
    for q in ["better", "okay", "worse"]:
        if "mask/%s" % q in f:
            for d in f["mask/%s" % q][:]:
                lab[d.decode("utf-8")] = q
    f.close()
    return lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="base policy checkpoint (.pth)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, help="output dir for scores + filter keys")
    p.add_argument("--K", type=int, default=32, help="re-prompts (action samples) per state")
    p.add_argument("--eps", type=float, default=0.30, help="L2 tolerance for VALID")
    p.add_argument("--max_states", type=int, default=80, help="cap states scored per demo")
    p.add_argument("--quantiles", type=float, nargs="+",
                   default=[], help="tau quantiles of v_bar to keep above")
    p.add_argument("--abs_taus", type=float, nargs="+",
                   default=[], help="absolute v_bar thresholds tau to keep above")
    p.add_argument("--pool_filter_key", default=None,
                   help="if set, only score/filter demos in this filter key (the candidate pool)")
    p.add_argument("--key_prefix", default="verified", help="prefix for created filter keys")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.ckpt, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    algo = policy.policy
    algo.set_eval()
    net = algo.nets["policy"]
    # enable true GMM variance for sampling (turn off low-noise eval mode)
    prev_lne = getattr(net, "low_noise_eval", None)
    net.low_noise_eval = False

    obs_keys = list(config.observation.modalities.obs.low_dim)
    print("obs keys:", obs_keys, "| K=%d eps=%.3f" % (args.K, args.eps))

    labels = quality_label_map(args.dataset)
    f = h5py.File(args.dataset, "r")
    if args.pool_filter_key is not None:
        demos = [d.decode("utf-8") for d in f["mask/%s" % args.pool_filter_key][:]]
        demos = sorted(demos, key=lambda x: int(x.split("_")[1]))
        print("restricting to pool '%s': %d demos" % (args.pool_filter_key, len(demos)))
    else:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

    scores = {}
    rng = np.random.RandomState(0)
    for di, demo in enumerate(demos):
        g = f["data/%s" % demo]
        T = g.attrs["num_samples"]
        # subsample states for efficiency
        idx = np.arange(T)
        if T > args.max_states:
            idx = np.sort(rng.choice(T, args.max_states, replace=False))
        obs_dict = {}
        for k in obs_keys:
            arr = g["obs/%s" % k][:][idx].astype(np.float32)
            obs_dict[k] = torch.from_numpy(arr).to(device)
        acts = torch.from_numpy(g["actions"][:][idx].astype(np.float32)).to(device)  # [B, ac]

        with torch.no_grad():
            dist = net.forward_train(obs_dict=obs_dict, goal_dict=None)  # batch_shape [B]
            samp = dist.sample((args.K,))            # [K, B, ac]
            d = torch.linalg.norm(samp - acts.unsqueeze(0), dim=-1)  # [K, B]
            valid = (d <= args.eps).float()          # [K, B]
            v_t = valid.mean(dim=0)                  # [B] fraction VALID per state
            v_bar = float(v_t.mean().item())

        scores[demo] = {
            "v_bar": v_bar,
            "label": labels.get(demo, "?"),
            "n": 1,
            "len": int(T),
        }
        if di % 50 == 0:
            print("  scored %d/%d  %s v_bar=%.3f label=%s" % (di, len(demos), demo, v_bar, labels.get(demo, "?")))

    if prev_lne is not None:
        net.low_noise_eval = prev_lne
    f.close()

    # ---- report: does v_bar separate the known quality tiers? ----
    by_lab = {}
    for d, s in scores.items():
        by_lab.setdefault(s["label"], []).append(s["v_bar"])
    print("\n=== mean v_bar by (held-out) quality label ===")
    for lab in ["better", "okay", "worse", "?"]:
        if lab in by_lab:
            vs = by_lab[lab]
            print("  %-7s n=%3d  mean v_bar=%.3f" % (lab, len(vs), float(np.mean(vs))))

    # ---- pseudo-label score = n * v_bar ; build filter keys at tau quantiles ----
    all_v = np.array([scores[d]["v_bar"] for d in demos])  # n=1 so score == v_bar
    with open(os.path.join(args.out, "verification_scores.json"), "w") as fp:
        json.dump({"scores": scores, "K": args.K, "eps": args.eps}, fp, indent=2)

    created = {}
    for q in args.quantiles:
        tau = float(np.quantile(all_v, q))
        keep = [d for d in demos if scores[d]["v_bar"] >= tau]
        key = "%s_geq_q%d" % (args.key_prefix, int(round(q * 100)))
        FileUtils.create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=keep, key_name=key)
        created[key] = {"tau": tau, "n_keep": len(keep), "quantile": q}
        print("  filter key %-18s tau=%.3f keeps %d/%d demos" % (key, tau, len(keep), len(demos)))
    for tau in args.abs_taus:
        keep = [d for d in demos if scores[d]["v_bar"] >= tau]
        key = "%s_tau%03d" % (args.key_prefix, int(round(tau * 100)))
        FileUtils.create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=keep, key_name=key)
        created[key] = {"tau": tau, "n_keep": len(keep)}
        print("  filter key %-18s tau=%.3f keeps %d/%d demos" % (key, tau, len(keep), len(demos)))

    with open(os.path.join(args.out, "filter_keys.json"), "w") as fp:
        json.dump(created, fp, indent=2)
    print("\nDONE. scores + filter keys written to", args.out)


if __name__ == "__main__":
    main()
