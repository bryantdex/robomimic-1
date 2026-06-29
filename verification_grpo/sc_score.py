"""
Self-consistency scoring (steps 1-2 of the requested algorithm), realized for Lift BC.

This is the SELF-CONSISTENCY analog of verify.py. There is NO verifier / no
"re-prompt to check VALID" step. Instead we treat each demonstration trajectory
as a prompt x and, at each visited state s_t, draw `n` action samples ("rollouts")
from the FROZEN BASE BC-GMM policy and let them VOTE:

  * greedy radius-eps clustering of the n samples -> distinct answers + vote counts
  * mode_t        = mean of the largest cluster   (the majority answer at s_t)
  * m_t           = size of the largest cluster
  * f_t  = m_t/n  = MAJORITY FRACTION at s_t       (policy self-consistency)

Per-demo aggregates (the quantities the downstream algorithm consumes):
  * M(i)   = mean_t f_t                      top-answer vote share  -> gate vs kappa
  * Mref(i)= M(i)                            reference majority fraction -> u_x = g(Mref)
  * p(i)   = mean_t exp(-||a_t - mode_t||/h) SOFT vote share of the demo's own answer
             (= the frequency weight the demo's pseudo-label receives in the
              frequency-weighted sum over distinct answers)

Unlike the hard "VALID within eps -> v_bar" of self-verification (which drives
corrupted v_bar exactly to 0.0 and yields a clean 0/1 separation), the soft vote
share p(i) is a graded agreement with the policy's OWN consensus: corrupted demos
retain a small-but-nonzero weight, so frequency-weighted training on these weights
is fuzzier than hard verification filtering -- by construction this lands BETWEEN
the unfiltered baseline and the self-verification filter.

Output: <out>/sc_scores.json with per-demo {M, Mref, p, label}.
"""
import argparse, json, os
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import h5py


def quality_label_map(hdf5_path):
    f = h5py.File(hdf5_path, "r")
    lab = {}
    for q in ["clean", "corrupted", "better", "okay", "worse"]:
        if "mask/%s" % q in f:
            for d in f["mask/%s" % q][:]:
                lab[d.decode("utf-8")] = q
    f.close()
    return lab


def greedy_cluster_topfrac(samples, eps):
    """samples: [n, ac] numpy. Greedy radius-eps clustering by vote count.
    Returns (mode_center [ac], top_count int)."""
    n = samples.shape[0]
    unassigned = list(range(n))
    best_center, best_count = samples[0], 0
    # seed clusters greedily from remaining points
    remaining = samples.copy()
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]:
            continue
        d = np.linalg.norm(samples - samples[i][None, :], axis=1)
        member = (d <= eps) & (~used)
        cnt = int(member.sum())
        if cnt > best_count:
            best_count = cnt
            best_center = samples[member].mean(axis=0)
        used[member] = True
    return best_center, best_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="frozen base policy checkpoint (.pth)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=32, help="rollouts (action samples) per state to vote over")
    p.add_argument("--eps_cluster", type=float, default=0.30, help="radius for distinct-answer voting")
    p.add_argument("--h", type=float, default=0.60, help="kernel bandwidth for soft vote share of demo answer")
    p.add_argument("--max_states", type=int, default=80)
    p.add_argument("--pool_filter_key", default="pool_all")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.ckpt, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    algo = policy.policy
    algo.set_eval()
    net = algo.nets["policy"]
    net.low_noise_eval = False  # sample with true GMM variance

    obs_keys = list(config.observation.modalities.obs.low_dim)
    print("obs keys:", obs_keys, "| n=%d eps_cluster=%.3f h=%.3f" % (args.n, args.eps_cluster, args.h))

    labels = quality_label_map(args.dataset)
    f = h5py.File(args.dataset, "r")
    demos = [d.decode("utf-8") for d in f["mask/%s" % args.pool_filter_key][:]]
    demos = sorted(demos, key=lambda x: int(x.split("_")[1]))
    print("scoring pool '%s': %d demos" % (args.pool_filter_key, len(demos)))

    scores = {}
    rng = np.random.RandomState(0)
    for di, demo in enumerate(demos):
        g = f["data/%s" % demo]
        T = g.attrs["num_samples"]
        idx = np.arange(T)
        if T > args.max_states:
            idx = np.sort(rng.choice(T, args.max_states, replace=False))
        obs_dict = {k: torch.from_numpy(g["obs/%s" % k][:][idx].astype(np.float32)).to(device) for k in obs_keys}
        acts = g["actions"][:][idx].astype(np.float32)  # [B, ac]

        with torch.no_grad():
            dist = net.forward_train(obs_dict=obs_dict, goal_dict=None)
            samp = dist.sample((args.n,)).cpu().numpy()  # [n, B, ac]

        B = acts.shape[0]
        f_list, p_list = [], []
        for b in range(B):
            s_b = samp[:, b, :]                      # [n, ac] the n votes at this state
            mode_c, top_cnt = greedy_cluster_topfrac(s_b, args.eps_cluster)
            f_list.append(top_cnt / float(args.n))   # majority fraction
            dist_demo = np.linalg.norm(acts[b] - mode_c)
            p_list.append(float(np.exp(-dist_demo / args.h)))  # soft vote share of demo answer
        M = float(np.mean(f_list))
        pval = float(np.mean(p_list))
        scores[demo] = {"M": M, "Mref": M, "p": pval, "label": labels.get(demo, "?"), "len": int(T)}
        if di % 10 == 0:
            print("  %3d/%d %s  M=%.3f p=%.3f  label=%s" % (di, len(demos), demo, M, pval, labels.get(demo, "?")))

    f.close()

    # ---- separation report ----
    by = {}
    for d, s in scores.items():
        by.setdefault(s["label"], {"M": [], "p": []})
        by[s["label"]]["M"].append(s["M"])
        by[s["label"]]["p"].append(s["p"])
    print("\n=== self-consistency signals by ground-truth label ===")
    for lab in ["clean", "corrupted", "better", "okay", "worse", "?"]:
        if lab in by:
            M = by[lab]["M"]; P = by[lab]["p"]
            print("  %-9s n=%3d  M(maj-frac)=%.3f[%.3f,%.3f]  p(soft-vote)=%.3f[%.3f,%.3f]"
                  % (lab, len(M), np.mean(M), min(M), max(M), np.mean(P), min(P), max(P)))

    # ordering accuracy of p: does it rank clean above corrupted?
    if "clean" in by and "corrupted" in by:
        cl = np.array(by["clean"]["p"]); co = np.array(by["corrupted"]["p"])
        pairs = (cl[:, None] > co[None, :]).mean()
        print("  p ordering acc (clean>corrupt over all pairs) = %.3f" % pairs)
        print("  p separation: clean mean=%.3f vs corrupt mean=%.3f (ratio %.2f)"
              % (cl.mean(), co.mean(), cl.mean() / max(co.mean(), 1e-9)))

    with open(os.path.join(args.out, "sc_scores.json"), "w") as fp:
        json.dump({"scores": scores, "n": args.n, "eps_cluster": args.eps_cluster, "h": args.h}, fp, indent=2)
    print("\nDONE. wrote", os.path.join(args.out, "sc_scores.json"))


if __name__ == "__main__":
    main()
