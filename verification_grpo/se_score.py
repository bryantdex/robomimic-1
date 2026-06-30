"""
Semantic-entropy scoring (steps 1-3 of the requested algorithm) for Lift BC.

Requested algorithm:
  "Sample G responses per unlabeled question, cluster them by semantic equivalence,
   reward each response by its cluster's empirical frequency p(c|q) ~ |c|/G, compute
   group-normalized advantages, and run the clipped policy-gradient update only on
   questions whose SEMANTIC ENTROPY falls within (delta_low, delta_high) -- thereby
   minimizing semantic entropy to favor self-consistent answers."

This is the SEMANTIC-ENTROPY label-free RLVR recipe. Like sc_score.py there is NO
verifier / no "re-prompt to check VALID" step (that is the separate self-verification
method, which scores 82.0% here). The only signals come from how the model's own
G samples cluster.

Mapping to Lift BC (same regime as the 45.3% / 82.0% anchors):
  * question x         = a demonstration trajectory
  * G responses        = G=32 action samples drawn from the FROZEN BASE BC-GMM policy
                         at each visited state s_t ("G responses per question")
  * semantic clustering= greedy radius-eps clustering of the G samples -> clusters c
                         with sizes |c| ("cluster by semantic equivalence")
  * p(c|q) = |c|/G     = cluster empirical frequency ("reward each response by freq")
  * semantic entropy   = H_t = -sum_c p(c) log p(c) at state s_t; per-demo
                         SE(i) = mean_t H_t  ("semantic entropy of the question")
  * reward of the demo = r(i) = mean_t p(c*_t|q), c*_t = cluster nearest the demo's
                         OWN action a_t (empirical freq of the demo's answer's cluster)

Clean demos visit in-distribution states where the base policy is confident -> the G
samples collapse to one big cluster -> LOW semantic entropy + HIGH demo-answer freq.
Action-corrupted demos drift to OOD states where the base policy is uncertain -> the
G samples scatter across many small clusters -> HIGH semantic entropy + LOW demo freq.
So semantic entropy is the self-consistency signal the filter window (delta_low,
delta_high) acts on.

Output: <out>/se_scores.json with per-demo {SE, r, M, label, len}.
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


def greedy_clusters(samples, eps):
    """samples: [G, ac]. Greedy radius-eps clustering by vote count.
    Returns list of (center [ac], count) for ALL clusters, plus the assignment so
    we can find which cluster the demo's own action falls in."""
    G = samples.shape[0]
    used = np.zeros(G, dtype=bool)
    clusters = []  # (center, count, member_mask)
    # seed clusters greedily, largest-first is approximated by iterating points and
    # always taking the biggest remaining radius-eps ball
    while not used.all():
        rem = np.where(~used)[0]
        best = None
        for i in rem:
            d = np.linalg.norm(samples - samples[i][None, :], axis=1)
            member = (d <= eps) & (~used)
            cnt = int(member.sum())
            if best is None or cnt > best[1]:
                best = (samples[member].mean(axis=0), cnt, member)
        center, cnt, member = best
        clusters.append((center, cnt))
        used[member] = True
    return clusters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="frozen base policy checkpoint (.pth)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--G", type=int, default=32, help="responses (action samples) per state to cluster")
    p.add_argument("--eps_cluster", type=float, default=0.30, help="radius for semantic-equivalence clustering")
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
    print("obs keys:", obs_keys, "| G=%d eps_cluster=%.3f" % (args.G, args.eps_cluster))

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
            samp = dist.sample((args.G,)).cpu().numpy()  # [G, B, ac]

        B = acts.shape[0]
        H_list, r_list, top_list = [], [], []
        for b in range(B):
            s_b = samp[:, b, :]                        # [G, ac] the G responses at this state
            clusters = greedy_clusters(s_b, args.eps_cluster)
            counts = np.array([c[1] for c in clusters], dtype=np.float64)
            pc = counts / counts.sum()                  # p(c|q) = |c|/G
            H = float(-(pc * np.log(pc + 1e-12)).sum())  # semantic entropy at this state
            H_list.append(H)
            top_list.append(float(counts.max() / args.G))  # majority fraction (for reporting)
            # reward of the demo's OWN answer = empirical freq of the cluster it falls in
            centers = np.stack([c[0] for c in clusters], axis=0)  # [nc, ac]
            d2 = np.linalg.norm(centers - acts[b][None, :], axis=1)
            ci = int(np.argmin(d2))
            # only credit the demo to that cluster if it is actually within eps of it,
            # else its answer is its own (size-1) cluster -> freq 1/G
            r_list.append(float(pc[ci]) if d2[ci] <= args.eps_cluster else 1.0 / args.G)
        SE = float(np.mean(H_list))
        r = float(np.mean(r_list))
        M = float(np.mean(top_list))
        scores[demo] = {"SE": SE, "r": r, "M": M, "label": labels.get(demo, "?"), "len": int(T)}
        if di % 10 == 0:
            print("  %3d/%d %s  SE=%.3f r=%.3f M=%.3f  label=%s"
                  % (di, len(demos), demo, SE, r, M, labels.get(demo, "?")))

    f.close()

    # ---- separation report ----
    by = {}
    for d, s in scores.items():
        by.setdefault(s["label"], {"SE": [], "r": []})
        by[s["label"]]["SE"].append(s["SE"])
        by[s["label"]]["r"].append(s["r"])
    print("\n=== semantic-entropy signals by ground-truth label ===")
    for lab in ["clean", "corrupted", "better", "okay", "worse", "?"]:
        if lab in by:
            SE = by[lab]["SE"]; R = by[lab]["r"]
            print("  %-9s n=%3d  SE(sem-entropy)=%.3f[%.3f,%.3f]  r(cluster-freq)=%.3f[%.3f,%.3f]"
                  % (lab, len(SE), np.mean(SE), min(SE), max(SE), np.mean(R), min(R), max(R)))
    if "clean" in by and "corrupted" in by:
        cl = np.array(by["clean"]["SE"]); co = np.array(by["corrupted"]["SE"])
        # ordering acc: clean should have LOWER entropy than corrupted
        pairs = (cl[:, None] < co[None, :]).mean()
        print("  SE ordering acc (clean<corrupt over all pairs) = %.3f" % pairs)
        print("  SE separation: clean mean=%.3f vs corrupt mean=%.3f" % (cl.mean(), co.mean()))

    with open(os.path.join(args.out, "se_scores.json"), "w") as fp:
        json.dump({"scores": scores, "G": args.G, "eps_cluster": args.eps_cluster}, fp, indent=2)
    print("\nDONE. wrote", os.path.join(args.out, "se_scores.json"))


if __name__ == "__main__":
    main()
