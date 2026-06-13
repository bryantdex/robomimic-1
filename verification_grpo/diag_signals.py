"""Diagnostic: which per-demo signal separates better vs worse demos?
Compares, under a (good) verifier policy:
  - mean GMM log-prob of demo actions
  - mean L2 (policy mode action vs demo action)
  - p90 L2 (worst-case disagreement)
  - fraction of states with logprob below a low percentile (consensus outliers)
  - trajectory length
"""
import argparse, h5py, numpy as np, torch
import robomimic.utils.file_utils as FileUtils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pool_filter_key", default="pool_noisy")
    ap.add_argument("--max_states", type=int, default=120)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=args.ckpt, device=dev, verbose=False)
    cfg, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
    algo = policy.policy; algo.set_eval()
    net = algo.nets["policy"]
    obs_keys = list(cfg.observation.modalities.obs.low_dim)

    f = h5py.File(args.dataset, "r")
    labs = {}
    for q in ["better", "okay", "worse"]:
        for d in f["mask/%s" % q][:]:
            labs[d.decode()] = q
    demos = [d.decode() for d in f["mask/%s" % args.pool_filter_key][:]]
    demos = sorted(demos, key=lambda x: int(x.split("_")[1]))

    rows = {}
    rng = np.random.RandomState(0)
    for demo in demos:
        g = f["data/%s" % demo]; T = g.attrs["num_samples"]
        idx = np.arange(T)
        if T > args.max_states:
            idx = np.sort(rng.choice(T, args.max_states, replace=False))
        od = {k: torch.from_numpy(g["obs/%s" % k][:][idx].astype(np.float32)).to(dev) for k in obs_keys}
        acts = torch.from_numpy(g["actions"][:][idx].astype(np.float32)).to(dev)
        with torch.no_grad():
            net.low_noise_eval = False
            dist = net.forward_train(obs_dict=od, goal_dict=None)
            logp = dist.log_prob(acts).cpu().numpy()         # [B]
            mode = dist.mean if hasattr(dist, "mean") else dist.sample()
            l2 = torch.linalg.norm(mode - acts, dim=-1).cpu().numpy()  # [B]
        rows[demo] = dict(label=labs.get(demo, "?"), mean_logp=float(np.mean(logp)),
                          mean_l2=float(np.mean(l2)), p90_l2=float(np.percentile(l2, 90)),
                          length=int(T))
    f.close()

    def by_lab(key):
        out = {}
        for d, r in rows.items():
            out.setdefault(r["label"], []).append(r[key])
        return {k: float(np.mean(v)) for k, v in out.items()}

    for key in ["mean_logp", "mean_l2", "p90_l2", "length"]:
        m = by_lab(key)
        sep = ""
        if "better" in m and "worse" in m:
            sep = "  (better-worse = %+.3f)" % (m["better"] - m["worse"])
        print("%-10s : %s%s" % (key, {k: round(v, 3) for k, v in m.items()}, sep))

    # rank correlation of each signal with "worse" (1 if worse else 0)
    import numpy as np2
    y = np2.array([1.0 if rows[d]["label"] == "worse" else 0.0 for d in demos])
    print("\nAUC-style: fraction of (better,worse) pairs correctly ordered")
    for key, hi_is_good in [("mean_logp", True), ("mean_l2", False), ("p90_l2", False), ("length", False)]:
        s = np2.array([rows[d][key] for d in demos])
        b = s[y == 0]; w = s[y == 1]
        # good demos should have higher score if hi_is_good else lower
        comp = 0; tot = 0
        for bi in b:
            for wi in w:
                tot += 1
                if (bi > wi) == hi_is_good:
                    comp += 1
                elif bi == wi:
                    comp += 0.5
        print("  %-10s ordering acc = %.3f" % (key, comp / tot))


if __name__ == "__main__":
    main()
