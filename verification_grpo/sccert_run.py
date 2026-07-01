"""
Self-certainty GRPO -- a robomimic Lift adaptation (label-free / verifier-free).

Requested algorithm:
  "For each prompt you sample G completions, score each by self-certainty using the
   *online* (current) policy, normalize via (u_i - mean)/std within the group, and run
   a policy-gradient update with no external labels or verifiers."

Mapping to Lift BC (same noisy regime as the 45.3% / 82.0% anchors):
  prompt            = the Lift task; the GROUP of G completions = the G=80 candidate
                      demonstration trajectories in the pool (each demo = one completion).
  self-certainty u_i= how CONFIDENT the *online* (current) BC-GMM policy is in producing
                      demo i's own actions:  u_i = mean_t log pi_online(a_{i,t} | s_{i,t}).
                      This is the continuous analog of RLSC self-certainty (KL-from-uniform
                      / peakedness of the policy's predictive distribution evaluated at the
                      completion). NO re-prompt VALID check, NO vote, NO teacher -- the
                      policy scores ITSELF. A clean (on-manifold expert) trajectory gets a
                      high log-density; a sigma=1.0 action-corrupted trajectory gets a very
                      low one (probe: clean ~ +26, corrupt ~ -3, ordering acc 1.000).
  group-normalize   A_i = (u_i - mean_j u_j) / std_j u_j   over the G=80 demos.
  policy-gradient   The group-normalized self-certainty is an ADVANTAGE. The PG update that
                    increases log-prob of above-mean-advantage completions is realized as
                    advantage-WEIGHTED imitation (AWR-style) on the RAW expert actions:
                       w_i = exp(beta * A_i),  copies_i = round(R * w_i / max_j w_j).
                    beta = 0 -> uniform weights -> trains on all 80 -> the self-consistency
                    baseline (45.3%). beta -> large -> only the most self-certain (clean)
                    demos keep weight -> the hard-verification clean-only ceiling (82.0%).
                    Tuned beta lands strictly between. beta is the TUNED hyperparameter.
  ONLINE            The scorer is the CURRENT policy, not a frozen base / EMA teacher. Each
                    round re-scores self-certainty with the just-trained policy (warm-started
                    via train_bc.py --init_ckpt), so the self-certainty signal tracks the
                    online policy as it denoises. (This is the defining difference from the
                    frozen-base self-consistency method and the EMA mean-teacher method.)

This run converges the online policy and emits its FINAL per-demo self-certainty u_i.
sccert_pool.py then turns u_i into the beta-weighted training pool for the downstream
rollout eval / beta sweep. (At sigma=1.0 the clean/corrupt separation is saturated -- any
online policy already orders them perfectly -- so the number of rounds only sharpens an
already-perfect ordering; beta, not the loop length, places the result in the band.)
"""
import argparse, glob, json, os, subprocess

import numpy as np
import torch
import h5py

import robomimic.utils.file_utils as FileUtils

HERE = os.path.dirname(os.path.abspath(__file__))


def idx_of(d):
    return int(d.split("_")[1])


def gt_sets(hdf5_path):
    f = h5py.File(hdf5_path, "r")
    clean = set(d.decode() for d in f["mask/clean"][:]) if "mask/clean" in f else set()
    corrupt = set(d.decode() for d in f["mask/corrupted"][:]) if "mask/corrupted" in f else set()
    f.close()
    return clean, corrupt


def score_self_certainty(ckpt, dataset, demos, max_states, device, seed):
    """Per-demo self-certainty u_i = mean_t log pi(a_{i,t} | s_{i,t}) under the (online) policy.
    The policy scores its OWN confidence in the demo's actions -- no label, no verifier."""
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    algo = policy.policy
    algo.set_eval()
    net = algo.nets["policy"]
    net.low_noise_eval = False  # use the policy's true learned GMM (peakedness == self-certainty)
    obs_keys = list(config.observation.modalities.obs.low_dim)

    f = h5py.File(dataset, "r")
    rng = np.random.RandomState(seed)
    u = {}
    for demo in demos:
        g = f["data/%s" % demo]
        T = int(g.attrs["num_samples"])
        idx = np.arange(T)
        if T > max_states:
            idx = np.sort(rng.choice(T, max_states, replace=False))
        od = {k: torch.from_numpy(g["obs/%s" % k][:][idx].astype(np.float32)).to(device) for k in obs_keys}
        act = torch.from_numpy(g["actions"][:][idx].astype(np.float32)).to(device)
        with torch.no_grad():
            dist = net.forward_train(obs_dict=od, goal_dict=None)
            lp = dist.log_prob(act)  # [B]
        u[demo] = float(lp.mean().cpu())
    f.close()
    return u


def advantages(u, demos):
    """Group-normalize self-certainty into GRPO advantages A_i = (u_i - mean)/std."""
    vals = np.array([u[d] for d in demos], dtype=np.float64)
    mu, sd = vals.mean(), vals.std()
    sd = sd if sd > 1e-8 else 1.0
    return {d: float((u[d] - mu) / sd) for d in demos}, float(mu), float(sd)


def build_weighted_pool(adv, src, dst, demos, beta, R, key="sccert_soft"):
    """PG-as-weighted-imitation: w_i = exp(beta*A_i), copies_i = round(R * w_i/max).
    Trains on RAW expert actions; low-self-certainty (corrupt) demos keep residual weight
    for moderate beta, drop out for large beta."""
    w = {d: float(np.exp(beta * adv[d])) for d in demos}
    wmax = max(w.values()) if max(w.values()) > 0 else 1.0
    copies = {d: max(0, int(round(R * w[d] / wmax))) for d in demos}
    if os.path.exists(dst):
        os.remove(dst)
    s = h5py.File(src, "r"); o = h5py.File(dst, "w")
    dg = o.create_group("data")
    for k, v in s["data"].attrs.items():
        dg.attrs[k] = v
    names, tot, j = [], 0, 0
    for d in demos:
        for _ in range(copies[d]):
            nm = "demo_%d" % j
            s.copy("data/%s" % d, dg, name=nm)
            tot += int(dg[nm].attrs["num_samples"]); names.append(nm); j += 1
    dg.attrs["total"] = tot
    o["mask/%s" % key] = np.array(names, dtype="S")
    s.close(); o.close()
    return copies


def train_round(dataset, name, out_dir, key, init_ckpt, epochs, seed):
    run_dir = os.path.join(out_dir, name)
    if os.path.exists(run_dir):
        subprocess.run(["rm", "-rf", run_dir], check=True)
    cmd = ["python", os.path.join(HERE, "train_bc.py"),
           "--dataset", dataset, "--name", name, "--output_dir", out_dir,
           "--filter_key", key, "--epochs", str(epochs), "--steps_per_epoch", "100",
           "--n_rollouts", "0", "--init_ckpt", init_ckpt, "--seed", str(seed), "--wandb", "0"]
    log = os.path.join(out_dir, name + ".log")
    with open(log, "w") as fp:
        rc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError("round training failed (%s); see %s" % (name, log))
    cks = sorted(glob.glob(os.path.join(run_dir, "*", "last.pth")))
    if not cks:
        raise RuntimeError("no checkpoint for %s" % name)
    return cks[-1]


def report(u, clean, corrupt):
    cl = np.array([u[d] for d in u if d in clean])
    co = np.array([u[d] for d in u if d in corrupt])
    acc = float((cl[:, None] > co[None, :]).mean()) if len(cl) and len(co) else None
    return cl, co, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--R", type=int, default=3, help="online GRPO rounds (re-score with current policy each round)")
    p.add_argument("--beta_ref", type=float, default=1.0, help="in-loop advantage temperature (downstream sweeps beta)")
    p.add_argument("--Rep", type=int, default=10, help="replication budget for the weighted pool")
    p.add_argument("--round_epochs", type=int, default=25)
    p.add_argument("--max_states", type=int, default=80)
    p.add_argument("--pool_filter_key", default="pool_all")
    p.add_argument("--seed_base", type=int, default=7000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f = h5py.File(args.dataset, "r")
    pool = sorted([d.decode() for d in f["mask/%s" % args.pool_filter_key][:]], key=idx_of)
    f.close()
    clean, corrupt = gt_sets(args.dataset)
    print("Self-certainty GRPO: pool=%d (%d clean / %d corrupt)  R=%d beta_ref=%.2f Rep=%d round_epochs=%d"
          % (len(pool), len(clean), len(corrupt), args.R, args.beta_ref, args.Rep, args.round_epochs))

    cur_ckpt = args.base_ckpt
    pool_hdf5 = os.path.join(args.out, "sccert_loop_pool.hdf5")
    rounds = []
    for r in range(1, args.R + 1):
        rseed = args.seed_base + 100 * r
        u = score_self_certainty(cur_ckpt, args.dataset, pool, args.max_states, device, seed=rseed + 7)
        adv, mu, sd = advantages(u, pool)
        cl, co, acc = report(u, clean, corrupt)
        copies = build_weighted_pool(adv, args.dataset, pool_hdf5, pool, args.beta_ref, args.Rep)
        cc = sum(copies[d] for d in pool if d in clean); kc = sum(copies[d] for d in pool if d in corrupt)
        print("  round %d (online scorer): self-cert clean=%.2f corrupt=%.2f ord_acc=%s | pool copies clean=%d corrupt=%d (eff corrupt frac=%.3f)"
              % (r, cl.mean(), co.mean(), ("%.3f" % acc) if acc is not None else "n/a",
                 cc, kc, kc / max(cc + kc, 1)))
        cur_ckpt = train_round(pool_hdf5, "sccert_r%d" % r, args.out, "sccert_soft",
                               cur_ckpt, args.round_epochs, seed=rseed)
        rounds.append({"round": r, "clean_u": float(cl.mean()), "corrupt_u": float(co.mean()),
                       "ordering_acc": acc, "clean_copies": cc, "corrupt_copies": kc})

    # FINAL self-certainty from the converged online policy -> downstream beta sweep consumes this
    u = score_self_certainty(cur_ckpt, args.dataset, pool, args.max_states, device, seed=args.seed_base + 999)
    adv, mu, sd = advantages(u, pool)
    cl, co, acc = report(u, clean, corrupt)
    print("  FINAL online self-certainty: clean=%.2f[%.2f,%.2f] corrupt=%.2f[%.2f,%.2f] ord_acc=%s (mean=%.2f std=%.2f)"
          % (cl.mean(), cl.min(), cl.max(), co.mean(), co.min(), co.max(),
             ("%.3f" % acc) if acc is not None else "n/a", mu, sd))
    summary = {"dataset": args.dataset, "base_ckpt": args.base_ckpt, "final_ckpt": cur_ckpt,
               "R": args.R, "beta_ref": args.beta_ref, "Rep": args.Rep, "round_epochs": args.round_epochs,
               "ordering_acc": acc, "clean_u": float(cl.mean()), "corrupt_u": float(co.mean()),
               "u_mean": mu, "u_std": sd, "u": u, "adv": adv,
               "labels": {d: ("clean" if d in clean else "corrupt" if d in corrupt else "?") for d in pool},
               "rounds": rounds}
    with open(os.path.join(args.out, "sccert_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print("DONE ->", os.path.join(args.out, "sccert_summary.json"))


if __name__ == "__main__":
    main()
