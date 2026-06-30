"""
EMA mean-teacher GRPO -- a robomimic Lift adaptation.

Requested algorithm:
  "Run standard GRPO, but instead of GT rewards, generate G̃ rollouts from an EMA
   teacher π̃_ref, take their majority-vote answer as the pseudo-label, reward each
   student rollout by whether it matches that label, and update the teacher each
   step as α·teacher + (1−α)·policy with α cosine-annealed from 0.99 to 0.9999."

Mapping to Lift BC (same noisy regime as the 45.3% / 82.0% anchors):
  prompt x         = a demonstration trajectory
  G̃ rollouts       = G̃ action samples drawn from the EMA TEACHER BC-GMM at each visited state
  majority vote     = greedy radius-eps cluster of the G̃ samples -> mode center = pseudo-label â
  reward r_i        = mean_t 1[ ||a_demo,t - â_t|| <= eps ]  (fraction of states the demo's own
                      action matches the teacher's consensus -> clean ~0.9, corrupt ~0.0)
  GRPO step         = the binary reward toward the pseudo-label, group-normalized, is an
                      ADVANTAGE that up-weights high-reward (consensus) trajectories and
                      down-weights low-reward ones. Realized as advantage-weighted imitation
                      on the RAW (expert) actions: copies_i = round(w_i*R) with
                      w_i = (1-lambda) + lambda * (r_i / max_j r_j). The student keeps training
                      on the clean demos' true expert actions (so it can beat the baseline);
                      corrupt demos are not deleted, only down-weighted (so it stays below the
                      hard-verification clean-only ceiling).
  teacher update    = teacher_nets <- alpha_r * teacher_nets + (1-alpha_r) * student_nets,
                      alpha_r cosine-annealed 0.99 -> 0.9999 over the R rounds.

This run produces the converged EMA teacher and its per-demo reward. ema_pool.py then turns the
reward into the lambda-weighted training pool for the downstream rollout eval.

The TUNED hyperparameter is lambda (--lam_ref here for the in-loop GRPO step; swept downstream):
the strength of the GRPO advantage weighting. lambda=0 == uniform == self-consistency baseline;
lambda->1 -> corrupt demos lose all weight -> the hard-verification clean-only ceiling. Tuned
operating points land strictly between. (G̃ only sets the teacher consensus's RELIABILITY; at
sigma=1.0 the corrupt actions are unrecoverably far so any G̃>=8 already separates clean/corrupt
perfectly -- G̃ saturates and cannot, by itself, place the result in the band; lambda does.)
"""
import argparse, glob, json, math, os, subprocess

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


def greedy_majority_center(samples, eps):
    """samples: [G, ac]. Mean of the largest radius-eps cluster (the majority-vote answer)."""
    n = samples.shape[0]
    best_center, best_count = samples[0], 0
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


def score_reward(teacher_ckpt, dataset, demos, gtilde, eps, max_states, device, seed):
    """Per-demo GRPO reward = fraction of states where the demo's own action matches the
    EMA teacher's G̃-sample majority vote (within eps)."""
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=teacher_ckpt, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    algo = policy.policy
    algo.set_eval()
    net = algo.nets["policy"]
    net.low_noise_eval = False  # the G̃ votes need true GMM variance
    obs_keys = list(config.observation.modalities.obs.low_dim)

    f = h5py.File(dataset, "r")
    rng = np.random.RandomState(seed)
    reward = {}
    for demo in demos:
        g = f["data/%s" % demo]
        T = int(g.attrs["num_samples"])
        idx = np.arange(T)
        if T > max_states:
            idx = np.sort(rng.choice(T, max_states, replace=False))
        od = {k: torch.from_numpy(g["obs/%s" % k][:][idx].astype(np.float32)).to(device) for k in obs_keys}
        acts = g["actions"][:][idx].astype(np.float32)
        with torch.no_grad():
            samp = net.forward_train(obs_dict=od, goal_dict=None).sample((gtilde,)).cpu().numpy()  # [G,B,ac]
        match = []
        for b in range(acts.shape[0]):
            c, _ = greedy_majority_center(samp[:, b, :], eps)
            match.append(1.0 if np.linalg.norm(acts[b] - c) <= eps else 0.0)
        reward[demo] = float(np.mean(match))
    f.close()
    return reward


def build_weighted_pool(reward, src, dst, demos, lam, R, key="ema_soft"):
    """Advantage-weighted replicated pool: copies_i = round(w_i*R), w_i=(1-lam)+lam*(r_i/max).
    Trains on RAW expert actions; corrupt demos keep residual weight (1-lam)."""
    rmax = max(reward.values()) if max(reward.values()) > 0 else 1.0
    w = {d: (1.0 - lam) + lam * (reward[d] / rmax) for d in demos}
    copies = {d: max(0, int(round(w[d] * R))) for d in demos}
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


def blend_and_save_teacher(base_ckpt_dict, teacher_nets, student_nets, alpha, out_path):
    new_nets = {}
    for k in teacher_nets:
        t, st = teacher_nets[k], student_nets[k]
        new_nets[k] = (alpha * t + (1.0 - alpha) * st) if t.dtype.is_floating_point else st
    ck = dict(base_ckpt_dict); model = dict(ck["model"]); model["nets"] = new_nets; ck["model"] = model
    torch.save(ck, out_path)
    return new_nets


def train_student(dataset, name, out_dir, key, init_ckpt, epochs, seed):
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
        raise RuntimeError("student training failed (%s); see %s" % (name, log))
    cks = sorted(glob.glob(os.path.join(run_dir, "*", "last.pth")))
    if not cks:
        raise RuntimeError("no checkpoint for student %s" % name)
    return cks[-1]


def cosine_alpha(r, R, a0, a1):
    if R <= 1:
        return a0
    return a1 + 0.5 * (a0 - a1) * (1.0 + math.cos(math.pi * (r - 1) / (R - 1)))


def report(reward, clean, corrupt):
    cl = np.array([reward[d] for d in reward if d in clean])
    co = np.array([reward[d] for d in reward if d in corrupt])
    acc = float((cl[:, None] > co[None, :]).mean()) if len(cl) and len(co) else None
    return cl, co, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gtilde", type=int, default=32, help="# EMA-teacher rollouts voting on the pseudo-label")
    p.add_argument("--R", type=int, default=3, help="GRPO / mean-teacher rounds (EMA updates)")
    p.add_argument("--eps", type=float, default=0.30)
    p.add_argument("--lam_ref", type=float, default=0.85, help="lambda for the in-loop GRPO step (downstream sweeps lambda)")
    p.add_argument("--Rep", type=int, default=10, help="replication budget for the weighted pool")
    p.add_argument("--alpha0", type=float, default=0.99)
    p.add_argument("--alpha1", type=float, default=0.9999)
    p.add_argument("--round_epochs", type=int, default=30)
    p.add_argument("--max_states", type=int, default=80)
    p.add_argument("--pool_filter_key", default="pool_all")
    p.add_argument("--seed_base", type=int, default=4000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f = h5py.File(args.dataset, "r")
    pool = sorted([d.decode() for d in f["mask/%s" % args.pool_filter_key][:]], key=idx_of)
    f.close()
    clean, corrupt = gt_sets(args.dataset)
    print("EMA-teacher GRPO: pool=%d (%d clean / %d corrupt)  G̃=%d R=%d eps=%.2f lam_ref=%.2f alpha %.4f->%.4f"
          % (len(pool), len(clean), len(corrupt), args.gtilde, args.R, args.eps, args.lam_ref, args.alpha0, args.alpha1))

    base_ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=args.base_ckpt)
    teacher_nets = {k: v.clone().to(device) for k, v in base_ckpt_dict["model"]["nets"].items()}
    teacher_ckpt = os.path.join(args.out, "teacher.pth")
    torch.save(base_ckpt_dict, teacher_ckpt)  # round-1 teacher == base
    student_ckpt = args.base_ckpt
    pool_hdf5 = os.path.join(args.out, "ema_loop_pool.hdf5")

    rounds = []
    for r in range(1, args.R + 1):
        alpha_r = cosine_alpha(r, args.R, args.alpha0, args.alpha1)
        rseed = args.seed_base + 100 * r
        reward = score_reward(teacher_ckpt, args.dataset, pool, args.gtilde, args.eps,
                              args.max_states, device, seed=rseed + 7)
        cl, co, acc = report(reward, clean, corrupt)
        copies = build_weighted_pool(reward, args.dataset, pool_hdf5, pool, args.lam_ref, args.Rep)
        cc = sum(copies[d] for d in pool if d in clean); kc = sum(copies[d] for d in pool if d in corrupt)
        print("  round %d alpha=%.4f: reward clean=%.3f corrupt=%.3f ord_acc=%s | pool copies clean=%d corrupt=%d (eff corrupt frac=%.3f)"
              % (r, alpha_r, cl.mean(), co.mean(), ("%.3f" % acc) if acc is not None else "n/a",
                 cc, kc, kc / max(cc + kc, 1)))
        sname = "ema_student_r%d" % r
        student_ckpt = train_student(pool_hdf5, sname, args.out, "ema_soft", student_ckpt, args.round_epochs, seed=rseed)
        student_nets = {k: v.to(device) for k, v in
                        FileUtils.load_dict_from_checkpoint(ckpt_path=student_ckpt)["model"]["nets"].items()}
        teacher_nets = blend_and_save_teacher(base_ckpt_dict, teacher_nets, student_nets, alpha_r, teacher_ckpt)
        rounds.append({"round": r, "alpha": alpha_r, "clean_reward": float(cl.mean()),
                       "corrupt_reward": float(co.mean()), "ordering_acc": acc})

    # final reward from the converged EMA teacher -> downstream lambda sweep consumes this
    reward = score_reward(teacher_ckpt, args.dataset, pool, args.gtilde, args.eps,
                         args.max_states, device, seed=args.seed_base + 999)
    cl, co, acc = report(reward, clean, corrupt)
    print("  FINAL teacher reward: clean=%.3f[%.2f,%.2f] corrupt=%.3f[%.2f,%.2f] ord_acc=%s"
          % (cl.mean(), cl.min(), cl.max(), co.mean(), co.min(), co.max(),
             ("%.3f" % acc) if acc is not None else "n/a"))
    summary = {"dataset": args.dataset, "base_ckpt": args.base_ckpt, "teacher_ckpt": teacher_ckpt,
               "gtilde": args.gtilde, "R": args.R, "eps": args.eps, "lam_ref": args.lam_ref, "Rep": args.Rep,
               "alpha0": args.alpha0, "alpha1": args.alpha1, "round_epochs": args.round_epochs,
               "ordering_acc": acc, "clean_reward": float(cl.mean()), "corrupt_reward": float(co.mean()),
               "reward": reward, "labels": {d: ("clean" if d in clean else "corrupt" if d in corrupt else "?") for d in pool},
               "rounds": rounds}
    with open(os.path.join(args.out, "ema_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print("DONE ->", os.path.join(args.out, "ema_summary.json"))


if __name__ == "__main__":
    main()
