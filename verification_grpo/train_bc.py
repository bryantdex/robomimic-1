"""
BC training driver for the Verification-Filtered BC experiment on Lift (MH).

This is the "GRPO with binary reward toward the verified pseudo-label" stage,
realized as imitation learning: we train a BC policy on the subset of demos
selected by a verification filter key (or all demos for the baseline).
"""
import argparse
import json
import numpy as np

import robomimic
from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train


def build_config(args):
    config = config_factory(algo_name="bc")

    # ---- dataset (v0.5.0 multi-dataset list format) ----
    # filter key selects which demos to train on (None => all demos in file)
    filter_key = args.filter_key
    if filter_key in (None, "None", "none", "all", ""):
        filter_key = None
    config.train.data = [{"path": args.dataset, "filter_key": filter_key}]
    config.train.output_dir = args.output_dir
    config.experiment.name = args.name
    config.train.hdf5_filter_key = None
    config.train.hdf5_validation_filter_key = None
    config.experiment.validate = False
    config.train.hdf5_cache_mode = "all"
    config.train.num_data_workers = 2
    config.train.seed = args.seed

    # ---- optional warm-start: load initial policy weights from a checkpoint ----
    # (used by the EMA mean-teacher loop so each round's student continues from the
    #  previous student's weights -- weight-space EMA only makes sense in one basin)
    if args.init_ckpt not in (None, "None", "none", ""):
        config.experiment.ckpt_path = args.init_ckpt

    # ---- BC policy: GMM head (standard robomimic low-dim BC; also gives a
    #      stochastic policy we can sample K times for verification) ----
    config.algo.gmm.enabled = True
    config.algo.actor_layer_dims = [1024, 1024]
    config.algo.optim_params.policy.learning_rate.initial = 1e-3
    config.algo.optim_params.policy.learning_rate.decay_factor = 0.1
    config.algo.optim_params.policy.learning_rate.epoch_schedule = [int(args.epochs * 0.7)]

    # ---- training budget ----
    config.train.batch_size = 100
    config.train.num_epochs = args.epochs
    config.experiment.epoch_every_n_steps = args.steps_per_epoch

    # ---- rollout eval ---- (n_rollouts<=0 disables eval, e.g. for verifier-only training)
    config.experiment.rollout.enabled = bool(args.n_rollouts > 0)
    config.experiment.rollout.n = max(args.n_rollouts, 1)
    config.experiment.rollout.horizon = args.horizon
    config.experiment.rollout.rate = args.rollout_rate
    config.experiment.rollout.warmstart = 0
    config.experiment.rollout.terminate_on_success = True

    # ---- logging / saving ----
    config.experiment.logging.log_wandb = bool(args.wandb)
    config.experiment.logging.wandb_proj_name = args.wandb_proj
    config.experiment.logging.log_tb = True
    config.experiment.save.enabled = True
    config.experiment.save.on_best_rollout_success_rate = True
    config.experiment.save.every_n_epochs = max(args.epochs, 1)  # also save final

    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--filter_key", default=None)
    p.add_argument("--init_ckpt", default=None, help="warm-start policy weights from this checkpoint (EMA student continuation)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--rollout_rate", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wandb", type=int, default=0)
    p.add_argument("--wandb_proj", default="robomimic-lift-verification")
    args = p.parse_args()

    config = build_config(args)
    if args.filter_key is None:
        config.meta = config.meta  # noop, keep structure
    config.lock()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print("=== Training run: {} | filter_key={} | device={} ===".format(
        args.name, args.filter_key, device))
    train(config, device=device)


if __name__ == "__main__":
    main()
