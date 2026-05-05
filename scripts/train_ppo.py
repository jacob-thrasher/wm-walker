"""
Walker2d-v4 Training Script
Uses Stable Baselines3 (PPO/SAC) + Gymnasium with optional GPU acceleration.

Requirements:
    pip install stable-baselines3[extra] gymnasium[mujoco] torch

Usage:
    python train_walker2d.py                  # Auto-detect GPU
    python train_walker2d.py --algo sac       # Use SAC instead of PPO
    python train_walker2d.py --no-gpu         # Force CPU
    python train_walker2d.py --timesteps 3e6  # Custom timestep budget
"""

import os
import argparse
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


# ── Hyperparameters ────────────────────────────────────────────────────────────

PPO_HYPERPARAMS = dict(
    n_steps=2048,           # Steps per env per rollout
    batch_size=512,         # Mini-batch size for SGD updates
    n_epochs=10,            # Epochs per rollout
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda
    clip_range=0.2,         # PPO clip epsilon
    ent_coef=0.0,           # Entropy bonus (keep low for locomotion)
    vf_coef=0.5,            # Value-function loss coefficient
    max_grad_norm=0.5,      # Gradient clipping
    learning_rate=3e-4,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
)

SAC_HYPERPARAMS = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,  # Replay buffer capacity
    learning_starts=10_000, # Steps before first gradient update
    batch_size=256,
    tau=0.005,              # Soft-update coefficient
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",        # Automatic entropy tuning
    policy_kwargs=dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
    ),
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_env(env_id: str, rank: int, seed: int = 0):
    """Factory for a single monitored environment (used by SubprocVecEnv)."""
    def _init():
        env = gym.make(env_id, render_width=64, render_height=64)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def build_vec_env(env_id: str, n_envs: int, seed: int, use_subproc: bool):
    """Build a vectorised (optionally multi-process) environment."""
    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(
            [make_env(env_id, i, seed) for i in range(n_envs)]
        )
    else:
        vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    # Normalise observations and rewards — crucial for locomotion tasks
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Walker2d-v4 with SB3")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo",
                   help="RL algorithm (default: ppo)")
    p.add_argument("--timesteps", type=float, default=3e6,
                   help="Total environment steps (default: 3 000 000)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="Parallel envs (default: 8 for PPO, 1 for SAC)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu", action="store_true",
                   help="Disable GPU even if available")
    p.add_argument("--log-dir", default="logs/walker2d",
                   help="TensorBoard + checkpoint directory")
    p.add_argument("--checkpoint-freq", type=int, default=100_000,
                   help="Save a checkpoint every N steps (default: 100 000)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.no_gpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        # Enable TF32 for ~3× faster matmuls on Ampere+ GPUs (negligible precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[CPU] No CUDA device found — running on CPU")

    print(f"Algorithm : {args.algo.upper()}")
    print(f"Device    : {device}")
    print(f"Timesteps : {int(args.timesteps):,}")

    # ── Environment setup ─────────────────────────────────────────────────────
    ENV_ID = "Walker2d-v4"

    # PPO benefits from many parallel envs; SAC uses a single env + replay buffer
    if args.n_envs is not None:
        n_envs = args.n_envs
    else:
        n_envs = 8 if args.algo == "ppo" else 1

    # Use subprocess workers only for PPO (SAC is already sample-efficient)
    use_subproc = args.algo == "ppo" and n_envs > 1
    vec_env = build_vec_env(ENV_ID, n_envs, args.seed, use_subproc)

    # Separate normalised eval env (stats kept in sync with training env)
    eval_env = build_vec_env(ENV_ID, n_envs=1, seed=args.seed + 100,
                             use_subproc=False)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.log_dir}/best_model", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.log_dir}/best_model",
        log_path=f"{args.log_dir}/eval",
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // n_envs, 1),
        save_path=f"{args.log_dir}/checkpoints",
        name_prefix="walker2d",
        save_vecnormalize=True,   # Save normalisation stats too
        verbose=1,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ── Model instantiation ───────────────────────────────────────────────────
    algo_cls = PPO if args.algo == "ppo" else SAC
    hyperparams = PPO_HYPERPARAMS if args.algo == "ppo" else SAC_HYPERPARAMS

    model = algo_cls(
        policy="MlpPolicy",
        env=vec_env,
        verbose=0,
        seed=args.seed,
        device=device,
        # tensorboard_log=f"{args.log_dir}/tensorboard",
        **hyperparams,
    )

    print(f"\nModel policy network:\n{model.policy}\n")
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Trainable parameters: {n_params:,}")

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n=== Training started ===\n")
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
        tb_log_name=args.algo.upper(),
    )
    print("\n=== Training complete ===\n")

    # ── Save final model + normalisation stats ────────────────────────────────
    final_path = f"{args.log_dir}/final_model"
    model.save(final_path)
    vec_env.save(f"{args.log_dir}/vecnormalize_final.pkl")
    print(f"Final model saved  → {final_path}.zip")
    print(f"VecNormalize saved → {args.log_dir}/vecnormalize_final.pkl")

    # ── Quick deterministic evaluation ───────────────────────────────────────
    print("\n=== Post-training evaluation (20 episodes) ===")
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()