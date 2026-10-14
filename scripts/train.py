"""Modular Training Script for LunarLander-v3 using PPO and Reward Shaping Wrappers.

Supports:
- baseline: Standard LunarLander-v3 environment
- time_penalty: Adds frame cost to prevent lingering
- hover_hack: Adds +1 bonus for hovering upright (demonstrates reward hacking)
- potential_shaped: Uses potential-based reward shaping to fix reward hacking
"""

import os
import sys
import time
import argparse
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.wrappers import (
    TimePenaltyWrapper,
    HoverHackingWrapper,
    PotentialShapedWrapper,
    DetailedMetricsWrapper
)

def create_env_fn(wrapper_type: str = "baseline", seed: int = 42) -> Callable[[], gym.Env]:
    """Factory to create and wrap LunarLander-v3 environments."""
    def _init() -> gym.Env:
        env = gym.make("LunarLander-v3")
        
        # Apply selected shaping wrapper
        if wrapper_type == "time_penalty":
            env = TimePenaltyWrapper(env, frame_penalty=0.15)
        elif wrapper_type == "hover_hack":
            env = HoverHackingWrapper(env, upright_angle_thresh=0.25, vel_y_thresh=0.35, bonus=1.0)
        elif wrapper_type == "potential_shaped":
            env = PotentialShapedWrapper(env, gamma=0.99, frame_penalty=0.1)
        
        # Track metrics and monitor for SB3
        env = DetailedMetricsWrapper(env)
        env = Monitor(env)
        return env
    return _init

def train(
    wrapper_type: str = "baseline",
    total_timesteps: int = 350_000,
    n_envs: int = 8,
    save_dir: str = "models",
    log_dir: str = "logs",
    device: str = "auto",
    resume: bool = False
):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[Notice] CUDA was requested but torch.cuda.is_available() is False (CPU-only PyTorch build installed). Falling back to CPU or install CUDA PyTorch.")
        device = "cpu"
    elif device == "auto" and torch.cuda.is_available():
        device = "cuda"

    print("=" * 65)
    print(f"TRAINING PPO AGENT - MODE: [{wrapper_type.upper()}] | DEVICE: [{device.upper()}]")
    print(f"Total Timesteps: {total_timesteps:,} | Parallel Environments: {n_envs}")
    if device == "cuda":
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model_name = f"ppo_lunar_{wrapper_type}"
    save_path = os.path.join(save_dir, f"{model_name}.zip")

    # Create vectorized environments
    env_fns = create_env_fn(wrapper_type)
    vec_env = make_vec_env(env_fns, n_envs=n_envs)

    # Resume from existing model if present, otherwise instantiate fresh
    if resume and os.path.exists(save_path):
        print(f"[Resume] Loading existing model weights from {save_path}...")
        model = PPO.load(save_path, env=vec_env, device=device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            verbose=1,
            device=device
        )

    start_time = time.time()
    print(f"\nTraining started at {time.strftime('%X')}...")
    
    model.learn(total_timesteps=total_timesteps)
    
    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s ({elapsed/60:.2f} mins).")

    # Save final trained weights
    model.save(save_path)
    print(f"[Model Saved] Policy saved to: {save_path}")

    vec_env.close()
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on LunarLander-v3")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "time_penalty", "hover_hack", "potential_shaped"],
        help="Reward shaping mode"
    )
    parser.add_argument("--timesteps", type=int, default=350_000, help="Total timesteps")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint if available")
    args = parser.parse_args()

    train(wrapper_type=args.mode, total_timesteps=args.timesteps, n_envs=args.n_envs, device=args.device, resume=args.resume)
