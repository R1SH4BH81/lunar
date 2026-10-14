"""Automated runner for all remaining reward shaping experiments and benchmark compilation."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train import train
from scripts.evaluate import evaluate_model
from scripts.compare_experiments import compare_all

def run_pipeline(device: str = "auto"):
    print("=" * 70)
    print("STARTING END-TO-END REWARD SHAPING & BENCHMARK PIPELINE")
    print("=" * 70)
    start_total = time.time()

    # 1. Train Hover Hack Model (Reward Hacking demonstration)
    print("\n[1/4] Training Hover Hack Model (200,000 steps)...")
    train(wrapper_type="hover_hack", total_timesteps=200_000, n_envs=8, device=device)

    # 2. Train Potential-Shaped Model (The Fix)
    print("\n[2/4] Training Potential-Shaped Model (200,000 steps)...")
    train(wrapper_type="potential_shaped", total_timesteps=200_000, n_envs=8, device=device)

    # 3. Evaluate and generate GIF recordings
    print("\n[3/4] Recording Flight GIFs for each trained policy...")
    evaluate_model(
        model_path="models/ppo_lunar_hover_hack.zip",
        n_episodes=5,
        output_media_path="recordings/hover_hack_flight.gif",
        wrapper_mode="hover_hack"
    )

    evaluate_model(
        model_path="models/ppo_lunar_potential_shaped.zip",
        n_episodes=5,
        output_media_path="recordings/potential_shaped_landing.gif",
        wrapper_mode="none"
    )

    # 4. Generate side-by-side comparison charts
    print("\n[4/4] Generating comparative benchmarks & plots...")
    active_models = {
        "Baseline": {
            "path": "models/ppo_lunar_baseline.zip",
            "desc": "Standard LunarLander-v3",
            "gif": "recordings/baseline_landing.gif"
        },
        "Hover_Hack": {
            "path": "models/ppo_lunar_hover_hack.zip",
            "desc": "Reward Hacked (+1.0 hover bonus)",
            "gif": "recordings/hover_hack_flight.gif",
            "eval_wrapper": "hover_hack"
        },
        "Potential_Shaped": {
            "path": "models/ppo_lunar_potential_shaped.zip",
            "desc": "The Fix: PBRS (Ng et al. 1999)",
            "gif": "recordings/potential_shaped_landing.gif"
        }
    }
    compare_all(active_models, n_episodes=10, output_plot="recordings/comparison_chart.png")

    total_time = time.time() - start_total
    print(f"\nALL EXPERIMENTS COMPLETED IN {total_time/60:.2f} MINUTES!")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full Lunar Lander reward shaping experiments pipeline")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    run_pipeline(device=args.device)
