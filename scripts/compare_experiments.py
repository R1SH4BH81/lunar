"""Compare Baseline vs Hacked vs Potential-Shaped RL models.

Generates comparative performance plots and tabular summaries.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.evaluate import evaluate_model

def compare_all(models_dict: dict, n_episodes: int = 15, output_plot: str = "recordings/comparison_chart.png"):
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE EXPERIMENT COMPARISON")
    print("=" * 70)

    results = []

    for name, config in models_dict.items():
        model_path = config["path"]
        wrapper_mode = config.get("eval_wrapper", "none")
        gif_path = config.get("gif", f"recordings/{name}.gif")

        if not os.path.exists(model_path):
            print(f"[Warning] Model file {model_path} not found. Skipping {name}...")
            continue

        print(f"\n---> Evaluating: {name} ({config['desc']})")
        metrics = evaluate_model(
            model_path=model_path,
            n_episodes=n_episodes,
            output_media_path=gif_path,
            wrapper_mode=wrapper_mode
        )
        metrics["Experiment"] = name
        metrics["Description"] = config["desc"]
        results.append(metrics)

    if not results:
        print("No models were evaluated.")
        return

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)
    summary_cols = ["Experiment", "mean_score", "solved_pct", "landed_pct", "mean_length", "mean_main_fuel"]
    print(df[summary_cols].to_string(index=False))

    # Generate comparison bar charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lunar Lander RL: Reward Shaping & Reward Hacking Analysis", fontsize=15, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(df)]

    # 1. Mean Score (Target >= 200)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df["Experiment"], df["mean_score"], color=colors, edgecolor="black")
    ax1.axhline(200, color="green", linestyle="--", label="Solved Threshold (+200)")
    ax1.set_title("Mean Game Score (Higher is Better)")
    ax1.set_ylabel("Score")
    ax1.legend()
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 5, f"{yval:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Solved Rate & Landing Rate
    ax2 = axes[0, 1]
    x = np.arange(len(df))
    width = 0.35
    ax2.bar(x - width/2, df["solved_pct"], width, label="Solved Rate (>=200)", color="#3470a3", edgecolor="black")
    ax2.bar(x + width/2, df["landed_pct"], width, label="Safe Landing Rate", color="#2ca02c", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Experiment"])
    ax2.set_title("Success Percentages (%)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 110)
    ax2.legend()

    # 3. Episode Duration (Frames) - Shows Hovering vs Landing
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df["Experiment"], df["mean_length"], color=colors, edgecolor="black")
    ax3.set_title("Average Flight Duration (Frames)")
    ax3.set_ylabel("Frames (Max 1000)")
    ax3.axhline(1000, color="red", linestyle=":", label="Timeout Limit (1000 frames)")
    ax3.legend()
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval:.0f}", ha="center", va="bottom", fontsize=10)

    # 4. Main Engine Fuel Usage
    ax4 = axes[1, 1]
    bars4 = ax4.bar(df["Experiment"], df["mean_main_fuel"], color=colors, edgecolor="black")
    ax4.set_title("Average Main Engine Fuel Ticks")
    ax4.set_ylabel("Ticks")
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"\n[Saved] Comparison charts saved to: {output_plot}")

    # Save CSV
    csv_path = os.path.splitext(output_plot)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"[Saved] Metrics CSV saved to: {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Lunar Lander RL experiments")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes per model")
    args = parser.parse_args()

    MODELS = {
        "Baseline": {
            "path": "models/ppo_lunar_baseline.zip",
            "desc": "Standard LunarLander-v3 (Default Reward)",
            "gif": "recordings/baseline_landing.gif"
        },
        "Hover_Hack": {
            "path": "models/ppo_lunar_hover_hack.zip",
            "desc": "Reward Hacking (+1.0/frame for hovering slow & upright)",
            "gif": "recordings/hover_hack_flight.gif",
            "eval_wrapper": "hover_hack"
        },
        "Potential_Shaped": {
            "path": "models/ppo_lunar_potential_shaped.zip",
            "desc": "The Fix: Potential-Based Reward Shaping (PBRS)",
            "gif": "recordings/potential_shaped_landing.gif"
        }
    }
    compare_all(MODELS, n_episodes=args.episodes)
