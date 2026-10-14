"""Evaluation and Recording Script for LunarLander-v3 Models.

Runs evaluation episodes on pure unshaped LunarLander-v3 (or wrapped),
computes comprehensive statistics (score, solved rate, crash rate, fuel use, duration),
and records visual GIF/MP4 videos of flight trajectories.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src
import gymnasium as gym
from stable_baselines3 import PPO

from src.nasa_renderer import NASALunarRenderer
from src.utils import save_frames_as_gif, save_frames_as_mp4, format_observation_table
from src.wrappers import DetailedMetricsWrapper, HoverHackingWrapper, PotentialShapedWrapper, TimePenaltyWrapper

def evaluate_model(
    model_path: str,
    n_episodes: int = 15,
    render_video: bool = True,
    output_media_path: str = "recordings/eval_landing.gif",
    wrapper_mode: str = "none",
    deterministic: bool = True
):
    print("=" * 65)
    print(f"EVALUATION: {os.path.basename(model_path)}")
    print(f"Episodes: {n_episodes} | Evaluated in Wrapper Mode: [{wrapper_mode.upper()}]")
    print("=" * 65)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = PPO.load(model_path)
    renderer = NASALunarRenderer(width=800, height=600) if render_video else None

    # Base env
    def make_eval_env():
        env = gym.make("LunarLander-v3")
        if wrapper_mode == "hover_hack":
            env = HoverHackingWrapper(env, upright_angle_thresh=0.25, vel_y_thresh=0.35, bonus=1.0)
        elif wrapper_mode == "potential_shaped":
            env = PotentialShapedWrapper(env)
        elif wrapper_mode == "time_penalty":
            env = TimePenaltyWrapper(env, frame_penalty=0.15)
        env = DetailedMetricsWrapper(env)
        return env

    env = make_eval_env()

    scores = []
    lengths = []
    main_fuels = []
    side_fuels = []
    landed_count = 0
    crashed_count = 0
    solved_count = 0

    best_frames = []
    best_score = -float("inf")

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=1000 + ep)
        if renderer:
            renderer.reset()
        done = False
        ep_score = 0.0
        # Record frames for the first few episodes to find the best in-frame trajectory
        record_this = render_video and (ep <= 3 or not best_frames)
        ep_frames = [renderer.render_frame(obs, action=0, step=0, score=0.0)] if record_this else []

        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_score += reward
            if record_this:
                is_crash = terminated and not truncated and ep_score < 0
                frame = renderer.render_frame(
                    obs,
                    action=int(action),
                    step=info["episode_steps"],
                    score=ep_score,
                    is_crashed=is_crash
                )
                ep_frames.append(frame)
            done = terminated or truncated

        # Add brief resting frames showing final state (Touchdown or Crash)
        if record_this and len(ep_frames) > 0:
            is_crash = terminated and not truncated and ep_score < 0
            for _ in range(12):
                ep_frames.append(renderer.render_frame(
                    obs,
                    action=0,
                    step=info["episode_steps"],
                    score=ep_score,
                    is_crashed=is_crash
                ))

        scores.append(ep_score)
        lengths.append(info["episode_steps"])
        main_fuels.append(info["main_engine_steps"])
        side_fuels.append(info["side_engine_steps"])

        is_crash = terminated and not truncated and ep_score < 0
        is_timeout = truncated and not info["landed_safely"]

        if ep_score >= 200.0:
            solved_count += 1

        if info["landed_safely"]:
            landed_count += 1
        elif is_crash:
            crashed_count += 1

        if record_this and ep_score > best_score:
            best_score = ep_score
            best_frames = ep_frames

        if ep_score >= 200.0:
            status = "SOLVED (>=200)"
        elif info["landed_safely"]:
            status = "LANDED SAFELY"
        elif is_crash:
            status = "CRASHED"
        elif is_timeout:
            status = "TIMED OUT (HOVERING)"
        else:
            status = "COMPLETED"

        print(f"Episode {ep:>2}/{n_episodes} | Score: {ep_score:>7.2f} | Steps: {info['episode_steps']:>4} | Main Eng: {info['main_engine_steps']:>3} | Status: {status}", flush=True)

    env.close()

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_length = np.mean(lengths)
    mean_main_fuel = np.mean(main_fuels)
    mean_side_fuel = np.mean(side_fuels)
    solved_pct = (solved_count / n_episodes) * 100.0
    landed_pct = (landed_count / n_episodes) * 100.0
    crashed_pct = (crashed_count / n_episodes) * 100.0

    print("-" * 65)
    print("EVALUATION SUMMARY:")
    print(f"Mean Score: {mean_score:.2f} +/- {std_score:.2f} (Target Solved: >= 200)")
    print(f"Solved Rate (Score >= 200): {solved_pct:.1f}% ({solved_count}/{n_episodes})")
    print(f"Safe Landing Rate:          {landed_pct:.1f}% ({landed_count}/{n_episodes})")
    print(f"Crash Rate:                 {crashed_pct:.1f}% ({crashed_count}/{n_episodes})")
    print(f"Average Flight Duration:    {mean_length:.1f} frames")
    print(f"Average Fuel Usage:         Main: {mean_main_fuel:.1f} ticks, Side: {mean_side_fuel:.1f} ticks")
    print("=" * 65)

    if render_video and len(best_frames) > 0:
        os.makedirs(os.path.dirname(output_media_path), exist_ok=True)
        save_frames_as_gif(best_frames, output_media_path, fps=30)
        print(f"[Saved] Flight recording GIF (best attempt, score={best_score:.2f}): {output_media_path}")

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "solved_pct": solved_pct,
        "landed_pct": landed_pct,
        "crashed_pct": crashed_pct,
        "mean_length": mean_length,
        "mean_main_fuel": mean_main_fuel,
        "mean_side_fuel": mean_side_fuel
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LunarLander-v3 policy")
    parser.add_argument("--model", type=str, default="models/ppo_lunar_baseline.zip", help="Path to model zip")
    parser.add_argument("--episodes", type=int, default=15, help="Number of test episodes")
    parser.add_argument("--gif", type=str, default="recordings/eval_landing.gif", help="Output GIF path")
    parser.add_argument("--wrapper", type=str, default="none", choices=["none", "hover_hack", "potential_shaped", "time_penalty"])
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        output_media_path=args.gif,
        wrapper_mode=args.wrapper
    )
