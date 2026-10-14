"""Generate authentic NASA Apollo 11 Mission Control Lunar Landing video/GIF.

Uses the trained solved PPO policy (models/ppo_lunar_baseline.zip) and renders
the Apollo Lunar Module touchdown at Tranquility Base with Apollo Guidance
Computer (AGC) telemetry.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src
import gymnasium as gym
from stable_baselines3 import PPO
from src.nasa_renderer import NASALunarRenderer
from src.utils import save_frames_as_gif

def record_nasa_mission(
    model_path: str = "models/ppo_lunar_baseline.zip",
    output_gif: str = "recordings/lunar_mission_landing.gif",
    seed: int = 1010  # episode 10 scored 265 points!
):
    print("=" * 65)
    print("RECORDING LUNAR LANDING TELEMETRY (LZ-1 BASE ALPHA)")
    print("=" * 65)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    model = PPO.load(model_path, device="cpu")
    env = gym.make("LunarLander-v3")
    renderer = NASALunarRenderer(width=800, height=600)

    obs, info = env.reset(seed=seed)
    frames = []
    done = False
    step = 0
    cum_score = 0.0

    print("Simulating Apollo descent trajectory...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Render authentic Apollo frame with NASA HUD
        frame = renderer.render_frame(obs, int(action), step, cum_score)
        frames.append(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        cum_score += reward
        step += 1
        done = terminated or truncated

    # Render a few final resting frames showing touchdown confirmed
    for _ in range(15):
        frame = renderer.render_frame(obs, 0, step, cum_score)
        frames.append(frame)

    env.close()

    print(f"Descent completed in {step} frames | Final Mission Score: {cum_score:.2f}")
    save_frames_as_gif(frames, output_gif, fps=30)
    print(f"[Saved] NASA Apollo 11 Landing Simulation: {output_gif}")
    return output_gif

if __name__ == "__main__":
    record_nasa_mission()
