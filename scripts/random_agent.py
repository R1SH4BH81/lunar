"""Watch it Fail: Baseline script running random actions in LunarLander-v3.

Prints detailed diagnostics of the 8 observation numbers and records the
chaotic tumbling and inevitable crash as a GIF/video.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
from src.nasa_renderer import NASALunarRenderer
from src.utils import (
    decode_observation,
    format_observation_table,
    save_frames_as_gif,
    save_frames_as_mp4,
    ACTION_NAMES
)

def run_random_agent(max_steps: int = 400, output_gif: str = "recordings/random_agent_crash.gif"):
    print("=" * 60)
    print("STEP 2: WATCH IT FAIL - RANDOM ACTIONS BASELINE")
    print("=" * 60)
    
    env = gym.make("LunarLander-v3")
    renderer = NASALunarRenderer(width=800, height=600)
    
    obs, info = env.reset(seed=42)
    frames = [renderer.render_frame(obs, action=0, step=0, score=0.0)]
    
    print("\nInitial Observation State (Spawn):")
    print(format_observation_table(obs))
    
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    
    print("\nSimulating random flight:")
    print("Frame | Action Taken        | Height (y) | Tilt (rad) | Step Rew | Total Rew")
    print("-" * 72)
    
    while not (terminated or truncated) and step_count < max_steps:
        # Sample random action: 0=Nothing, 1=Left Engine, 2=Main Engine, 3=Right Engine
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        is_crashed = terminated and total_reward < -50
        frame = renderer.render_frame(
            obs,
            action=action,
            step=step_count,
            score=total_reward,
            is_crashed=is_crashed
        )
        frames.append(frame)
        
        # Print progress every 15 frames or when done
        if step_count % 15 == 0 or terminated or truncated:
            action_str = f"[{action}] {ACTION_NAMES[action]:<17}"
            print(f"{step_count:>5} | {action_str} | {obs[1]:>10.3f} | {obs[4]:>10.3f} | {reward:>8.2f} | {total_reward:>9.2f}")
    
    # If crashed, add a few resting frames showing the crash state
    if terminated and total_reward < 0:
        for _ in range(12):
            frame = renderer.render_frame(obs, action=0, step=step_count, score=total_reward, is_crashed=True)
            frames.append(frame)

    env.close()
    
    print("\n" + "=" * 60)
    outcome = "CRASHED (-100 pts)!" if terminated and total_reward < 0 else "TIMED OUT / TERMINATED"
    print(f"Episode finished in {step_count} frames.")
    print(f"Outcome: {outcome}")
    print(f"Cumulative Episode Score: {total_reward:.2f}")
    print("Final State Vector:")
    print(format_observation_table(obs, reward=reward))
    print("=" * 60)
    
    # Save recording
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    save_frames_as_gif(frames, output_gif, fps=30)
    print(f"\n[Saved] Visual GIF recording saved to: {output_gif}")
    
    return total_reward, step_count

if __name__ == "__main__":
    run_random_agent()
