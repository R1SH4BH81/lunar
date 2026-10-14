"""Utility functions for observation logging, action formatting, and media recording."""

import os
import numpy as np
import imageio
from typing import List, Dict, Any, Optional

OBSERVATION_KEYS = [
    "pos_x",          # 0: left / right of pad (0.0 = center)
    "pos_y",          # 1: height above pad (0.0 = ground)
    "vel_x",          # 2: horizontal velocity (negative = moving left)
    "vel_y",          # 3: vertical velocity (negative = falling)
    "angle",          # 4: tilt angle in radians (0.0 = upright)
    "angular_vel",    # 5: rotational velocity (spin)
    "left_leg_down",  # 6: left leg contact (1.0 = yes, 0.0 = no)
    "right_leg_down"  # 7: right leg contact (1.0 = yes, 0.0 = no)
]

OBSERVATION_DESCRIPTIONS = [
    "Left/Right of pad (x)",
    "Height above pad (y)",
    "Horizontal velocity (vx)",
    "Vertical velocity (vy)",
    "Tilt angle (rad)",
    "Angular velocity (rad/s)",
    "Left leg contact",
    "Right leg contact"
]

ACTION_NAMES = {
    0: "Do Nothing",
    1: "Fire Left Engine",
    2: "Fire Main Engine",
    3: "Fire Right Engine"
}


def decode_observation(obs: np.ndarray) -> Dict[str, float]:
    """Convert an 8-element observation array into a named dictionary."""
    return {k: float(obs[i]) for i, k in enumerate(OBSERVATION_KEYS)}


def format_observation_table(obs: np.ndarray, action: Optional[int] = None, reward: Optional[float] = None) -> str:
    """Format the observation as a readable diagnostic string."""
    lines = ["+---------------------------+--------------+"]
    lines.append("| Observation Variable      | Value        |")
    lines.append("+---------------------------+--------------+")
    for desc, val in zip(OBSERVATION_DESCRIPTIONS, obs):
        lines.append(f"| {desc:<25} | {val:>12.4f} |")
    lines.append("+---------------------------+--------------+")
    if action is not None:
        lines.append(f"Action Taken: [{action}] {ACTION_NAMES.get(action, 'Unknown')}")
    if reward is not None:
        lines.append(f"Reward Received: {reward:+.4f}")
    return "\n".join(lines)


def save_frames_as_gif(frames: List[np.ndarray], output_path: str, fps: int = 30) -> str:
    """Save a sequence of RGB numpy frames as an animated GIF."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # If episode is long, subsample frames to keep GIF smooth and file size optimal
    step = 2 if len(frames) > 300 else 1
    selected_frames = frames[::step]
    effective_fps = max(10, int(fps / step))
    duration = 1000.0 / effective_fps
    imageio.mimsave(output_path, selected_frames, format="GIF", duration=duration, loop=0)
    return output_path


def save_frames_as_mp4(frames: List[np.ndarray], output_path: str, fps: int = 50) -> str:
    """Save a sequence of RGB numpy frames as an MP4 video."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return output_path
