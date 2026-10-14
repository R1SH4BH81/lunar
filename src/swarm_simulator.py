"""Autonomous Lunar Swarm Fleet Simulation.

Simulates and renders a simultaneous fleet of 10-15 Apollo Lunar Modules
descending onto Base Alpha in real time:
- PGNCS Solved Aces (Executing precision retro-burns toward LZ-1)
- Hover Exploiters (Loitering cautiously in holding patterns)
- Trainee / Exploratory units (Demonstrating initial learning / tumbling / crashes)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src
from stable_baselines3 import PPO
from src.nasa_renderer import NASALunarRenderer
from src.utils import save_frames_as_gif

class SwarmShip:
    def __init__(self, ship_id: int, name: str, policy_type: str, env: gym.Env, seed: int):
        self.ship_id = ship_id
        self.name = name
        self.policy_type = policy_type
        self.env = env
        self.obs, _ = self.env.reset(seed=seed)
        
        # Gently spread initial horizontal spawn across lunar sky
        spread_offset = (ship_id - 5.5) * 0.12
        self.obs[0] = np.clip(self.obs[0] + spread_offset, -0.85, 0.85)
        # Apply offset to Box2D body if possible
        try:
            self.env.unwrapped.lander.position.x += spread_offset
        except Exception:
            pass
            
        self.done = False
        self.is_landed = False
        self.is_crashed = False
        self.action = 0
        self.score = 0.0
        self.step_count = 0

class LunarSwarmFleet:
    """Orchestrates multi-agent lunar landing fleet."""

    def __init__(
        self,
        n_ships: int = 12,
        baseline_model_path: str = "models/ppo_lunar_baseline.zip",
        hover_model_path: str = "models/ppo_lunar_hover_hack.zip",
        width: int = 800,
        height: int = 600
    ):
        self.n_ships = n_ships
        self.renderer = NASALunarRenderer(width=width, height=height)
        
        # Load available models (force CPU)
        self.baseline_model = PPO.load(baseline_model_path, device="cpu") if os.path.exists(baseline_model_path) else None
        self.hover_model = PPO.load(hover_model_path, device="cpu") if os.path.exists(hover_model_path) else None

        self.ships: List[SwarmShip] = []
        for i in range(n_ships):
            if i < int(n_ships * 0.5):
                ptype = "PGNCS Ace"
                pname = f"LM-{i+1:02d} [ACE]"
            elif i < int(n_ships * 0.75):
                ptype = "Hover Exploit"
                pname = f"LM-{i+1:02d} [HACK]"
            else:
                ptype = "Trainee"
                pname = f"LM-{i+1:02d} [TRAIN]"

            env = gym.make("LunarLander-v3")
            ship = SwarmShip(i + 1, pname, ptype, env, seed=100 + i * 37)
            self.ships.append(ship)

    def step(self) -> Dict[str, int]:
        active_count = 0
        landed_count = 0
        crashed_count = 0

        for ship in self.ships:
            if ship.done:
                if ship.is_landed:
                    landed_count += 1
                elif ship.is_crashed:
                    crashed_count += 1
                ship.action = 0
                continue

            active_count += 1

            # Select action based on policy
            if ship.policy_type == "PGNCS Ace" and self.baseline_model:
                action, _ = self.baseline_model.predict(ship.obs, deterministic=True)
            elif ship.policy_type == "Hover Exploit" and self.hover_model:
                action, _ = self.hover_model.predict(ship.obs, deterministic=True)
            else:
                # Trainee exploratory policy: noisy random actions mixed with main thruster
                if np.random.rand() < 0.35:
                    action = np.random.choice([0, 1, 2, 3])
                elif self.baseline_model:
                    action, _ = self.baseline_model.predict(ship.obs, deterministic=False)
                else:
                    action = np.random.choice([0, 2])

            ship.action = int(action)
            obs, reward, terminated, truncated, info = ship.env.step(action)
            ship.obs = obs
            ship.score += reward
            ship.step_count += 1

            if terminated or truncated:
                ship.done = True
                if ship.score > 100 or (obs[6] == 1.0 and obs[7] == 1.0 and abs(obs[3]) < 0.2):
                    ship.is_landed = True
                    landed_count += 1
                else:
                    ship.is_crashed = True
                    crashed_count += 1

        return {
            "active": active_count,
            "landed": landed_count,
            "crashed": crashed_count
        }

    def close(self):
        for ship in self.ships:
            try:
                ship.env.close()
            except Exception:
                pass

def record_swarm_mission(
    n_ships: int = 12,
    max_steps: int = 360,
    output_gif: str = "recordings/lunar_swarm_fleet.gif"
) -> str:
    print("=" * 65)
    print(f"SIMULATING LUNAR SWARM DESCENT ({n_ships} SHIPS SIMULTANEOUSLY)")
    print("=" * 65)

    fleet = LunarSwarmFleet(n_ships=n_ships)
    frames = []

    for step in range(max_steps):
        stats = fleet.step()

        ship_render_data = []
        for ship in fleet.ships:
            ship_render_data.append({
                "obs": ship.obs,
                "action": ship.action,
                "name": ship.name,
                "is_crashed": ship.is_crashed,
                "scale": 0.72
            })

        frame = fleet.renderer.render_swarm_frame(
            ships=ship_render_data,
            step=step,
            active_count=stats["active"],
            landed_count=stats["landed"],
            crashed_count=stats["crashed"]
        )
        frames.append(frame)

        if stats["active"] == 0 and step > 100:
            print(f"All ships finished descent at step {step}!")
            break

    # Add resting frames
    for _ in range(12):
        frame = fleet.renderer.render_swarm_frame(
            ships=ship_render_data,
            step=step,
            active_count=stats["active"],
            landed_count=stats["landed"],
            crashed_count=stats["crashed"]
        )
        frames.append(frame)

    fleet.close()

    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    save_frames_as_gif(frames, output_gif, fps=30)
    print(f"[Saved] Swarm recording saved to: {output_gif}")
    return output_gif

if __name__ == "__main__":
    record_swarm_mission()
