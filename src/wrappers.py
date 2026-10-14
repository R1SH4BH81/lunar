"""Gymnasium wrappers for reward shaping experiments in LunarLander-v3.

Includes:
1. TimePenaltyWrapper: Deducts points per frame to incentivize faster landing.
2. HoverHackingWrapper: Demonstrates reward hacking (rewards hovering, causing agent to avoid landing).
3. PotentialShapedWrapper: Fixes reward hacking using potential-based shaping (Ng et al., 1999).
4. DetailedMetricsWrapper: Logs fuel usage, raw game score, and shaped score.
"""

import gymnasium as gym
import numpy as np


class DetailedMetricsWrapper(gym.Wrapper):
    """Tracks episode metrics: raw environment score, shaped score, fuel use, and outcome."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.raw_episode_reward = 0.0
        self.shaped_episode_reward = 0.0
        self.main_engine_steps = 0
        self.side_engine_steps = 0
        self.episode_steps = 0

    def reset(self, **kwargs):
        self.raw_episode_reward = 0.0
        self.shaped_episode_reward = 0.0
        self.main_engine_steps = 0
        self.side_engine_steps = 0
        self.episode_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.raw_episode_reward += reward
        self.episode_steps += 1

        if action == 2:
            self.main_engine_steps += 1
        elif action in (1, 3):
            self.side_engine_steps += 1

        info["raw_step_reward"] = reward
        info["main_engine_steps"] = self.main_engine_steps
        info["side_engine_steps"] = self.side_engine_steps
        info["episode_steps"] = self.episode_steps
        info["raw_episode_reward"] = self.raw_episode_reward

        # Check landing condition: both legs down and low velocity
        landed_safely = (obs[6] == 1.0 and obs[7] == 1.0 and abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2)
        info["landed_safely"] = landed_safely

        return obs, reward, terminated, truncated, info


class TimePenaltyWrapper(gym.RewardWrapper):
    """Penalizes each frame by a small cost to discourage stalling and encourage prompt landing."""

    def __init__(self, env: gym.Env, frame_penalty: float = 0.1):
        super().__init__(env)
        self.frame_penalty = frame_penalty

    def reward(self, reward: float) -> float:
        return reward - self.frame_penalty


class HoverHackingWrapper(gym.RewardWrapper):
    """Demonstrates reward hacking.

    Adds a static +1.0 per frame bonus whenever the lander is upright and slow.
    Because an episode maxes at 1,000 frames, hovering in midair yields up to 1,000
    points—far higher than the +200-300 points for a real landing.
    The agent learns that landing immediately ENDS the attempt and cuts off payments,
    so it chooses to hover forever.
    """

    def __init__(self, env: gym.Env, upright_angle_thresh: float = 0.25, vel_y_thresh: float = 0.35, bonus: float = 1.0):
        super().__init__(env)
        self.upright_angle_thresh = upright_angle_thresh
        self.vel_y_thresh = vel_y_thresh
        self.bonus = bonus
        self.current_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_obs = obs

        # Calculate reward hack bonus
        angle = abs(obs[4])
        vel_y = abs(obs[3])
        if angle < self.upright_angle_thresh and vel_y < self.vel_y_thresh:
            hacked_reward = reward + self.bonus
            info["reward_hacked_bonus"] = self.bonus
        else:
            hacked_reward = reward
            info["reward_hacked_bonus"] = 0.0

        return obs, hacked_reward, terminated, truncated, info


class PotentialShapedWrapper(gym.Wrapper):
    """Fixes reward hacking using Potential-Based Reward Shaping (PBRS).

    Instead of rewarding a static condition each frame (which encourages hovering),
    we compute the CHANGE in potential between frames:
        F = gamma * Phi(s') - Phi(s) - frame_cost
    This guarantees policy invariance (optimal trajectory remains landing), rewards
    descent progress, and charges a small per-frame fee so hovering loses points.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99, frame_penalty: float = 0.15):
        super().__init__(env)
        self.gamma = gamma
        self.frame_penalty = frame_penalty
        self.prev_potential = 0.0

    def _compute_potential(self, obs: np.ndarray) -> float:
        # Distance to center pad
        dist = np.sqrt(obs[0] ** 2 + obs[1] ** 2)
        speed = np.sqrt(obs[2] ** 2 + obs[3] ** 2)
        angle = abs(obs[4])

        # Potential increases as rocket gets closer to pad, slower, and upright
        # Near ground (obs[1] < 0.2), reward legs contact
        legs = (obs[6] + obs[7]) * 10.0
        potential = - (50.0 * dist + 40.0 * speed + 30.0 * angle) + legs
        return potential

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self._compute_potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_potential = self._compute_potential(obs)

        # Potential difference (PBRS)
        shaping = self.gamma * curr_potential - self.prev_potential
        shaped_reward = reward + shaping - self.frame_penalty

        self.prev_potential = curr_potential
        info["shaping_term"] = shaping
        info["potential"] = curr_potential

        return obs, shaped_reward, terminated, truncated, info
