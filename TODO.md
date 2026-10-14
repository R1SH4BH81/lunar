# Lunar Lander Reinforcement Learning & Reward Shaping: Project Plan

## Overview
This project trains an AI to safely land a rocket in Gymnasium's `LunarLander-v3` using Proximal Policy Optimization (PPO) with Stable-Baselines3, explores reward shaping mechanisms, demonstrates reward hacking (hovering behavior), and fixes it with potential-based reward shaping.

---

## Phase 1: Environment Setup & Verification
- [x] **1.1** Set up Python virtual environment / dependencies
- [x] **1.2** Install dependencies: `gymnasium[box2d]`, `stable-baselines3`, `torch`, `imageio`, `moviepy`, `matplotlib`, `pandas`
- [x] **1.3** Verify Box2D and PyTorch setup (verified `gymnasium.make("LunarLander-v3")`)

## Phase 2: Observation & Random Agent Baseline ("Watch it Fail")
- [x] **2.1** Create `scripts/random_agent.py` to run random actions
- [x] **2.2** Log and explain the 8-dimensional observation vector per frame
- [x] **2.3** Record and save video/GIF of the tumbling/crashing random lander (`recordings/random_agent_crash.gif`)

## Phase 3: Train Standard PPO Agent ("Train & Land")
- [x] **3.1** Create modular training pipeline (`train.py`) using PPO with `MlpPolicy`
- [x] **3.2** Train baseline model for 300k timesteps
- [x] **3.3** Save trained baseline model to `models/ppo_lunar_baseline.zip`
- [x] **3.4** Create evaluation & video recording script (`evaluate.py`) to record flight GIFs

## Phase 4: Custom Reward Shaping Experiments ("Break It & Fix It")
- [x] **4.1** Implement Gymnasium wrapper framework in `src/wrappers.py`
- [x] **4.2** **Experiment 1 (Time Efficiency Shaping):** Add per-frame cost to penalize hovering/stalling and encourage faster landings
- [x] **4.3** **Experiment 2 (Reward Hacking):** Add positive reward (+1/frame) for upright posture and low velocity; observe agent learning to hover indefinitely rather than landing
- [x] **4.4** **Experiment 3 (The Fix - Potential-Based Shaping):** Reward relative improvements in potential function plus terminal rewards, preventing perpetual hovering while retaining smooth descent
- [x] **4.5** Train models for each experiment and record comparative GIFs/videos

## Phase 5: Visual Dashboard & Interactive Analysis
- [x] **5.1** Generate comparison plots: learning curves, episode length vs fuel usage, cumulative reward breakdowns (`recordings/comparison_chart.png`)
- [x] **5.2** Provide interactive viewer / runner CLI to replay any trained policy (`app.py`)
- [x] **5.3** Comprehensive documentation and guide with resume/portfolio-ready summary (`README.md`)
