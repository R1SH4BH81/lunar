"""Lunar Lander Reinforcement Learning & Reward Shaping Package."""

import sys

# Prevent broken system protobuf/tensorflow/tensorboard conflicts from interrupting PyTorch & SB3
sys.modules["tensorboard"] = None
sys.modules["torch.utils.tensorboard"] = None
