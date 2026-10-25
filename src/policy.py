"""CNN feature extractor used by the Stable-Baselines3 PPO policy."""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CarRacingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) != 3:
            raise ValueError(
                "Expected channel-first image observations with three dimensions; "
                f"got {observation_space.shape}"
            )

        channels, height, width = observation_space.shape
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            flattened_size = self.cnn(
                th.zeros(1, channels, height, width)
            ).shape[1]

        # This is the dense feature network shared by PPO's actor and critic.
        # It must output features_dim values, not the three CarRacing actions:
        # SB3 constructs and trains the action/value heads after this extractor.
        self.policy_net = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        visual_features = self.cnn(observations)
        return self.policy_net(visual_features)
