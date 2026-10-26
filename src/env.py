from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DEFAULT_MAX_EPISODE_STEPS = 3_000
DEFAULT_FRAME_STACK_SIZE = 4
EVALUATION_SEEDS = frozenset({202, 42, 203})


class GrayScaleFrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, stack_size: int = DEFAULT_FRAME_STACK_SIZE):
        super().__init__(env)
        if stack_size < 1:
            raise ValueError("stack_size must be at least 1")

        height, width, channels = env.observation_space.shape
        if channels != 3:
            raise ValueError("GrayScaleFrameStack expects RGB observations")

        self.stack_size = stack_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(stack_size, height, width),
            dtype=np.uint8,
        )
        self._frames: np.ndarray | None = None

    @staticmethod
    def _to_grayscale(observation: np.ndarray) -> np.ndarray:
        gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        return np.rint(gray).astype(np.uint8)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        gray = self._to_grayscale(observation)
        self._frames = np.repeat(gray[np.newaxis, ...], self.stack_size, axis=0)
        return self._frames.copy(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._frames is None:
            raise RuntimeError("Call reset() before step()")

        gray = self._to_grayscale(observation)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = gray
        return self._frames.copy(), reward, terminated, truncated, info


class NoProgressTermination(gym.Wrapper):
    """End a training episode after too many steps without a new tile."""

    def __init__(
        self,
        env: gym.Env,
        patience: int = 100,
        penalty: float = 100.0,
    ):
        super().__init__(env)
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if penalty < 0:
            raise ValueError("penalty cannot be negative")
        self.patience = patience
        self.penalty = penalty
        self._steps_without_progress = 0

    def reset(self, **kwargs):
        self._steps_without_progress = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0.0:
            # Visiting a new tile gives positive reward after the time penalty.
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1

        if (
            self._steps_without_progress >= self.patience
            and not terminated
            and not truncated
        ):
            reward -= self.penalty
            terminated = True
            info = dict(info)
            info["no_progress_termination"] = True

        return observation, reward, terminated, truncated, info


class RandomTrackSeed(gym.Wrapper):
    
    def __init__(self, env: gym.Env, excluded_seeds=EVALUATION_SEEDS):
        super().__init__(env)
        self.excluded_seeds = frozenset(excluded_seeds)
        self._seed_rng: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None or self._seed_rng is None:
            self._seed_rng = np.random.default_rng(seed)

        track_seed = int(self._seed_rng.integers(0, np.iinfo(np.int32).max))
        while track_seed in self.excluded_seeds:
            track_seed = int(self._seed_rng.integers(0, np.iinfo(np.int32).max))

        observation, info = self.env.reset(seed=track_seed, options=options)
        info["track_seed"] = track_seed
        return observation, info


def make_env(
    render_mode: str | None = None,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    frame_stack_size: int = DEFAULT_FRAME_STACK_SIZE,
) -> gym.Env:
    env = gym.make(
        "CarRacing-v3",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    return GrayScaleFrameStack(env, stack_size=frame_stack_size)


def smoke_check(seed: int = 202, steps: int = 5) -> None:
    env = make_env()
    try:
        observation, info = env.reset(seed=seed)
        expected_shape = (DEFAULT_FRAME_STACK_SIZE, 96, 96)
        assert observation.shape == expected_shape, observation.shape
        assert observation.dtype == np.uint8, observation.dtype
        print(f"Reset: observation={observation.shape} {observation.dtype}")

        # Straight steering, moderate throttle, and no brake.
        action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        for step in range(steps):
            observation, reward, terminated, truncated, info = env.step(action)
            assert env.action_space.contains(action), action
            assert observation.shape == expected_shape, observation.shape
            print(
                f"Step {step + 1}: reward={reward:.3f}, "
                f"terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    smoke_check(seed=args.seed, steps=args.steps)
