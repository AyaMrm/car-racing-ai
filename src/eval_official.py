from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


TRACK_TILE_COUNTS = {202: 276, 42: 283, 203: 360}


def check_model(
    model_path: str | Path,
    seed: int = 202,
    steps: int = 20,
    brake_scale: float = 1.0,
    min_gas: float = 0.0,
    render: bool = False,
    trace_every: int = 0,
) -> None:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0.0 <= brake_scale <= 1.0:
        raise ValueError("brake_scale must be between 0 and 1")
    if not 0.0 <= min_gas <= 1.0:
        raise ValueError("min_gas must be between 0 and 1")
    if trace_every < 0:
        raise ValueError("trace_every cannot be negative")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    if hasattr(model, "reset_state"):
        model.reset_state()

    env = gym.make(
        "CarRacing-v3",
        max_episode_steps=3_000,
        render_mode="human" if render else None,
    )
    total_reward = 0.0
    model_actions = []
    executed_actions = []
    terminated = False
    try:
        observation, _ = env.reset(seed=seed)
        with torch.inference_mode():
            for step in range(steps):
                image = torch.as_tensor(observation, dtype=torch.uint8)
                action_tensor = model(image)
                if action_tensor.shape != (3,):
                    raise ValueError(
                        f"Expected action shape (3,), got {tuple(action_tensor.shape)}"
                    )

                model_action = action_tensor.cpu().numpy().astype(
                    np.float32, copy=True
                )
                action = model_action.copy()
                action[2] *= brake_scale
                action[1] = max(float(action[1]), min_gas)
                if not env.action_space.contains(action):
                    raise ValueError(f"Action is outside the action space: {action}")
                model_actions.append(model_action)
                executed_actions.append(action)

                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if trace_every and (
                    (step + 1) % trace_every == 0 or step == 0
                ):
                    print(
                        f"Trace step {step + 1}: "
                        f"model_action={model_action}, executed_action={action}, "
                        f"reward={reward:.3f}, cumulative_reward={total_reward:.3f}"
                    )
                if terminated or truncated:
                    print(
                        f"Episode ended at step {step + 1}: "
                        f"terminated={terminated}, truncated={truncated}"
                    )
                    break

        print(f"Raw observation shape: {observation.shape}")
        print(f"Actions produced: {len(executed_actions)}")
        print(f"First model action [steering, gas, brake]: {model_actions[0]}")
        print(f"First executed action [steering, gas, brake]: {executed_actions[0]}")
        model_action_array = np.asarray(model_actions)
        executed_action_array = np.asarray(executed_actions)
        mean_model_action = model_action_array.mean(axis=0)
        mean_executed_action = executed_action_array.mean(axis=0)
        print(
            "Mean model action [steering, gas, brake]: "
            f"[{mean_model_action[0]:.3f}, {mean_model_action[1]:.3f}, "
            f"{mean_model_action[2]:.3f}]"
        )
        print(
            "Mean executed action [steering, gas, brake]: "
            f"[{mean_executed_action[0]:.3f}, {mean_executed_action[1]:.3f}, "
            f"{mean_executed_action[2]:.3f}]"
        )
        model_high_brake_steps = int(
            np.count_nonzero(model_action_array[:, 2] > 0.5)
        )
        high_brake_steps = int(
            np.count_nonzero(executed_action_array[:, 2] > 0.5)
        )
        both_high_steps = int(
            np.count_nonzero(
                (executed_action_array[:, 1] > 0.5)
                & (executed_action_array[:, 2] > 0.5)
            )
        )
        print(
            f"Model brake > 0.5: {model_high_brake_steps}/"
            f"{len(model_actions)}"
        )
        print(
            f"Executed brake > 0.5: {high_brake_steps}/"
            f"{len(executed_actions)} "
            f"({100 * high_brake_steps / len(executed_actions):.1f}%)"
        )
        print(
            "Steps with both gas and brake > 0.5: "
            f"{both_high_steps}/{len(executed_actions)}"
        )
        print(f"Total reward over checked steps: {total_reward:.3f}")
        print(f"Brake scale applied: {brake_scale:.2f}")
        print(f"Minimum gas applied: {min_gas:.2f}")
        tile_count = TRACK_TILE_COUNTS.get(seed)
        if tile_count is not None and not terminated:
            estimated_tiles = (
                (total_reward + 0.1 * len(executed_actions))
                * tile_count
                / 1000.0
            )
            print(
                f"Approx. newly visited tiles from reward: {estimated_tiles:.1f} "
                f"of {tile_count} (rough estimate)"
            )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exported TorchScript model")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--brake-scale",
        type=float,
        default=1.0,
        help="Multiply the model's brake output by a factor in [0, 1]",
    )
    parser.add_argument(
        "--min-gas",
        type=float,
        default=0.0,
        help="Set a diagnostic lower bound for gas in [0, 1]",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the CarRacing window during this evaluation",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help="Print action and reward details every N steps (0 disables tracing)",
    )
    args = parser.parse_args()
    check_model(
        args.model,
        seed=args.seed,
        steps=args.steps,
        brake_scale=args.brake_scale,
        min_gas=args.min_gas,
        render=args.render,
        trace_every=args.trace_every,
    )
