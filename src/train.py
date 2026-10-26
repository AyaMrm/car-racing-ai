from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env import NoProgressTermination, RandomTrackSeed, make_env
from src.policy import CarRacingCNN, CarRacingCNNLegacy


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The PPO configuration must contain a YAML mapping")
    return config


def get_features_extractor_class(network_config: dict):
    name = network_config.get("features_extractor", "CarRacingCNN")
    extractors = {
        "CarRacingCNN": CarRacingCNN,
        "CarRacingCNNLegacy": CarRacingCNNLegacy,
    }
    try:
        return extractors[name]
    except KeyError as exc:
        raise ValueError(f"Unknown features_extractor: {name}") from exc


def make_training_env(config: dict, index: int):
    env_config = config["environment"]
    log_dir = Path(config["paths"]["logs"])

    def _make():
        env = make_env(
            max_episode_steps=env_config["max_episode_steps"],
            frame_stack_size=env_config["frame_stack_size"],
        )
        progress_patience = int(env_config.get("no_progress_patience", 0))
        if progress_patience > 0:
            env = NoProgressTermination(
                env,
                patience=progress_patience,
                penalty=float(env_config.get("no_progress_penalty", 100.0)),
            )
        env = RandomTrackSeed(env)
        return Monitor(env, filename=str(log_dir / f"env_{index}"))

    return _make


def train(
    config_path: str | Path = "configs/ppo.yaml",
    resume_from: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    env_config = config["environment"]
    training_config = config["training"]
    ppo_config = config["ppo"]
    network_config = config["network"]
    features_extractor_class = get_features_extractor_class(network_config)
    paths = config["paths"]

    n_envs = int(env_config["n_envs"])
    rollout_size = n_envs * int(ppo_config["n_steps"])
    if rollout_size % int(ppo_config["batch_size"]) != 0:
        raise ValueError(
            "n_envs * n_steps must be divisible by batch_size; "
            f"got {rollout_size} and {ppo_config['batch_size']}"
        )

    checkpoint_dir = Path(paths["checkpoints"])
    log_dir = Path(paths["logs"])
    final_model_path = Path(paths["final_model"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv(
        [make_training_env(config, index) for index in range(n_envs)]
    )

    checkpoint_frequency = max(
        int(training_config["checkpoint_frequency"]) // n_envs, 1
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_car_racing",
    )

    if resume_from is not None:
        if "additional_timesteps" not in training_config:
            raise ValueError(
                "A resume config must define training.additional_timesteps"
            )
        total_timesteps = int(training_config["additional_timesteps"])
        load_kwargs = {}
        if features_extractor_class is CarRacingCNNLegacy:
            # The legacy archive stores CarRacingCNN as its extractor class.
            # Override that class before SB3 reconstructs the saved policy so
            # the old projection-layer parameters load into the legacy layout.
            load_kwargs["custom_objects"] = {
                "policy_kwargs": {
                    "features_extractor_class": CarRacingCNNLegacy,
                    "features_extractor_kwargs": {
                        "features_dim": int(network_config["features_dim"])
                    },
                    "net_arch": {
                        "pi": network_config["policy_hidden_layers"],
                        "vf": network_config["value_hidden_layers"],
                    },
                }
            }
        model = PPO.load(
            str(resume_from), env=env, device="auto", **load_kwargs
        )
        # A resumed model keeps its learned weights but uses the current
        # experiment's optimizer schedule and exploration settings.
        learning_rate = float(ppo_config["learning_rate"])
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        for parameter_group in model.policy.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        model.ent_coef = float(ppo_config["ent_coef"])
    else:
        total_timesteps = int(training_config["total_timesteps"])
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=float(ppo_config["learning_rate"]),
            n_steps=int(ppo_config["n_steps"]),
            batch_size=int(ppo_config["batch_size"]),
            n_epochs=int(ppo_config["n_epochs"]),
            gamma=float(ppo_config["gamma"]),
            gae_lambda=float(ppo_config["gae_lambda"]),
            clip_range=float(ppo_config["clip_range"]),
            ent_coef=float(ppo_config["ent_coef"]),
            vf_coef=float(ppo_config["vf_coef"]),
            max_grad_norm=float(ppo_config["max_grad_norm"]),
            seed=int(training_config["seed"]),
            policy_kwargs={
                "features_extractor_class": features_extractor_class,
                "features_extractor_kwargs": {
                    "features_dim": int(network_config["features_dim"])
                },
                "net_arch": {
                    "pi": network_config["policy_hidden_layers"],
                    "vf": network_config["value_hidden_layers"],
                },
            },
            verbose=1,
            device="auto",
        )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=resume_from is None,
        )
        model.save(str(final_model_path))
    finally:
        env.close()

    return final_model_path.with_suffix(".zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument(
        "--resume",
        help="Continue training from an SB3 .zip checkpoint",
    )
    args = parser.parse_args()
    saved_path = train(args.config, resume_from=args.resume)
    print(f"Training finished. Saved model: {saved_path}")
