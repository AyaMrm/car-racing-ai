"""Export a trained SB3 PPO policy as a raw-image inference module."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import Tensor, nn

from src.policy import CarRacingCNN


class CarRacingInference(nn.Module): 

    def __init__(
        self,
        feature_extractor: CarRacingCNN,
        action_head: nn.Linear,
    ) -> None:
        super().__init__()
        self.cnn = feature_extractor.cnn
        self.policy_net = feature_extractor.policy_net
        self.action_head = action_head
        self.register_buffer(
            "frame_stack", torch.zeros((4, 96, 96), dtype=torch.uint8)
        )
        self.frames_initialized = False
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.jit.export
    def reset_state(self) -> None:
        self.frame_stack.zero_()
        self.frames_initialized = False

    def forward(self, observation: Tensor) -> Tensor:
        if observation.dim() != 3 or observation.shape[2] != 3:
            raise ValueError("Expected one RGB frame tensor shaped (96, 96, 3)")
        if observation.shape[0] != 96 or observation.shape[1] != 96:
            raise ValueError("Expected 96x96 CarRacing observations")

        rgb = observation.to(dtype=torch.float32)
        gray_float = (
            rgb[:, :, 0] * 0.299
            + rgb[:, :, 1] * 0.587
            + rgb[:, :, 2] * 0.114
        )
        gray = torch.round(gray_float).to(dtype=torch.uint8)

        if not self.frames_initialized:
            self.frame_stack = gray.unsqueeze(0).repeat(4, 1, 1)
            self.frames_initialized = True
        else:
            self.frame_stack = torch.cat(
                (self.frame_stack[1:], gray.unsqueeze(0)), dim=0
            )

        stacked_image = self.frame_stack.unsqueeze(0).to(dtype=torch.float32) / 255.0
        features = self.policy_net(self.cnn(stacked_image))
        action_mean = self.action_head(features).squeeze(0)

        lower = torch.tensor([-1.0, 0.0, 0.0], device=action_mean.device)
        upper = torch.tensor([1.0, 1.0, 1.0], device=action_mean.device)
        return torch.maximum(torch.minimum(action_mean, upper), lower)


def build_inference_model(checkpoint_path: str | Path) -> CarRacingInference:
    model = PPO.load(str(checkpoint_path), device="cpu")
    extractor = model.policy.features_extractor
    if not isinstance(extractor, CarRacingCNN):
        raise TypeError(
            "The checkpoint does not use CarRacingCNN; cannot export this policy"
        )
    if not isinstance(model.policy.action_net, nn.Linear):
        raise TypeError("Expected PPO's actor output head to be a linear layer")

    inference_model = CarRacingInference(
        feature_extractor=extractor,
        action_head=model.policy.action_net,
    )
    inference_model.eval()
    return inference_model


def export_model(
    checkpoint_path: str | Path,
    output_path: str | Path = "submission/model.pt",
) -> Path:
    inference_model = build_inference_model(checkpoint_path)
    scripted_model = torch.jit.script(inference_model)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted_model, str(destination))
    return destination


def predict_numpy(model: nn.Module, observation: np.ndarray) -> np.ndarray:
    image = torch.as_tensor(observation, dtype=torch.uint8)
    with torch.inference_mode():
        action = model(image)
    return action.cpu().numpy().astype(np.float32, copy=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to the trained SB3 PPO .zip file")
    parser.add_argument("--output", default="submission/model.pt")
    args = parser.parse_args()
    path = export_model(args.checkpoint, args.output)
    print(f"Exported inference model: {path}")
