# Car Racing AI

A PPO agent for Gymnasium's `CarRacing-v3`. The agent receives four stacked grayscale frames and predicts continuous steering, gas, and brake actions. Its visual feature network uses three convolutional layers followed by a four-layer `policy_net`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Check the environment

```powershell
python -m src.env
```

## Train

The four-layer, 900k-step experiment starts from scratch:

```powershell
python -m src.train --config configs/ppo_policy_4layer_900k.yaml
```

Training logs and checkpoints are written under `logs/` and `checkpoints/`. These generated folders are excluded from Git.

## Export and evaluate

After training, export the Stable-Baselines3 checkpoint as a TorchScript model:

```powershell
python -m src.export checkpoints/ppo_car_racing_policy_net_4layer_900k.zip --output submission/model.pt
```

Evaluate the exported model on a reserved track seed:

```powershell
python -m src.eval_official --model submission/model.pt --seed 202 --steps 3000
```

Use seeds `202`, `42`, and `203` for the challenge evaluations. Add `--render` to watch the car while evaluation runs.

## Project files

- `configs/`: PPO and experiment settings.
- `src/env.py`: CarRacing environment and grayscale frame stacking.
- `src/policy.py`: CNN feature extractor and PPO feature networks.
- `src/train.py`: training and checkpoint saving.
- `src/export.py`: exports a standalone inference model.
- `src/evaluate.py`, `src/eval_official.py`: development and challenge-style evaluation.
- `submission/`: exported model for submission.
- `record/`: contains a video recording of the car running: [`Recording 2026-10-27 211808.mp4`](record/Recording%202026-10-27%20211808.mp4).
- `journal.md`: development notes, experiments, and results.

See [`carracing_ai_challenge_guidelines.md`](carracing_ai_challenge_guidelines.md) for the challenge requirements.
