# Development Journal

## Day 1 — Project architecture

I chose a small, modular structure so that configuration, environment setup, model code, training, and evaluation each have a clear responsibility.

```text
car-racing-ai/
├── configs/
│   └── ppo.yaml
├── src/
│   ├── env.py
│   ├── policy.py
│   ├── train.py
│   ├── evaluate.py
│   ├── eval_official.py
│   ├── export.py
│   └── utils.py
├── submission/
│   └── model.pt
├── checkpoints/
├── logs/
├── journal.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Day 1 — Environment setup

I implemented `src/env.py` as the first working component. Its `make_env()` function creates `CarRacing-v3` with a 3,000-step episode limit, matching the challenge instructions. It accepts an optional render mode and otherwise keeps the environment headless for training.

The module also provides a small smoke check that can be run with `python -m src.env`. It resets the environment with seed 202, checks the preprocessed observation format, and takes a few valid continuous actions of the form `[steering, gas, brake]`. The sample action drives straight with moderate throttle and no braking. This check is intended to confirm the environment interface before starting PPO; it does not train an agent.

The environment is closed in a `finally` block so it is released even if a check fails. Reward changes are not applied; the environment keeps its default reward.


I added the `GrayScaleFrameStack` wrapper to `src/env.py`. It converts each RGB frame to grayscale using standard luminance weights and keeps a rolling stack of the four most recent frames. The resulting observation has shape `(4, 96, 96)`: the first dimension represents time, so the policy can infer basic motion from changes between frames.

At the start of each episode, the first grayscale frame is copied into all four stack slots. On each subsequent step, the oldest frame is dropped and the newest frame is appended. The wrapper returns `uint8` pixel values in the range 0–255. Stable-Baselines3's default image preprocessing scales these image values to 0–1 for the policy, so the wrapper does not convert them to floating point or normalize them a second time.

I have not added frame skipping or cropped the bottom indicator area yet. Keeping every frame and the full image gives us a simpler baseline; those changes can be evaluated later if needed. The smoke check in `src/env.py` now checks for the preprocessed `(4, 96, 96)` observation format and still takes a few continuous actions. It has not yet been run after this change.


## Day 2 : 

### The CNN policy

I added a custom CNN feature extractor in `src/policy.py`. It receives the four grayscale frames as an image tensor shaped `(4, 96, 96)`. The three convolution layers use 32, 64, and 64 filters. The first layer looks at larger image patches and moves across the image in larger steps; the following layers combine smaller visual details into more useful patterns. ReLU activations let the network learn non-linear patterns. The output is flattened and passed through one dense layer with 256 units. This produces a compact visual summary for PPO.

Stable-Baselines3's PPO policy then uses this summary in two small output branches: an actor that chooses steering, gas, and brake, and a critic that estimates how promising the current situation is. The actor uses a continuous action distribution, matching CarRacing's action space. SB3 scales the `uint8` image pixels from 0-255 to 0-1 before the CNN processes them.

### PPO settings

`configs/ppo.yaml` keeps the first training settings in one place. The initial run is set to one million environment steps with four environments, each limited to 3,000 steps per episode. PPO collects 1,024 steps from each environment before each update, giving a rollout of 4,096 transitions. It divides that rollout into batches of 256 and makes 10 optimization passes over the collected data. The learning rate controls the size of each update; `gamma` discounts future rewards; `gae_lambda` helps estimate advantages; and `clip_range` limits how much the policy can change in one PPO update. These are starting values to measure, not a claim that they are optimal.

The CNN produces 256 features. The actor and critic have no additional hidden layers in the first version, keeping the network small and making the design easier to explain. The final action and value layers are still created by Stable-Baselines3.

### Random training tracks and saving

`RandomTrackSeed` chooses a new random track seed at every episode reset. It explicitly skips 202, 42, and 203, which are reserved for evaluation. The training run itself uses a fixed base seed (`12345`) so the sequence of random training tracks can be reproduced. Four environments are created through `DummyVecEnv`; this is a straightforward single-process setup for the first experiment.

`src/train.py` reads the YAML configuration, creates the environments and PPO agent, periodically saves checkpoints, and saves the final training model under `checkpoints/`. It does not create the final submission artifact yet; export and evaluation are separate later steps. `PyYAML` was added to `requirements.txt` so the script can read the YAML file. Checkpoints and logs are excluded from Git.

This pipeline has been written but not run. The first run will be used to confirm the dependencies, tensor shapes, and training speed before changing hyperparameters or adding frame skipping/cropping.


### Raw-image inference and export preparation

The challenge gives the evaluator a raw RGB image, while PPO trains on a four-frame grayscale stack. I added `CarRacingInference` in `src/export.py` to bridge those formats. It accepts one RGB `uint8` tensor shaped `(96, 96, 3)`, converts it to grayscale, updates a four-frame history, scales the pixels to 0-1, and runs the trained CNN and PPO actor head. It returns three clipped action values in the order steering, gas, brake.

The first image after the inference module is created is repeated in all four stack positions, matching the training environment wrapper. The exported module has a `reset_state()` method for starting another episode when the same model instance is reused. I also added `predict_numpy()` as a helper to convert Gymnasium's NumPy image into a tensor before calling the model.

`export_model()` loads the SB3 `.zip` checkpoint, copies the trained feature extractor and actor head into the inference module, scripts it, and writes `submission/model.pt`. This is a prepared export path, not a confirmed match to the organizer's loader. The official reference does not show how it turns a NumPy image into a tensor, whether it calls the model with a batch dimension, or whether it resets the model between runs. It also does not specify its PyTorch version. Those details still need confirmation, and the artifact must be tested with the official calling pattern before a long training run.

I have not run the exporter or trained a checkpoint yet. I should not describe this inference path as working until the model has been trained, exported, loaded using the evaluator's method, and called on a real environment observation.


### Environment smoke check

I ran `python -m src.env` in the project virtual environment. The reset returned an observation with shape `(4, 96, 96)` and type `uint8`, as expected. The environment accepted all five continuous test actions, and the episode did not terminate or truncate during this short check. The first reward was `7.146`; the next four were `-0.100` each.

This confirms that the environment starts, the grayscale frame stack has the expected shape and type, and valid actions can be sent. The test used a fixed steering/throttle action for only five steps, so it does not show that an agent can complete a lap or that PPO is working. The first reward alone is not a performance result.


### PPO pilot run

I ran `python -m src.train --config configs/ppo_pilot.yaml`. PPO completed one iteration and collected all 4,096 requested timesteps on the CPU. The console reported 81 FPS and 50 seconds elapsed. The SB3 checkpoint was saved as `checkpoints/ppo_car_racing_pilot.zip`.

This confirms that the training script can build the policy, interact with the four environments, perform a PPO update, and save a checkpoint. One update is only a pipeline check; it is not enough to judge lap completion or driving quality. The next check is to export this pilot checkpoint and verify that the inference artifact can be created before starting a long training run.


### Pilot checkpoint export

I ran `python -m src.export checkpoints/ppo_car_racing_pilot.zip --output checkpoints/ppo_car_racing_pilot_model.pt`. The export completed and created `checkpoints/ppo_car_racing_pilot_model.pt`. PyTorch printed deprecation warnings for `torch.jit.script` and `torch.jit.save`, but no export error occurred.

This confirms that the pilot SB3 checkpoint can be converted into the prepared TorchScript artifact. It does not yet confirm that the evaluator's exact `torch.load()` call can load it or that the artifact returns the expected action when called with an observation. I need to test that interface before a long training run, and keep the TorchScript deprecation warning in mind when choosing the final submission format.


### Official loader compatibility check

I tested the exported pilot artifact using the same default `torch.load(path)` call shown in the challenge reference. Loading failed before inference. The installed PyTorch version recognized the file as a TorchScript archive, but rejected it because `torch.load()` defaults to `weights_only=True` and does not allow TorchScript archives in that mode. Therefore, the export succeeding is not enough: the current artifact does not work with the reference loader as written.

I will not start the long training run yet. The organizer needs to confirm whether the evaluator will use `torch.jit.load()`, call `torch.load(path, weights_only=False)`, or run a PyTorch version/configuration that accepts the submitted artifact. This is a compatibility blocker in the submission interface, not a PPO training failure.


### TorchScript inference check

I loaded the pilot export with `torch.jit.load()` and called it with a zero-valued RGB tensor shaped `(96, 96, 3)`. It returned an action tensor shaped `(3,)`: approximately `[0.0045, 0.0045, 0.0000]`, which is within the steering, gas, and brake bounds. PyTorch printed a deprecation warning for `torch.jit.load()`. This confirms a direct TorchScript load and a basic forward call, but it does not match the challenge reference's default `torch.load()` call, which failed in this environment.

The output initially had a `grad_fn`, so I froze the inference model's parameters to avoid building a gradient graph during evaluation. I also prepared `src/eval_official.py` to check the exported model on real RGB observations from CarRacing for a short run. This check loads with `torch.jit.load()`, verifies each action against the environment action space, and uses seed 202 only for evaluation. I have not run this real-image check yet.


### Short raw-image environment inference

After re-exporting the pilot checkpoint, I ran `python -m src.eval_official --model checkpoints/ppo_car_racing_pilot_model.pt --seed 202 --steps 20`. The exported model received raw RGB observations shaped `(96, 96, 3)`, produced 20 actions, and every action passed the CarRacing action-space check. The run completed without termination or truncation and reported a total reward of `8.870` over those 20 steps. The first action was approximately `[0.126, 0.250, 0.271]` for steering, gas, and brake.

This confirms that the TorchScript model can preprocess real environment frames, maintain its frame stack, and return valid actions for a short run when loaded with `torch.jit.load()`. It is not a lap-time or driving-quality result: the pilot had only one PPO update, and this was only 20 steps. The run still prints the TorchScript deprecation warning. The challenge reference's default `torch.load()` remains incompatible in this PyTorch environment, so official-loader compatibility is still unresolved.


### Analysis of the 100k training run

The Monitor logs for `logs/100k` contain 32 completed episodes: eight from each of the four environments. Every episode reached the 3,000-step limit. Their mean return was `-229.694` overall; the best was `-155.932` and the worst was `-257.929`. This indicates that the 100k-step policy did not finish a lap during these training episodes. The episode lengths alone do not prove whether the car stayed on the road.

On seed 202, the exported model also reached the 3,000-step time limit with reward `-256.522`. Using the challenge's 276 tiles for seed 202 and the reward formula (about `1000 / tile_count` for a newly visited tile, minus `0.1` per step), this corresponds to roughly 12 newly visited tiles. This is only a reward-based estimate, not a measured position or distance.

I updated `src/eval_official.py` to report average steering, gas, and brake; the number of steps with brake above 0.5; simultaneous high gas and brake; and the rough tile estimate for known seeds. I have not rerun it since adding these measurements. The high brake value in only the first action is not enough evidence by itself to identify the cause, so I will measure the full action sequence before changing the policy or environment.


### Full action statistics on seed 202

I reran the 100k model with the expanded action diagnostics for 3,000 steps on seed 202. It timed out at the episode limit with reward `-256.522`, and the reward-based estimate was about 12.0 newly visited tiles out of 276. Mean action was approximately `[-0.097, 0.259, 0.001]` for steering, gas, and brake. Brake was above 0.5 on 0 of 3,000 steps, and gas and brake were both above 0.5 on 0 steps.

This measurement does not support the hypothesis that persistent braking is keeping the car still, so I will not cap or disable the brake based on these results. The agent is still far from finishing a lap, and the low mean gas is worth investigating. Next I will keep observations and action mapping unchanged and run a longer baseline, so any change in learning can be compared with the 100k result.


### Measured action behavior and continuation plan

The full-episode action diagnostics on seed 202 showed mean actions of `[-0.097, 0.259, 0.001]` for steering, gas, and brake. The brake exceeded 0.5 on 0 of 3,000 steps, and gas and brake were both above 0.5 on 0 steps. This does not support a persistent-braking diagnosis, so I am leaving the action mapping unchanged for the next experiment.

I added resume support to `src/train.py` and created `configs/ppo_continue.yaml`. The configuration requests 200,000 additional timesteps from the 100k SB3 checkpoint, with the same observations, actions, environment horizon, and PPO settings. It saves the continued model and logs separately. The planned target is about 300,000 total timesteps; SB3 collects complete rollouts, so the exact final count may be slightly higher. This continuation has not been run yet.


### 300k continuation evaluation on seed 202

I evaluated the continued model for 3,000 steps on seed 202. The episode reached the time limit (`terminated=False`, `truncated=True`) and earned `105.797`, compared with `-256.522` for the 100k model on the same seed. The reward-based estimate rose from about 12 to about 112 newly visited tiles out of 276. This is a substantial improvement, but the model still did not finish a lap and the tile count is only an estimate from reward.

The mean deterministic action was approximately `[-0.067, 0.033, 0.319]` for steering, gas, and brake. Brake exceeded 0.5 on 5 of 3,000 steps; gas and brake were both above 0.5 on 2 steps. The mean brake is not zero, but the high-brake events are rare. I will avoid changing the action mapping based on this one evaluation. The next comparison is to evaluate the same checkpoint on the other known seeds and determine whether the improvement generalizes.


### Continuation training logs and brake ablation setup

The resumed run started from the 100k PPO checkpoint and requested 200,000 additional timesteps, so the evaluated model represents about 300k total training timesteps. The Monitor files in `logs/continued` contain 64 completed episodes (16 per environment); all lasted 3,000 steps. Their overall mean return is `-234.393`. The latest Monitor time is about 7,030 seconds, roughly two hours for the continuation on this machine. The SB3 training reward is based on sampled, noisy actions, whereas the exported evaluation policy uses the actor's deterministic mean action. That difference can help explain why their returns differ, but it does not by itself prove the cause of the gap.

The 300k deterministic evaluation on seed 202 had mean brake `0.319` and mean gas `0.033`. I added `--brake-scale` and `--render` options to `src/eval_official.py`. The brake-scale option changes only the action sent to the environment during an evaluation, while reporting both the model output and the executed action. This permits a quick no-brake diagnostic on the same checkpoint before making any training change. A brake-scaled evaluation is only a diagnostic; if it helps, training and final inference must use the same action mapping.

I have not run the brake ablation or the rendered evaluation yet. No brake limit, frame skip, or PPO hyperparameter change has been applied to training.



## Day 3

### PPO network update and validation

My mentor asked me to add a `policy_net` to the model. My first version was incomplete: the CNN output was not flattened, the hard-coded size (`64 * 7 * 7`) did not match the real size, and the new network was not used in `forward()`. Its final 3-output layer also duplicated the action head that PPO already creates.

I fixed `src/policy.py`. The 3 convolution layers now flatten their output, and then the `policy_net` transforms it (4096 values -> 512 neurons -> 256 features). Stable-Baselines3 uses these 256 features for the actor and the critic. I updated `src/export.py` to use the same network at inference time.

A one-iteration test (4,096 steps) followed by an export worked, and the actions stayed within bounds. This only checks that the pipeline works technically, not the quality of the driving. This new network has different parameters from the old one, so I could not resume the old 300k checkpoint with it.

### 500k policy run and full-sequence export check

I trained the new model for about 500k steps. On the random training tracks, it improved clearly: the average score went from about -231 to about +609. However, on the reserved tracks, it performed poorly:

- seed 202: -213 (about 24 tiles);
- seed 42: it left the track after 232 steps;
- seed 203: it left the track after 238 steps.

I also found a bug in the export. The exported TorchScript model did not produce exactly the same actions as the training model. The cause was the grayscale computation: NumPy computes in double precision, while my export used single precision. A single pixel differed by one intensity level (at step 132), but that was enough to change the actions (at step 135). I fixed the export to compute in double precision. Since then, images, actions, and rewards have been identical over the whole tested sequence. The export problem is solved. The real problem, poor driving on unseen tracks, remains open.

### Deterministic-policy training adjustment

The model got good scores during training, where actions are sampled randomly, but was unreliable during evaluation, where it always uses the mean action. I therefore adjusted the training:

- learning rate: 0.0003 -> 0.0001 (smaller updates);
- entropy coefficient: 0.01 -> 0.001 (less forced exploration).

These are hypotheses to test, not proven improvements. I also fixed `src/train.py`: when resuming a training run, `PPO.load()` reloaded the old learning rate, and changing only the attribute would have had no effect. The code now updates the learning rate everywhere (attribute, schedule, and optimizer). Each experiment has its own log and checkpoint folders so that nothing is overwritten. No result is claimed here: the model must first be evaluated on seeds 202, 42, and 203.

### Six-hour training plan and checkpoint comparison

I compared three models. The result that looked inconsistent on seed 202 (-213 versus +51) actually came from two different models.

| Model | Seed 202 | Seed 42 | Seed 203 |
|---|---|---|---|
| Legacy 300k | about 112 tiles, reached the 3,000-step limit | about 39 tiles, reached the 3,000-step limit | about 46 tiles, reached the 3,000-step limit |
| Tuned 500k (lr 1e-4, ent 1e-3) | about 97 tiles, reached the 3,000-step limit | about 42 tiles, left the track at step 2,666 | about 50 tiles, left the track at step 1,895 |

Tiles are estimated from the reward, not measured directly. When the car leaves the playfield, the episode stops with a -100 penalty. Both models make about the same progress, but the legacy one is more stable. The legacy 300k initially could not be resumed with the current CNN because its parameters were incompatible. I later added explicit legacy architecture support and kept its export as a fallback.

For the next experiment, I added a training-only rule: an episode stops with a -100 penalty if the car visits no new tile for 100 steps. I did not add frame skipping, so that I could isolate the effect of this rule.

### 300k no-progress fine-tuning result

I trained for 300,000 more steps with the "stop if no progress" rule, then evaluated in the normal environment. The model reached the 3,000-step limit on all three seeds, but did not finish any lap:

- seed 202: about 110 tiles;
- seed 42: about 52 tiles;
- seed 203: about 26 tiles.

It survived longer on 42 and 203, but progress remained uneven. The rule therefore did not clearly improve the driving.

### Legacy 300k architecture resume support

Since the legacy 300k covered the reserved tracks slightly better, I wanted to continue its training. I added a `ancienne architecture CNN` class (same convolutions, old single `projection` layer) next to the current CNN. `train.py` chooses the right class from the config, and `export.py` automatically detects the legacy architecture. The export of the 300k model gives exactly the same actions as the SB3 model (maximum difference 0.0).

I then resumed training for 300,000 steps, for 606,208 steps in total (about 4 hours), without the stop rule. Results on the reserved seeds, all at the 3,000-step limit, with no lap completed:

- seed 202: about 150 tiles (112 before);
- seed 42: about 38 tiles (39 before);
- seed 203: about 41 tiles (46 before).

The average went from about 66 to 76 tiles, but only thanks to seed 202. The model brakes a lot on 202 and 42. This was a modest, uneven gain.

### Standard continuation configuration prepared

I prepared the next step: `configs/ppo_legacy_600k_standard_continue.yaml` resumes the legacy 600k model for 300,000 more steps, with the normal environment, `learning_rate: 0.0001`, and `ent_coef: 0.001`. I am not using the stop rule, because it helped survival without allowing a lap to be finished. The configuration and checkpoint loading were validated without starting the training. No result is claimed.

### Standard continuation launched

On October 26, I launched this training with:

```powershell
.venv\Scripts\python.exe -m src.train `
  --config configs\ppo_legacy_600k_standard_continue.yaml `
  --resume checkpoints\ppo_car_racing_legacy_600k_lr1e4_ent1e3.zip
```

The logs and checkpoints have their own folders (`logs/legacy_600k_standard_continue` and `checkpoints/legacy_600k_standard_continue`), so nothing is overwritten. The final model will be `checkpoints/ppo_car_racing_legacy_900k_standard.zip`.

At the time of this entry, the process was still running, with no completed episode and no checkpoint yet. No improvement is therefore claimed. Once training finishes, I must export the model, evaluate it on seeds 202, 42, and 203, and compare it with the current best (`ppo_car_racing_legacy_600k_lr1e4_ent1e3_model.pt`).

## Day 4 

### Four-layer policy network experiment

I changed `CarRacingCNN.policy_net` to use four fully connected layers after the three convolution layers: `4096 -> 2048 -> 1024 -> 512 -> 256`. The last 256 values are visual features shared by PPO's actor and critic; they are not the three driving actions. Stable-Baselines3 still creates the actor and critic output heads.

I prepared `configs/ppo_policy_4layer_900k.yaml` for a fresh 900,000-timestep run. It selects `CarRacingCNN`, uses the existing grayscale four-frame observations, keeps the normal 3,000-step episode limit, and uses four vector environments. The run keeps the current conservative PPO settings (`learning_rate: 0.0001`, `ent_coef: 0.001`) so that the main architectural change is the four-layer feature network. I did not enable the training-only no-progress termination rule.

This configuration starts from scratch: it uses `total_timesteps`, does not load an older checkpoint, and writes logs, intermediate checkpoints, and the final SB3 model to new `policy_net_4layer_900k` paths. Old checkpoints have incompatible dense-layer shapes and must not be used with `--resume`. The four-layer network has about 11.2 million parameters, so training may be slower than the smaller earlier network. The configuration has been prepared, but this experiment has not yet been launched and no performance improvement is claimed.

To run the experiment:

```powershell
.venv\Scripts\python.exe -m src.train --config configs\ppo_policy_4layer_900k.yaml
```

After training, export and evaluate the model on the three reserved seeds:

```powershell
.venv\Scripts\python.exe -m src.export checkpoints\ppo_car_racing_policy_net_4layer_900k.zip --output checkpoints\ppo_car_racing_policy_net_4layer_900k_model.pt
.venv\Scripts\python.exe -m src.eval_official --model checkpoints\ppo_car_racing_policy_net_4layer_900k_model.pt --seed 202 --steps 3000
.venv\Scripts\python.exe -m src.eval_official --model checkpoints\ppo_car_racing_policy_net_4layer_900k_model.pt --seed 42 --steps 3000
.venv\Scripts\python.exe -m src.eval_official --model checkpoints\ppo_car_racing_policy_net_4layer_900k_model.pt --seed 203 --steps 3000
```

I also fixed the startup issue seen in the remote runtime. A fresh run no longer requires `ancienne architecture CNN` just to import `train.py`; legacy support is optional and is only needed when selecting or exporting an old legacy checkpoint. The 900k config declares the expected `policy_net` widths (`2048, 1024, 512, 256`). Before learning starts, `train.py` checks that the constructed PPO model actually has those four layers and prints their widths. If the remote `policy.py` is stale or has a different architecture, training stops with a clear error instead of spending hours on the wrong network. The exporter follows the same optional-legacy rule for new-model exports.
