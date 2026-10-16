# Self-distilled delay compensation for frozen WoR

This pipeline collects CARLA data with Traffic Manager and trains only a `DelayFeatureAdapter`. It needs the public WoR checkpoint, but **no WoR Q-value labels, semantic masks, or original training dataset**. The current camera observation is the frozen teacher; a delayed observation is the adapter's input. The camera encoders, speed encoder, action head, and their batch-normalization statistics stay frozen.

## Quick start

Run from the `TimeParam` repository. Training requires the dependencies in `requirements-delay.txt`; CARLA is needed only for collection and closed-loop evaluation. Use the existing CARLA/Leaderboard/ScenarioRunner environment for those commands, including its CARLA Python package and `CARLA_ROOT` setting. Start a compatible CARLA server first. These commands own synchronous ticking and reload the selected town, so use a dedicated simulator session without another ticking client.

```bash
python -m pip install -r requirements-delay.txt
export CARLA_ROOT=/path/to/CARLA_0.9.16

# All routes in this XML, with separate trajectories for each traffic density.
python collect_carla_delay_data.py \
  --routes routes/routes_devtest.xml \
  --output data/carla_delay \
  --vehicle-densities 0 20 50 --seeds 42 43 \
  --fps 20 --seconds 300

# Uses the existing public checkpoint by default. No act0 or act_0 is read.
python finetune_time_adapter_from_json.py \
  --data-dir data/carla_delay \
  --output outputs/time_adapter.th \
  --delays-ms 0 50 100 150 200 --max-latency-s 0.2 \
  --ego-state full --feature-weight 1.0 --epochs 10

# Three arms on only the whole routes reserved by the training split.
python evaluate_time_adapter.py \
  --routes routes/routes_devtest.xml --validation-routes-only \
  --adapter outputs/time_adapter.th --output outputs/delay_eval_run1 \
  --vehicle-densities 0 20 50 --seeds 44 45 \
  --delays-ms 0 50 100 150 200 --fps 20
```

Collection and evaluation refuse existing run directories to protect previous recordings. `--route-ids` selects explicit XML route IDs; all routes run if omitted. `--host`, `--port`, and `--tm-port` select a server. Repeat collection with `--weather WetNoon`, different seeds and other route XMLs/towns into the same dataset root. Weather names are CARLA `WeatherParameters` presets. Traffic actors use Traffic Manager's collision/traffic-light defaults; the ego follows the XML route with `set_path`. See the [official Traffic Manager documentation](https://carla.readthedocs.io/en/latest/tuto_G_traffic_manager/).

The collector covers vehicle traffic; pedestrian workloads are not added by these new scripts. Confirm the recorded routes contain the desired turns, intersections, lights, weather and congestion. Traffic Manager may stall or deviate: metadata records the completion percentage rather than assuming every recording completes. Camera RGB is saved losslessly as PNG; disk use is substantial.

## What is optimized

For current time `t` and measured observation age `d`:

```text
current RGB(t) -> frozen visual encoder -> current features -> frozen head -> teacher
stale RGB(t-d) -> frozen visual encoder -> delay adapter     -> frozen head -> student
                                           ^
                            d, speed(t), yaw rate(t), previous control
```

The default loss is `KL(student || teacher) + feature_weight * MSE(corrected_features, current_features)`. The reverse KL is computed explicitly, with detached teacher outputs. `--kl-direction teacher_student` enables the conventional teacher-to-student distillation alternative. `--temperature` defaults to **1**, with the usual temperature-squared scaling; the old config's `temperature: 0.01` was for unavailable Q-label supervision and is not used here.

Both distributions use the **current** route command and current speed, with the same speed-bin interpolation as deployment. This repository's `ImageAgent.action_prob` concatenates repeated steering and throttle logits before its softmax; that legacy layout differs from the original Q-training joint-action layout. The new teacher/student path deliberately matches the deployed distribution. Steering/throttle decoding, braking threshold and speed caps retain their existing behavior. This experiment does not silently replace the policy's action composition.

The adapter learns a bounded residual gated by delay. Its last layer starts at zero, and the residual is multiplied by delay, giving exact identity at zero delay even after training. Its feature dimension is inferred (512 wide-only; 576 with narrow features). `--ego-state full` supplies `[speed_m_s, yaw_rate_rad_s, previous_steer, previous_throttle, previous_brake]`; `speed` and `none` allow ablations. The scale, hidden size, state dimensions and delay range are saved in the checkpoint and restored at deployment. The bounded correction defaults to 0.1 per feature component; feature-space distance and validation metrics should guide later scale experiments.

## Collection/data contract

Each trajectory has its own `data.json`, `route.xml`, and `rgb/` directory. Camera frames must be synchronized. Files contain **uncropped RGB**:

- Wide: 240 x 480 panorama, stitching three 240 x 160 views with yaw -55, 0, +55 degrees and FOV 60 degrees.
- Narrow: 240 x 384, yaw 0, FOV 50 degrees.
- Pose: `camera_x`, `camera_z` from the policy config. Lens settings match the repository's sensor setup.
- Shared preprocessing applies `wide_crop_top` and `narr_crop_bottom` exactly once. A zero crop is supported.

Example `data.json` (include many frames in a real sequence):

```json
{
  "schema_version": 1,
  "town": "Town02",
  "route_id": "stable_route_geometry_hash",
  "frames": [
    {
      "frame": 123,
      "timestamp": 6.15,
      "wide_rgb": "rgb/000000_wide.png",
      "narr_rgb": "rgb/000000_narr.png",
      "spd": 4.2,
      "cmd": 3,
      "yaw_rate": 0.03,
      "previous_control": [0.02, 0.4, 0.0]
    }
  ]
}
```

`timestamp` is the CARLA image capture time in seconds, strictly increasing within a trajectory. `spd` is m/s. Commands are zero-based WoR values: left=0, right=1, straight=2, lane follow=3, change left=4, change right=5. The collector uses the same route command generator as deployment. `previous_control` is the last applied control read at the current timestamp, before calculating the next control. Yaw rate is converted from CARLA degrees/s to radians/s. Telemetry and route commands remain current when only images are delayed.

The loader enumerates each requested delay for every eligible current frame. It selects the newest capture at or before `t-delay`, measures the timestamp difference, and never crosses trajectories. Warmup frames and incomplete windows caused by gaps are excluded; larger-than-supported ages are not silently clamped. Camera yaw augmentation and the original Q-label loader are not used. Converting other logs requires this schema and the same camera contract; timestamps and commands are mandatory rather than guessed.

## Splits, provenance and outputs

Repeated runs of identical route geometry are grouped before the deterministic train/validation split. At least two distinct route groups are required. Training writes:

- `time_adapter.th`: best validation checkpoint, adapter settings, frozen teacher SHA-256, policy config, training/validation route identities, delays, objective settings and seed.
- `time_adapter.split.json`: whole-trajectory `train` and `val` lists relative to the dataset root; reusable through `--split-manifest`.
- `time_adapter.metrics.json`: train/validation action KL and feature MSE per delay, including the unadapted delayed baseline and sample counts.

`--heldout-towns Town02` reserves every trajectory in that town for validation (collect other towns for training). Explicit manifests must partition every trajectory and cannot place repetitions of one route in both sets. Route hashes group exact route geometry, not every overlapping street segment. Audit spatial overlap or use an unseen town for a stronger generalization test.

Closed-loop evaluation rejects known training routes and checks the frozen checkpoint hash/config when loading the adapter. `--validation-routes-only` is convenient for the first baseline. For final reporting, pass a separate held-out test XML without that flag; using validation routes to choose the checkpoint and report final performance risks optimistic estimates.

## Closed-loop protocol and timing

The evaluator reloads the world and seeds Traffic Manager for each `(route, traffic density, seed, arm, delay)` combination, applying the same weather and camera rig. It runs:

1. Frozen WoR with zero injected delay.
2. Frozen WoR with each requested observation delay.
3. Frozen WoR plus the adapter at those same delays, including zero for an identity check.

Controls are computed and applied every camera tick; no separate delayed-control queue confounds the observation-delay comparison. Delay values must align to the selected tick interval. Initial insufficient-history controls are marked `warmup` and excluded from steady-state timing summaries, but their driving outcomes remain part of the route result. Real sensor frames, rather than callback processing times, determine observation age. The legacy `test_wor.py` still works, but its control-latency queue is a separate experiment; use this evaluator for the matched observation-delay study.

Each run saves `result.json`, the run config/route, and `timing.jsonl`. Root-level `results.json` and `summary.json` report completion rate, route completion, collision counts by actor class, red-light events, lane-invasion events/markings and measured latency. The existing repository scorer supplies collision/red-light/completion criteria; this is a lightweight leaderboard-like evaluation, not the official leaderboard benchmark. Errors are written to `error.json` and stop the sweep rather than disappearing from the results.

Timing fields have distinct meanings:

- `observation_age_sim_s`: current simulator time minus selected image capture time.
- `sensing_to_actuation_sim_s`: selected capture time to the first simulator tick after that control was sent. This normally includes one additional physics step (e.g. 200 ms image age + 50 ms actuation step).
- `receive_to_apply_wall_s`: earliest selected camera callback receipt to completion of the control RPC, including buffering and inference. This starts at receipt; CARLA does not provide a host-monotonic camera exposure timestamp.
- `inference_wall_s`: measured agent execution time, including GPU synchronization when outputs are converted into controls.

Simulator time and wall time are different clocks. Synchronous CARLA pauses while inference and logging run, so these results measure injected observation aging, not a validated real-time compute schedule. Do not add wall inference time to simulator delay and call it a measured total. The timing log retains both clocks for analysis.

## Verification

```bash
python -m pytest -q tests/test_self_distillation.py
```

Tests cover label-free paired frames and current telemetry, timestamp gaps, route leakage, camera colors/order/crops, both KL directions/gradient boundaries, exact zero-delay identity after optimization, checkpoint restoration, wide/narrow and speed-head variants, partial batches, frozen BN/weights, runtime history/state and exact sensor-frame synchronization. Synthetic-data smoke tests validate plumbing only; driving improvements require CARLA trajectories and the closed-loop runs above.
