# Time-Parameterized Control Refactor Report

## Purpose

This work prepares the World on Rails (WoR) autonomous-driving policy for a
time-parameterized control experiment. The intended research question is
whether a fixed, pretrained driving policy can be augmented with a small,
latency-aware module that compensates for stale visual observations.

The design preserves the original driver model rather than retraining it. A
future delay adapter will be the only trainable component.

## Original Policy Structure

Before the refactor, `CameraModel.policy()` performed both of these operations
in one method:

1. Convert camera images into a pooled visual feature vector using the ResNet
   visual backbone.
2. Pass that vector through the action head to obtain steering, throttle, and
   brake logits for the active route command.

The default WoR configuration uses the wide-camera visual backbone and produces
a 512-dimensional pooled feature vector. The action head maps that feature to
the driving policy output.

## Completed Refactor

`pcla_agents/wor/rails/models/main_model.py` was refactored to expose the two
stages explicitly:

```text
camera images
  -> extract_policy_features(...)
  -> pooled visual feature
  -> policy_from_features(...)
  -> steering, throttle, and brake logits
```

The following methods now separate those responsibilities:

- `extract_policy_features(wide_rgb, narr_rgb)` returns the pooled visual
  feature used by the policy head.
- `policy_from_features(features, cmd, spd=None)` applies the existing action
  head to an already-computed feature vector.
- `policy(...)` remains the normal inference entry point. It calls the two new
  methods and optionally accepts a delay adapter and latency input.

When no delay adapter is supplied, the feature vector is unmodified. Therefore,
the refactored policy follows the original inference path:

```text
images -> original visual feature -> original action head -> original control logits
```

## Verification Performed

`tmp_test.py` verifies policy-path equivalence by comparing:

```text
model.policy(...)
```

against:

```text
features = model.extract_policy_features(...)
model.policy_from_features(features, ...)
```

The tensor comparison passed, confirming that the refactored wrapper and the
explicit feature/action path return the same policy outputs when no delay
adapter is used.

## Intended Time-Parameterized Architecture

The next implementation step will insert a lightweight trainable delay adapter
between the frozen visual encoder and the frozen action head:

```text
stale camera image
  -> frozen WoR visual backbone
  -> pooled feature h
  -> trainable DelayFeatureAdapter(h, latency)
  -> corrected feature h'
  -> frozen WoR action head
  -> steer / throttle / brake logits
```

The adapter will produce a bounded residual correction. A useful formulation is:

```text
h' = h + alpha * normalized_latency * tanh(adapter(h, latency))
```

At zero latency, `h'` equals `h`; the original driver policy is therefore
preserved exactly.

## Training Plan

The pretrained WoR model will remain frozen, including its visual backbone and
action head. Training will update only the delay adapter.

Each training example should use:

```text
visual input:   image captured at T - delay
latency input:  delay
target:         planner-derived action label at T
```

This relationship is essential. Supplying a current image with a random delay
label would allow the model to ignore timing information.

## Runtime Plan

During CARLA evaluation, `ImageAgent` will maintain a short history of camera
frames. At decision time `T`, it will select an image packet from approximately
`T - delay`, pass the actual image age to the delay adapter, and apply the
resulting control immediately. This isolates stale visual perception from the
existing simulated control-application delay.

## Next Steps

1. Add `DelayFeatureAdapter` as a new model module.
2. Load the adapter separately from the unchanged `wor_nc` checkpoint.
3. Add a camera-frame history buffer to `ImageAgent`.
4. Add an `--observation-latency` evaluation argument.
5. Create a temporally delayed training dataset that pairs old images with
   current action targets.
6. Train and save adapter-only checkpoints.
7. Compare the original driver, a latency-blind delayed baseline, and the
   time-aware delay adapter at multiple observation delays.

## Implementation Safeguard

Before using the original training `forward()` path, confirm that
`action_logits()` remains a module-level function. `CameraModel.forward()`
calls it directly, so it must not be accidentally nested inside the
`CameraModel` class during refactoring.
