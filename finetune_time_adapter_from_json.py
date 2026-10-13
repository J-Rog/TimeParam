#!/usr/bin/env python3
"""
Train a time-delay feature adapter for the frozen WoR camera policy.

Each training sample:
  stale wide/narrow images at time t - delay
  -> frozen CameraModel feature extractor
  -> trainable DelayFeatureAdapter(features, delay)
  -> frozen CameraModel action head
  -> KL loss against action values from the current frame at time t

Expected JSON dataset structure:
  dataset_root/
    sequence_001/
      data.json
      ... referenced RGB files ...
    sequence_002/
      data.json
      ...

Each frame must contain the camera paths used by JSONLabeledMainDataset and
action labels such as act0 / act_0 / act_val_0.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from pcla_agents.wor.rails.datasets.json_main_dataset import (
    JSONLabeledMainDataset,
    _read_rgb,
)
from pcla_agents.wor.rails.models.main_model import CameraModel
from pcla_agents.wor.rails.models.time_adapter import DelayFeatureAdapter


class DelayedJSONActionDataset(JSONLabeledMainDataset):
    """
    Returns stale visual observations and the action-value target from a newer,
    current frame in the same trajectory.
    """

    def __init__(self, data_dir, config_path, max_latency_s, fps):
        super().__init__(
            data_dir,
            config_path,
            use_augment=False,
        )

        self.max_latency_s = float(max_latency_s)
        self.fps = float(fps)
        self.max_delay_steps = max(1, round(self.max_latency_s * self.fps))

        # A sample identifies its sequence, current time index, and camera.
        self.samples = []
        for sequence_index, sequence in enumerate(self.seq_meta):
            for current_index in range(sequence["n"]):
                for camera_index in range(len(self.camera_yaws)):
                    self.samples.append(
                        (sequence_index, current_index, camera_index)
                    )

        if not self.samples:
            raise RuntimeError(
                "No valid sequences were found. "
                "Expected data.json files below --data-dir."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence_index, current_index, camera_index = self.samples[index]
        sequence = self.seq_meta[sequence_index]
        frames = sequence["frames"]
        sequence_dir = sequence["dir"]

        # Randomly choose a delay that is feasible at this point in the sequence.
        delay_steps = random.randint(
            0,
            min(self.max_delay_steps, current_index),
        )
        stale_index = current_index - delay_steps

        stale_frame = frames[stale_index]
        current_frame = frames[current_index]

        def resolve_path(path):
            return os.path.join(sequence_dir, path.lstrip("/"))

        # Input: stale camera images.
        wide_path, narr_path = self._resolve_rgb_paths(
            stale_frame,
            camera_index,
        )
        stale_wide = _read_rgb(resolve_path(wide_path))
        stale_narr = _read_rgb(resolve_path(narr_path))

        stale_wide = stale_wide[self.wide_crop_top:, :, :]
        stale_narr = stale_narr[:-self.narr_crop_bottom, :, :]

        # Target: action distribution from the current frame.
        current_action_values = self._read_action_values(
            current_frame,
            camera_index,
        )

        current_speed = float(
            self._scalarize(current_frame.get("spd", 0.0))
        )
        current_command = int(
            self._scalarize(current_frame.get("cmd", 0))
        )

        latency_s = self._frame_time(
            current_frame,
            current_index,
        ) - self._frame_time(
            stale_frame,
            stale_index,
        )
        latency_s = max(0.0, float(latency_s))

        return (
            stale_wide,
            stale_narr,
            current_action_values,
            current_speed,
            current_command,
            latency_s,
        )

    def _frame_time(self, frame, fallback_index):
        # Prefer a real simulator timestamp when the converted JSON contains it.
        for key in ("timestamp", "time", "sim_time"):
            if key in frame:
                return float(self._scalarize(frame[key]))

        # Otherwise assume uniformly sampled data.
        return fallback_index / self.fps


def combine_action_logits(raw_logits, num_steers, num_throts):
    """
    Convert separate steer/throttle/brake logits into the joint WoR action
    distribution, matching CameraModel's original action_logits logic.
    """
    steer_logits = raw_logits[..., :num_steers]
    throt_logits = raw_logits[
        ...,
        num_steers:num_steers + num_throts,
    ]
    brake_logits = raw_logits[..., -1:]

    steer_logits = steer_logits.repeat(1, 1, 1, num_throts)
    throt_logits = throt_logits.repeat_interleave(num_steers, dim=-1)

    return torch.cat(
        [steer_logits + throt_logits, brake_logits],
        dim=-1,
    )


def action_logits_from_features(base_model, features, speeds):
    """
    Run only the frozen action head. Gradients still flow from this function
    into `features`, and therefore into the trainable adapter.
    """
    if base_model.all_speeds:
        raw_logits = base_model.act_head(features).view(
            -1,
            base_model.num_cmds,
            base_model.num_speeds,
            base_model.num_steers + base_model.num_throts + 1,
        )
    else:
        speed_features = base_model.spd_encoder(speeds[:, None])
        action_input = torch.cat([features, speed_features], dim=1)
        raw_logits = base_model.act_head(action_input).view(
            -1,
            base_model.num_cmds,
            1,
            base_model.num_steers + base_model.num_throts + 1,
        )

    return combine_action_logits(
        raw_logits,
        base_model.num_steers,
        base_model.num_throts,
    )


def load_base_model(config_path, checkpoint_path, device):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # The checkpoint supplies all weights, so do not download ImageNet weights.
    config["imagenet_pretrained"] = False

    base_model = CameraModel(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    base_model.load_state_dict(state_dict)

    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    return base_model, config


def train(args):
    if args.max_latency_s <= 0:
        raise ValueError("--max-latency-s must be positive.")

    device = torch.device(args.device)
    base_model, config = load_base_model(
        args.config_path,
        args.checkpoint,
        device,
    )

    if base_model.act_head[0].in_features != 512:
        raise ValueError(
            "This script expects a 512-D visual feature vector, but the "
            f"loaded action head expects {base_model.act_head[0].in_features}."
        )

    delay_adapter = DelayFeatureAdapter(
        feature_dim=512,
        max_latency_s=args.max_latency_s,
    ).to(device)

    optimizer = Adam(
        delay_adapter.parameters(),
        lr=1e-4,
    )

    dataset = DelayedJSONActionDataset(
        data_dir=args.data_dir,
        config_path=args.config_path,
        max_latency_s=args.max_latency_s,
        fps=args.fps,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        delay_adapter.train()
        running_loss = 0.0

        for (
            stale_wides,
            stale_narrs,
            current_action_values,
            current_speeds,
            current_commands,
            latencies_s,
        ) in loader:

            stale_wides = (
                stale_wides.float()
                .permute(0, 3, 1, 2)
                .to(device, non_blocking=True)
            )
            stale_narrs = (
                stale_narrs.float()
                .permute(0, 3, 1, 2)
                .to(device, non_blocking=True)
            )
            current_action_values = current_action_values.float().to(
                device,
                non_blocking=True,
            )
            current_speeds = current_speeds.float().to(
                device,
                non_blocking=True,
            )
            current_commands = current_commands.long().to(
                device,
                non_blocking=True,
            )
            latencies_s = latencies_s.float().to(
                device,
                non_blocking=True,
            )

            # Extract stale visual features with no gradients through base model.
            with torch.no_grad():
                stale_features = base_model.extract_policy_features(
                    stale_wides,
                    stale_narrs,
                )

            corrected_features = delay_adapter(
                stale_features,
                latencies_s,
            )
            student_all_logits = action_logits_from_features(
                base_model,
                corrected_features,
                current_speeds,
            )

            # JSON labels: [batch, command, action, speed]
            # Policy logits: [batch, command, speed, action]
            target_all = F.softmax(
                current_action_values.permute(0, 1, 3, 2)
                / config["temperature"],
                dim=-1,
            )

            # Train using the route command active at the current frame.
            batch_indices = torch.arange(
                student_all_logits.shape[0],
                device=device,
            )
            student_action_logits = student_all_logits[
                batch_indices,
                current_commands,
            ]
            current_action_target = target_all[
                batch_indices,
                current_commands,
            ]

            loss = F.kl_div(
                F.log_softmax(student_action_logits, dim=-1),
                current_action_target,
                reduction="batchmean",
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={global_step} "
                    f"loss={running_loss / args.log_every:.6f}"
                )
                running_loss = 0.0

        checkpoint = {
            "state_dict": delay_adapter.state_dict(),
            "epoch": epoch + 1,
            "max_latency_s": args.max_latency_s,
            "feature_dim": 512,
        }
        torch.save(checkpoint, args.output)
        print(f"Saved adapter checkpoint: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing trajectory folders with data.json files.",
    )
    parser.add_argument(
        "--config-path",
        default="outputs/config_timeparam.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "pcla_agents/wor_pretrained/nocrash_weights/"
            "main_model_16.th"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/time_adapter.th",
    )
    parser.add_argument("--max-latency-s", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
