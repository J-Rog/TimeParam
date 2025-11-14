#!/usr/bin/env python3
"""
Export the WoR CameraModel (standard BatchNorm version) to TorchScript.

Usage example:

    python export_torchscript.py \
        --config outputs/config_bn.yaml \
        --checkpoint outputs/bn_1.0r_no_mlp.th \
        --output models/camera_model.ts \
        --device cuda
"""

import argparse
import os
import sys
import yaml

import torch

# Ensure repo root is on sys.path when run from anywhere
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pcla_agents.wor.rails.models import CameraModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export CameraModel to TorchScript.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config used during training/evaluation (e.g., outputs/config_bn.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to the .th/.pth checkpoint. Defaults to config['main_model_dir'].",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the TorchScript .ts file.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on during export (default: cuda if available).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading the checkpoint (default False).",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(os.path.dirname(config_dir))

    for key, value in list(config.items()):
        if key.endswith("_dir") and isinstance(value, str):
            config[key] = os.path.abspath(os.path.join(project_root, value))

    return config


def build_model(config, checkpoint_path, device, strict_load):
    model = CameraModel(config).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    model.load_state_dict(state_dict, strict=strict_load)
    model.eval()
    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config.get("main_model_dir")
    if not checkpoint_path:
        raise ValueError("Checkpoint path not provided and main_model_dir missing from config.")
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    model = build_model(config, checkpoint_path, device, strict_load=args.strict)

    print(f"Scripting model on device {device} ...")
    scripted = torch.jit.script(model)
    scripted.save(args.output)
    print(f"TorchScript model saved to {args.output}")


if __name__ == "__main__":
    main()
