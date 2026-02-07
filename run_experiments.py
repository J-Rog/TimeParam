#!/usr/bin/env python3
"""
Utility to launch many test_wor.py sweeps defined in a YAML file.

See experiments/overnight.yaml for an example configuration.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SCRIPT = REPO_ROOT / "test_wor.py"
CONFIG_OUT_DIR = REPO_ROOT / "tmp" / "batch_configs"


def resolve_path(path_str):
    if not path_str:
        return None
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


def slugify(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def write_overrides(base_config, overrides, run_name):
    if not overrides:
        return base_config
    if not base_config:
        raise ValueError(f"config_overrides provided for {run_name} but no agent_config specified")
    base_path = Path(base_config)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config {base_path} not found")
    with base_path.open("r") as src:
        data = yaml.safe_load(src) or {}
    for key, value in overrides.items():
        data[key] = value
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG_OUT_DIR / f"{slugify(run_name)}.yaml"
    with out_path.open("w") as dst:
        yaml.safe_dump(data, dst, sort_keys=False)
    return str(out_path)


def extend_flag(cmd, flag, values):
    if values is None:
        return
    if isinstance(values, (list, tuple)):
        if not values:
            return
        cmd.append(flag)
        cmd.extend(str(v) for v in values)
    else:
        cmd.extend([flag, str(values)])


def run_single_experiment(run_name, cfg, dry_run, script_path):
    agent = cfg.get("agent")
    route = cfg.get("route")
    sweep_log = cfg.get("sweep_log")
    if not agent or not route or not sweep_log:
        raise ValueError(f"Experiment {run_name} missing required agent/route/sweep_log fields")

    cmd = [sys.executable, str(script_path)]
    cmd.extend(["--agent", agent])
    cmd.extend(["--route", resolve_path(route)])

    if cfg.get("route_id") is not None:
        cmd.extend(["--route-id", str(cfg["route_id"])])

    agent_config = resolve_path(cfg.get("agent_config"))
    agent_config = write_overrides(agent_config, cfg.get("config_overrides"), run_name)
    if agent_config:
        cmd.extend(["--agent-config", agent_config])

    cmd.extend(["--sweep-log", resolve_path(sweep_log)])

    control_latencies = (
        cfg.get("sweep_control_latencies")
        or cfg.get("control_latencies")
        or cfg.get("latencies")
    )
    if control_latencies:
        extend_flag(cmd, "--sweep-control-latencies", control_latencies)

    vehicle_densities = cfg.get("sweep_vehicle_density", cfg.get("vehicle_densities"))
    if vehicle_densities:
        extend_flag(cmd, "--sweep-vehicle-density", vehicle_densities)

    pedestrian_densities = cfg.get("sweep_pedestrian_density", cfg.get("pedestrian_densities"))
    if pedestrian_densities:
        extend_flag(cmd, "--sweep-pedestrian-density", pedestrian_densities)

    extra_args = cfg.get("extra_args", [])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n--- Running {run_name} ---")
    print(" ".join(cmd))
    if dry_run:
        return

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Experiment {run_name} failed with exit code {completed.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Batch runner for test_wor sweeps.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML file describing experiments (see experiments/overnight.yaml).",
    )
    parser.add_argument(
        "--script",
        default=str(DEFAULT_SCRIPT),
        help="Runner script to execute for each experiment (default: test_wor.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f) or {}

    experiments = config.get("experiments") or []
    if not experiments:
        raise ValueError("No experiments found in config file.")

    base_script = resolve_path(args.script)
    for exp in experiments:
        routes = exp.get("routes") or [{}]
        base = {k: v for k, v in exp.items() if k != "routes"}
        base_extra_args = list(base.pop("extra_args", []))
        base_overrides = dict(base.get("config_overrides") or {})
        base.pop("config_overrides", None)
        for idx, route_cfg in enumerate(routes, start=1):
            run_cfg = base.copy()
            run_cfg.update(route_cfg)

            merged_overrides = dict(base_overrides)
            if route_cfg.get("config_overrides"):
                merged_overrides.update(route_cfg["config_overrides"])
            if merged_overrides:
                run_cfg["config_overrides"] = merged_overrides

            combined_extra = list(base_extra_args)
            if route_cfg.get("extra_args"):
                combined_extra.extend(route_cfg["extra_args"])
            if combined_extra:
                run_cfg["extra_args"] = combined_extra

            script_override = resolve_path(run_cfg.get("script")) if run_cfg.get("script") else base_script
            run_cfg.pop("script", None)

            run_name = f"{exp.get('name', 'experiment')}_{route_cfg.get('name', idx)}"
            run_single_experiment(run_name, run_cfg, args.dry_run, script_override)


if __name__ == "__main__":
    main()
