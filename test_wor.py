import argparse
import os
import sys

# Allow importing the Leaderboard/scenario_runner utilities that ship in WorldOnRails.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_ROOT = os.path.join(CURRENT_DIR, "leaderboard_codes")
SCENARIO_RUNNER_ROOT = os.path.join(CURRENT_DIR, "scenario_runner")

for extra_path in (LEADERBOARD_ROOT, SCENARIO_RUNNER_ROOT):
    if os.path.isdir(extra_path) and extra_path not in sys.path:
        sys.path.append(extra_path)

CARLA_ROOT = os.environ.get("CARLA_ROOT", os.path.expanduser("~/CARLA_0.9.16"))
CARLA_PYTHON_PATHS = [
    os.path.join(CARLA_ROOT, "PythonAPI"),
    os.path.join(CARLA_ROOT, "PythonAPI", "carla"),
]
for path in CARLA_PYTHON_PATHS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)

from wor_runner import FixedLatencyAdapter, run_sweep

DEFAULT_CONTROL_LATENCY = 0.05
DEFAULT_VEHICLE_DENSITY = 20
DEFAULT_PEDESTRIAN_DENSITY = 50
MIN_CONTROL_PERIOD = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Run the WoR agent with leaderboard-style scoring.")
    parser.add_argument(
        "--route",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_route.xml"),
        help="Path to the route XML file to follow.",
    )
    parser.add_argument(
        "--agent",
        default="wor_nc",
        help="Agent identifier recognized by PCLA (default: wor_nc).",
    )
    parser.add_argument(
        "--sweep-control-latencies",
        type=float,
        nargs="+",
        help="If set, run once for each listed control latency (seconds).",
    )
    parser.add_argument(
        "--sweep-vehicle-density",
        type=int,
        nargs="+",
        help="Vehicle density values to sweep.",
    )
    parser.add_argument(
        "--sweep-pedestrian-density",
        type=int,
        nargs="+",
        help="Pedestrian density values to sweep.",
    )
    parser.add_argument(
        "--sweep-log",
        type=str,
        help="Optional CSV path for logging sweep results (rows appended).",
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        help="Override path to the agent config file (useful when testing multiple checkpoints).",
    )
    parser.add_argument(
        "--route-id",
        type=int,
        help="Restrict the provided route XML to this <route id>. When unset, the first route is used.",
    )
    return parser.parse_args()


def _timing_factory(args, control_latency, fixed_dt):
    return FixedLatencyAdapter(control_latency, fixed_dt, MIN_CONTROL_PERIOD)


def main():
    args = parse_args()
    run_sweep(
        args,
        _timing_factory,
        DEFAULT_CONTROL_LATENCY,
        DEFAULT_VEHICLE_DENSITY,
        DEFAULT_PEDESTRIAN_DENSITY,
    )


if __name__ == "__main__":
    main()
