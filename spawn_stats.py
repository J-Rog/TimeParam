#!/usr/bin/env python3
import argparse
import os
import statistics
import sys
from collections import defaultdict

import yaml

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

import carla

from leaderboard_codes.route_indexer import RouteIndexer as LegacyRouteIndexer
from wor_common import (
    DEFAULT_TM_PORT,
    _apply_route_weather,
    _load_world_and_wait,
    _materialize_route_subset,
    destroy,
    make_sync,
    spawn_tm_vehicles,
    spawn_walkers,
)

DEFAULT_ROUTES_CONFIG = os.path.join(CURRENT_DIR, "experiments", "full.yaml")
DEFAULT_ROUTE_NAMES = ["route0", "route1", "route2", "route3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure realized CARLA spawn counts for specific vehicle/pedestrian targets."
    )
    parser.add_argument(
        "--route",
        help=(
            "Single route XML file used to pick the town and ego spawn point. "
            "If omitted, routes are loaded from --routes-config/--route-names."
        ),
    )
    parser.add_argument(
        "--route-id",
        type=int,
        help="When --route is provided, optionally restrict it to this <route id>.",
    )
    parser.add_argument(
        "--routes-config",
        default=DEFAULT_ROUTES_CONFIG,
        help="YAML experiment config containing named route entries (default: experiments/full.yaml).",
    )
    parser.add_argument(
        "--route-names",
        nargs="+",
        default=DEFAULT_ROUTE_NAMES,
        help="Named routes to run from --routes-config when --route is not provided.",
    )
    parser.add_argument(
        "--scenario-json",
        default=os.path.join(CURRENT_DIR, "leaderboard_codes", "no_scenarios.json"),
        help="Scenario JSON used with the RouteIndexer (default: leaderboard no_scenarios).",
    )
    parser.add_argument(
        "--town",
        help="Override town name from the route XML.",
    )
    parser.add_argument(
        "--vehicle-targets",
        type=int,
        nargs="+",
        default=[0, 5, 20],
        help="Vehicle target counts to test (space separated list).",
    )
    parser.add_argument(
        "--pedestrian-targets",
        type=int,
        nargs="+",
        default=[0, 10, 40],
        help="Pedestrian target counts (must match length of vehicle targets).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of times to respawn each (vehicle, pedestrian) pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed passed to the spawn helpers (iteration index is added on top).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="CARLA server host (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000).",
    )
    parser.add_argument(
        "--fixed-dt",
        type=float,
        default=0.05,
        help="Simulator fixed delta time to use while sampling spawns.",
    )
    parser.add_argument(
        "--reload-world-between-pairs",
        action="store_true",
        help="Reload the CARLA world before each (vehicles, pedestrians) target pair to minimize divergence.",
    )
    return parser.parse_args()


def _resolve_path(path_value, *, config_dir=None):
    if os.path.isabs(path_value):
        return path_value

    candidates = []
    if config_dir:
        candidates.append(os.path.abspath(os.path.join(config_dir, path_value)))
    candidates.append(os.path.abspath(os.path.join(CURRENT_DIR, path_value)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def _route_specs_from_config(config_path, route_names):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    experiments = config.get("experiments") or []
    if not experiments:
        raise ValueError(f"No experiments found in route config {config_path}")

    config_dir = os.path.dirname(config_path)
    by_name = {}
    for experiment in experiments:
        for route_cfg in experiment.get("routes") or []:
            name = route_cfg.get("name")
            route_path = route_cfg.get("route")
            if not name or not route_path:
                continue
            if name in by_name:
                continue
            by_name[name] = {
                "name": name,
                "route": _resolve_path(route_path, config_dir=config_dir),
                "route_id": route_cfg.get("route_id"),
            }

    selected = []
    for route_name in route_names:
        if route_name not in by_name:
            available = ", ".join(sorted(by_name.keys())) or "(none)"
            raise ValueError(
                f"Route name '{route_name}' not found in {config_path}. Available route names: {available}"
            )
        selected.append(by_name[route_name])
    return selected


def resolve_route_specs(args):
    if args.route:
        route_path = os.path.abspath(args.route)
        return [
            {
                "name": os.path.basename(route_path),
                "route": route_path,
                "route_id": args.route_id,
            }
        ]

    config_path = os.path.abspath(args.routes_config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Routes config {config_path} not found")
    return _route_specs_from_config(config_path, args.route_names)


def load_route_config(route_xml, scenario_json):
    indexer = LegacyRouteIndexer(route_xml, scenario_json, 1)
    config = indexer.next()
    if not config or not getattr(config, "trajectory", None):
        raise RuntimeError(f"Route file {route_xml} did not yield any waypoints via RouteIndexer")
    return config


def spawn_ego(world, route_config):
    bp_lib = world.get_blueprint_library()
    matches = bp_lib.filter("vehicle.tesla.model3")
    if not matches:
        raise RuntimeError("vehicle.tesla.model3 blueprint not found in the CARLA library")
    ego_bp = matches[0]
    ego_bp.set_attribute("role_name", "ego")
    start_location = route_config.trajectory[0]
    ego_waypoint = world.get_map().get_waypoint(
        start_location, project_to_road=True, lane_type=carla.LaneType.Any
    )
    ego_spawn = ego_waypoint.transform
    ego_spawn.location.z += 0.5

    ego = None
    for _ in range(5):
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego:
            break
        ego_spawn.location.z += 0.2
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle at the designated route start.")
    return ego


def summarize(samples):
    if not samples:
        return "0 (no samples)"
    avg = statistics.mean(samples)
    if len(samples) > 1:
        std = statistics.pstdev(samples)
        return f"{avg:.2f} ± {std:.2f} (min={min(samples)}, max={max(samples)})"
    return f"{samples[0]} (single sample)"


def main():
    args = parse_args()
    route_specs = resolve_route_specs(args)
    for route_spec in route_specs:
        if not os.path.isfile(route_spec["route"]):
            raise FileNotFoundError(f"Route file {route_spec['route']} not found")

    scenario_json = os.path.abspath(args.scenario_json)
    if not os.path.isfile(scenario_json):
        raise FileNotFoundError(f"Scenario JSON {scenario_json} not found")

    if len(args.vehicle_targets) != len(args.pedestrian_targets):
        raise ValueError("--vehicle-targets and --pedestrian-targets must have the same length")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    tm_port = DEFAULT_TM_PORT

    results = defaultdict(list)
    pairs = list(zip(args.vehicle_targets, args.pedestrian_targets))

    for route_idx, route_spec in enumerate(route_specs, start=1):
        route_label = route_spec["name"]
        route_xml = route_spec["route"]
        route_id = route_spec.get("route_id")
        if route_id is not None:
            route_xml = _materialize_route_subset(route_xml, route_id)

        preview_config = load_route_config(route_xml, scenario_json)
        route_town = args.town or getattr(preview_config, "town", None)
        if not route_town or route_town == "_":
            raise RuntimeError(
                f"Route '{route_label}' does not define a town; provide --town to override."
            )

        print(
            f"\n=== Route {route_idx}/{len(route_specs)}: {route_label} "
            f"(town={route_town}, route_id={route_id}) ==="
        )

        world = _load_world_and_wait(client, route_town)
        tm = client.get_trafficmanager(tm_port)

        def prepare_world(force_reload=False):
            nonlocal world, tm
            if force_reload:
                world = _load_world_and_wait(client, route_town)
            else:
                world = client.get_world()
                try:
                    world.wait_for_tick()
                except RuntimeError:
                    pass
            _apply_route_weather(world, preview_config)
            tm = client.get_trafficmanager(tm_port)
            make_sync(world, tm, fixed_dt=args.fixed_dt)

        for pair_idx, (veh_target, ped_target) in enumerate(pairs, start=1):
            print(
                f"\n=== Target pair {pair_idx}/{len(pairs)} -> "
                f"vehicles={veh_target}, pedestrians={ped_target} ==="
            )
            for iteration in range(args.iterations):
                seed_offset = args.seed + iteration

                force_reload = args.reload_world_between_pairs or (pair_idx == 1 and iteration == 0)
                prepare_world(force_reload=force_reload)

                ego = None
                vehicles = []
                walkers = []
                walker_controllers = []
                try:
                    ego = spawn_ego(world, preview_config)
                    world.tick()
                    vehicles = spawn_tm_vehicles(client, world, tm_port, n_vehicles=veh_target, seed=seed_offset)
                    walkers, walker_controllers = spawn_walkers(
                        client, world, n_walkers=ped_target, seed=seed_offset
                    )
                    veh_real = len(vehicles)
                    ped_real = len(walkers)
                    results[(route_label, veh_target, ped_target)].append((veh_real, ped_real))
                    print(
                        f"  Iter {iteration + 1}/{args.iterations}: spawned {veh_real} vehicles "
                        f"and {ped_real} pedestrians (seed={seed_offset})"
                    )
                finally:
                    for controller in walker_controllers:
                        try:
                            controller.stop()
                        except RuntimeError:
                            pass
                    destroy(walkers)
                    destroy(walker_controllers)
                    destroy(vehicles)
                    if ego is not None:
                        destroy([ego])
                    settings = world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    world.apply_settings(settings)
                    tm.set_synchronous_mode(False)

                # Prepare for the next measurement when reloading between pairs
                if args.reload_world_between_pairs:
                    continue

    print("\n=== Summary ===")
    for route_spec in route_specs:
        route_label = route_spec["name"]
        for pair in pairs:
            samples = results[(route_label, pair[0], pair[1])]
            veh_samples = [s[0] for s in samples]
            ped_samples = [s[1] for s in samples]
            print(
                f"{route_label} targets {pair[0]:>3}/{pair[1]:>3} -> "
                f"vehicles: {summarize(veh_samples)} | pedestrians: {summarize(ped_samples)}"
            )


if __name__ == "__main__":
    main()
