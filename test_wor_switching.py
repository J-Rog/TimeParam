import argparse
import math
import os
import sys
from dataclasses import dataclass

import carla

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

from wor_runner import TimingAdapter, run_sweep

DEFAULT_CONTROL_LATENCY = 0.10
DEFAULT_VEHICLE_DENSITY = 20
DEFAULT_PEDESTRIAN_DENSITY = 50


@dataclass
class ResolutionProfile:
    name: str
    wide_scale: float
    narr_scale: float
    latency_seconds: float
    latency_steps: int

    @classmethod
    def build(cls, name, wide_scale, narr_scale, latency_seconds, fixed_dt):
        if latency_seconds < 0:
            raise ValueError(f"Latency for profile '{name}' must be non-negative")
        if latency_seconds == 0:
            return cls(name, wide_scale, narr_scale, 0.0, 0)
        steps = max(1, int(round(latency_seconds / fixed_dt)))
        actual_latency = steps * fixed_dt
        return cls(name, wide_scale, narr_scale, actual_latency, steps)


class MultiResolutionController:
    """Keeps the WoR agent aligned with the active resolution/latency profile."""

    def __init__(self, pcla, lane_profile: ResolutionProfile, traffic_light_profile: ResolutionProfile):
        self._pcla = pcla
        self._profiles = {
            'lane': lane_profile,
            'traffic_light': traffic_light_profile,
        }
        self._current = None

    def profile_for_mode(self, mode):
        key = 'traffic_light' if mode == 'traffic_light' else 'lane'
        profile = self._profiles[key]
        self._apply_profile(profile)
        return profile

    def _apply_profile(self, profile: ResolutionProfile):
        if self._current == profile.name:
            return
        agent = getattr(self._pcla, "agent_instance", None)
        if not agent:
            self._current = profile.name
            return
        changed = False
        if hasattr(agent, "wide_scale") and not math.isclose(agent.wide_scale, profile.wide_scale, rel_tol=1e-3, abs_tol=1e-3):
            agent.wide_scale = profile.wide_scale
            changed = True
        if hasattr(agent, "narr_scale") and not math.isclose(agent.narr_scale, profile.narr_scale, rel_tol=1e-3, abs_tol=1e-3):
            agent.narr_scale = profile.narr_scale
            changed = True
        if changed:
            print(
                f"[MultiRes] Switched to profile '{profile.name}': "
                f"wide_scale={profile.wide_scale:.2f}, narr_scale={profile.narr_scale:.2f}, "
                f"latency={profile.latency_seconds * 1000:.0f} ms"
            )
        self._current = profile.name


class TrafficLightModeManager:
    """Detects when to switch to the high-resolution profile near traffic lights."""

    def __init__(self, scorer, activation_distance=5.0):
        self._scorer = scorer
        self._world = scorer._world
        self._map = self._world.get_map()
        self.activation_distance = activation_distance
        self._light_map = self._build_light_map()
        self._active_light_id = None
        self._activation_count = 0

    def _build_light_map(self):
        actors = self._world.get_actors().filter("traffic.traffic_light*")
        return {actor.id: actor for actor in actors}

    def _refresh_light_if_needed(self, light_id):
        if light_id and light_id not in self._light_map:
            self._light_map = self._build_light_map()

    def update(self, vehicle):
        vehicle_loc, vehicle_forward, vehicle_wp = self._vehicle_pose(vehicle)

        if self._active_light_id is not None:
            self._refresh_light_if_needed(self._active_light_id)
            light = self._light_map.get(self._active_light_id)
            if light:
                trigger_distance = self._distance_to_light_trigger(vehicle_loc, light)
                if trigger_distance is not None and trigger_distance <= self.activation_distance + 10:
                    return "traffic_light", light
            self._active_light_id = None

        light, distance = self._nearest_light_from_pose(vehicle_loc, vehicle_forward, vehicle_wp)
        if light:
            trigger_distance = self._distance_to_light_trigger(vehicle_loc, light)
            if trigger_distance is not None and trigger_distance <= self.activation_distance:
                self._active_light_id = light.id
                self._activation_count += 1
                return "traffic_light", light
        return "lane", None

    def activation_count(self):
        return self._activation_count

    def current_light_id(self):
        return self._active_light_id

    def _vehicle_pose(self, vehicle):
        if not self._light_map:
            self._light_map = self._build_light_map()
        vehicle_tf = vehicle.get_transform()
        vehicle_loc = vehicle_tf.location
        vehicle_forward = vehicle_tf.get_forward_vector()
        try:
            vehicle_wp = self._map.get_waypoint(
                vehicle_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            vehicle_wp = None
        return vehicle_loc, vehicle_forward, vehicle_wp

    def _nearest_light(self, vehicle):
        vehicle_loc, vehicle_forward, vehicle_wp = self._vehicle_pose(vehicle)
        return self._nearest_light_from_pose(vehicle_loc, vehicle_forward, vehicle_wp)

    def _nearest_light_from_pose(self, vehicle_loc, vehicle_forward, vehicle_wp):
        best_light = None
        best_distance = None
        for light in self._light_map.values():
            distance = self._distance_to_stopline(vehicle_loc, vehicle_forward, vehicle_wp, light)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_light = light
        return best_light, best_distance

    def _distance_to_stopline(self, vehicle_loc, vehicle_forward, vehicle_wp, light):
        stop_lines = self._scorer._get_stop_waypoints(light)
        if not stop_lines or vehicle_wp is None:
            return None
        min_distance = None
        for wp in stop_lines:
            if wp.road_id != vehicle_wp.road_id or wp.lane_id != vehicle_wp.lane_id:
                continue
            stop_loc = wp.transform.location
            rel_x = stop_loc.x - vehicle_loc.x
            rel_y = stop_loc.y - vehicle_loc.y
            rel_z = stop_loc.z - vehicle_loc.z
            distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
            if min_distance is None or distance < min_distance:
                min_distance = distance
        return min_distance

    @staticmethod
    def _distance_to_light_trigger(vehicle_loc, light):
        if not light:
            return None
        try:
            trigger = light.trigger_volume
            base_tf = light.get_transform()
            trigger_loc = base_tf.transform(trigger.location)
        except RuntimeError:
            return None
        rel_x = trigger_loc.x - vehicle_loc.x
        rel_y = trigger_loc.y - vehicle_loc.y
        rel_z = trigger_loc.z - vehicle_loc.z
        return math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)


class MultiResAdapter(TimingAdapter):
    def __init__(self, args, control_latency, fixed_dt):
        super().__init__(control_latency, fixed_dt)
        self._args = args
        self.lane_profile = ResolutionProfile.build(
            "lane_following",
            args.lane_wide_scale,
            args.lane_narr_scale,
            control_latency,
            fixed_dt,
        )
        self.base_latency_steps = self.lane_profile.latency_steps
        if abs(self.lane_profile.latency_seconds - control_latency) > 1e-6:
            print(
                f"Requested lane latency {control_latency:.3f}s rounded to "
                f"{self.lane_profile.latency_seconds:.3f}s (steps={self.lane_profile.latency_steps})."
            )

        self.traffic_profile = ResolutionProfile.build(
            "traffic_light_focus",
            args.traffic_light_wide_scale,
            args.traffic_light_narr_scale,
            args.traffic_light_latency,
            fixed_dt,
        )
        if abs(self.traffic_profile.latency_seconds - args.traffic_light_latency) > 1e-6:
            print(
                f"Requested traffic-light latency {args.traffic_light_latency:.3f}s rounded to "
                f"{self.traffic_profile.latency_seconds:.3f}s (steps={self.traffic_profile.latency_steps})."
            )

        self.current_mode = "lane"
        self.current_profile = self.lane_profile
        self.latency_steps = self.current_profile.latency_steps
        self.control_period_steps = self.latency_steps
        self.mode_manager = None
        self.multires_controller = None

    def attach(self, pcla, scorer, args):
        self.mode_manager = TrafficLightModeManager(
            scorer,
            activation_distance=self._args.traffic_light_distance,
        )
        self.multires_controller = MultiResolutionController(pcla, self.lane_profile, self.traffic_profile)
        self.current_mode = "lane"
        self.current_profile = self.multires_controller.profile_for_mode(self.current_mode)
        self.latency_steps = self.current_profile.latency_steps
        self.control_period_steps = self.latency_steps
        print(
            f"Multi-res setup -> lane: wide/narr {self.lane_profile.wide_scale:.2f}/{self.lane_profile.narr_scale:.2f} "
            f"@ {self.lane_profile.latency_seconds*1000:.0f} ms | "
            f"traffic-light: wide/narr {self.traffic_profile.wide_scale:.2f}/{self.traffic_profile.narr_scale:.2f} "
            f"@ {self.traffic_profile.latency_seconds*1000:.0f} ms (trigger {self._args.traffic_light_distance:.1f} m)"
        )

    def update(self, ego):
        if self.mode_manager and self.multires_controller:
            mode, _ = self.mode_manager.update(ego)
            self.current_mode = mode
            self.current_profile = self.multires_controller.profile_for_mode(mode)
        else:
            self.current_mode = "lane"
            self.current_profile = self.lane_profile
        self.latency_steps = self.current_profile.latency_steps
        self.control_period_steps = self.latency_steps

    def overlay_lines(self):
        lines = []
        if not self.current_profile:
            return lines
        lines.append(
            f"MultiRes mode: {self.current_mode} | lat {self.current_profile.latency_seconds*1000:.0f} ms "
            f"| wide/narr {self.current_profile.wide_scale:.2f}/{self.current_profile.narr_scale:.2f}"
        )
        tl_id = self.mode_manager.current_light_id() if self.mode_manager else None
        lines.append(
            f"Traffic-light target: {tl_id if tl_id is not None else '-'} "
            f"| triggers {self.mode_manager.activation_count() if self.mode_manager else 0}"
        )
        return lines

    def summary_extra(self):
        if not self.mode_manager:
            return ""
        return f"TL switches {self.mode_manager.activation_count()}"

    def result_fields(self, control_latency):
        tl_switches = self.mode_manager.activation_count() if self.mode_manager else 0
        return {
            'control_period': float(control_latency),
            'control_latency': float(control_latency),
            'control_steps': self.base_latency_steps,
            'latency_steps': self.base_latency_steps,
            'traffic_light_triggers': tl_switches,
            'traffic_light_latency': float(self._args.traffic_light_latency),
            'traffic_light_latency_steps': self.traffic_profile.latency_steps,
            'traffic_light_distance': float(self._args.traffic_light_distance),
            'lane_wide_scale': self.lane_profile.wide_scale,
            'lane_narr_scale': self.lane_profile.narr_scale,
            'traffic_light_wide_scale': self.traffic_profile.wide_scale,
            'traffic_light_narr_scale': self.traffic_profile.narr_scale,
        }


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
        "--traffic-light-latency",
        type=float,
        default=0.15,
        help="Control latency (seconds) for the high-resolution profile near traffic lights.",
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
        "--traffic-light-distance",
        type=float,
        default=5.0,
        help="Distance (m) to a stop line that triggers the traffic-light profile.",
    )
    parser.add_argument(
        "--lane-wide-scale",
        type=float,
        default=0.75,
        help="Wide camera resize factor during lane following.",
    )
    parser.add_argument(
        "--lane-narr-scale",
        type=float,
        default=0.75,
        help="Narrow camera resize factor during lane following.",
    )
    parser.add_argument(
        "--traffic-light-wide-scale",
        type=float,
        default=1.0,
        help="Wide camera resize factor when focusing on traffic lights.",
    )
    parser.add_argument(
        "--traffic-light-narr-scale",
        type=float,
        default=1.0,
        help="Narrow camera resize factor when focusing on traffic lights.",
    )
    parser.add_argument(
        "--route-id",
        type=int,
        help="Restrict the provided route XML to this <route id>. When unset, the first route is used.",
    )
    return parser.parse_args()


def _timing_factory(args, control_latency, fixed_dt):
    return MultiResAdapter(args, control_latency, fixed_dt)


def main():
    args = parse_args()
    if args.traffic_light_latency < 0:
        raise ValueError("Traffic-light latency must be non-negative")
    if args.traffic_light_distance <= 0:
        raise ValueError("Traffic-light trigger distance must be positive")
    run_sweep(
        args,
        _timing_factory,
        DEFAULT_CONTROL_LATENCY,
        DEFAULT_VEHICLE_DENSITY,
        DEFAULT_PEDESTRIAN_DENSITY,
    )


if __name__ == "__main__":
    main()
