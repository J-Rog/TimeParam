import carla
import csv
import os
import random
import sys
import time
import math
import copy
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from types import SimpleNamespace
try:
    import pygame
except ImportError:
    pygame = None

from PCLA import PCLA
from leaderboard_codes.route_indexer import RouteIndexer as LegacyRouteIndexer
from leaderboard_codes.route_manipulation import interpolate_trajectory

# Allow importing the Leaderboard/scenario_runner utilities that ship in WorldOnRails
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

try:
    from leaderboard_codes.utils.statistics_manager import StatisticsManager
    from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Unable to import Leaderboard / ScenarioRunner modules. "
        "Ensure the leaderboard + scenario_runner folders exist in this workspace "
        "and CARLA_ROOT/PythonAPI/carla is accessible."
    ) from exc

DEFAULT_TM_PORT = 8000
# Defaults used when no sweep values are provided.
DEFAULT_CONTROL_LATENCY = 0.05
DEFAULT_VEHICLE_DENSITY = 20
DEFAULT_PEDESTRIAN_DENSITY = 50


class _CriterionBucket:
    """Lightweight container that mimics ScenarioRunner criteria objects."""

    def __init__(self):
        self.events = []


class _DummyScenario:
    """StatisticsManager expects a scenario exposing route data and criteria."""

    def __init__(self, route, criteria):
        self.route = route
        self.timeout_node = SimpleNamespace(timeout=False)
        self._criteria = criteria

    def get_criteria(self):
        return self._criteria


class _RouteCompletionTracker:
    """Simplified copy of RouteCompletionTest that only tracks percentage."""

    WINDOW_SIZE = 4

    def __init__(self, route_transforms):
        self.transforms = [tf for tf, _ in route_transforms]
        self.route_length = len(self.transforms)
        self.current_index = 0
        self._accumulated = self._build_accumulated()
        self._total_distance = self._accumulated[-1] if self._accumulated else 1.0
        self.percentage = 0.0

    def _build_accumulated(self):
        accum = []
        total = 0.0
        if not self.transforms:
            return accum
        prev = self.transforms[0].location
        for tf in self.transforms:
            loc = tf.location
            total += loc.distance(prev)
            accum.append(total)
            prev = loc
        return accum

    def update(self, vehicle_location):
        if not self.transforms:
            return self.percentage
        window_end = min(self.current_index + self.WINDOW_SIZE + 1, self.route_length)
        for idx in range(self.current_index, window_end):
            ref_tf = self.transforms[idx]
            forward = ref_tf.get_forward_vector()
            rel_vec = vehicle_location - ref_tf.location
            dot = rel_vec.x * forward.x + rel_vec.y * forward.y + rel_vec.z * forward.z
            if dot > 0.0:
                self.current_index = idx

        if self._total_distance > 0:
            raw = self._accumulated[self.current_index] / self._total_distance * 100.0
            self.percentage = round(min(raw, 100.0), 2)
        return self.percentage


class LeaderboardLikeScorer:
    """
    Lightweight adapter around StatisticsManager so test_wor.py can emit driving scores.
    """

    COLLISION_COOLDOWN_FRAMES = 20

    def __init__(self, world, route_xml, scenario_json=None):
        if not os.path.isfile(route_xml):
            raise FileNotFoundError(f"Route XML {route_xml} not found")
        scenario_json = scenario_json or os.path.join(CURRENT_DIR, "leaderboard_codes", "no_scenarios.json")
        if not os.path.isfile(scenario_json):
            raise FileNotFoundError(f"Scenario JSON {scenario_json} not found")

        self._world = world
        self._map = world.get_map()
        self._stopline_cache = {}

        self.route_indexer = LegacyRouteIndexer(route_xml, scenario_json, 1)
        self.route_config = self.route_indexer.next()
        if self.route_config is None:
            raise RuntimeError("RouteIndexer returned no routes")
        self.route_index = self.route_config.index
        self.route_id = f"{self.route_config.name}_rep{self.route_config.repetition_index}"

        stats_dir = os.path.join(CURRENT_DIR, ".leaderboard_tmp")
        os.makedirs(stats_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(stats_dir, "simulation_results.json")
        self.debug_path = os.path.join(stats_dir, "live_results.txt")
        self.statistics = StatisticsManager(self.checkpoint_path, self.debug_path)
        try:
            self.statistics.clear_records()
        except Exception:
            pass
        self.statistics.save_progress(0, self.route_indexer.total)
        self.statistics.create_route_data(self.route_id, self.route_index)

        _, dense_route = interpolate_trajectory(world, self.route_config.trajectory)

        self.criteria = [_CriterionBucket()]
        self.scenario = _DummyScenario(dense_route, self.criteria)
        self.statistics.set_scenario(self.scenario)

        self.route_tracker = _RouteCompletionTracker(dense_route)
        self.route_percent = 0.0

        self._frame = 0
        self.route_event = TrafficEvent(TrafficEventType.ROUTE_COMPLETION, frame=0)
        self.criteria[0].events.append(self.route_event)
        self.completed_event_sent = False
        self._lane_event = None
        self._lane_invasion_marks = 0
        self._lane_percent = 0.0
        self._final_scores = None

        self._collision_counts = {
            'vehicle': 0,
            'pedestrian': 0,
            'static': 0,
        }
        self._collision_block_active = False
        self._red_light_events = 0
        self._last_red_light_id = None
        self._collision_last_frames = {}
        self._collision_unique_keys = set()

        self._wallclock_start = time.time()
        self.sim_time = 0.0


    def on_collision(self, event):
        other = event.other_actor
        frame = getattr(event, "frame", self._frame)
        key = other.id if other else 'static'
        last_frame = self._collision_last_frames.get(key, -1)
        if last_frame >= 0 and frame - last_frame < self.COLLISION_COOLDOWN_FRAMES:
            return
        self._collision_last_frames[key] = frame

        unique_key = key
        if other is None:
            actor_loc = None
            if hasattr(event, 'actor') and event.actor:
                try:
                    actor_loc = event.actor.get_transform().location
                except RuntimeError:
                    actor_loc = None
            if actor_loc:
                unique_key = (
                    'static',
                    round(actor_loc.x, 1),
                    round(actor_loc.y, 1),
                    round(actor_loc.z, 1),
                )
        if unique_key in self._collision_unique_keys:
            return
        self._collision_unique_keys.add(unique_key)
        self._collision_block_active = True

        if other and 'walker.' in other.type_id:
            event_type = TrafficEventType.COLLISION_PEDESTRIAN
            self._collision_counts['pedestrian'] += 1
        elif other and 'vehicle.' in other.type_id:
            event_type = TrafficEventType.COLLISION_VEHICLE
            self._collision_counts['vehicle'] += 1
        else:
            event_type = TrafficEventType.COLLISION_STATIC
            self._collision_counts['static'] += 1
        other_name = other.type_id if other else 'static.world'
        traffic_event = TrafficEvent(event_type, frame=frame, message=f"Collision with {other_name}")
        self.criteria[0].events.append(traffic_event)

    def on_lane_invasion(self, event):
        markings_crossed = len(event.crossed_lane_markings)
        if markings_crossed <= 0:
            return
        self._lane_invasion_marks += markings_crossed
        if self._lane_event is None:
            frame = getattr(event, "frame", self._frame)
            self._lane_event = TrafficEvent(TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION, frame=frame)
            self.criteria[0].events.append(self._lane_event)
        else:
            frame = getattr(event, "frame", self._frame)
            self._lane_event.set_frame(frame)
        lane_percentage = min(100.0, self._lane_invasion_marks * 2.5)
        self._lane_event.set_dict({'crossings': self._lane_invasion_marks, 'percentage': lane_percentage})
        self._lane_percent = float(self._lane_invasion_marks)

    def tick(self, vehicle, dt, frame):
        self._frame = frame
        self.sim_time += dt
        percentage = self.route_tracker.update(vehicle.get_transform().location)
        self.route_percent = percentage
        self.route_event.set_dict({'route_completed': percentage})
        self.route_event.set_frame(frame)
        if percentage >= 99.0 and not self.completed_event_sent:
            completion_event = TrafficEvent(TrafficEventType.ROUTE_COMPLETED, frame=frame, message="Route completed")
            self.criteria[0].events.append(completion_event)
            self.completed_event_sent = True
        self._check_red_light(vehicle, frame)

    def finalize(self):
        duration_system = time.time() - self._wallclock_start
        failure_message = "" if self.completed_event_sent else "Manual stop"
        entry_status = 'Finished' if self.completed_event_sent else 'Crashed'
        self.statistics.save_entry_status(entry_status)
        self.statistics.save_progress(self.route_index + 1, self.route_indexer.total)
        self.statistics.compute_route_statistics(
            self.route_index,
            duration_time_system=duration_system,
            duration_time_game=self.sim_time,
            failure_message=failure_message,
        )
        try:
            record = self.statistics._results.checkpoint.records[self.route_index]
            self._final_scores = record.scores
        except (AttributeError, IndexError):
            record = None
        return record

    def get_overlay_lines(self):
        lines = []
        lines.append(
            f"Route {self.route_percent:5.1f}% | Collisions v/p/s "
            f"{self._collision_counts['vehicle']}/"
            f"{self._collision_counts['pedestrian']}/"
            f"{self._collision_counts['static']}"
        )
        lane_text = f"Lane crossings {int(self._lane_invasion_marks)}"
        completion = "Reached" if self.completed_event_sent else "In progress"
        lines.append(f"{lane_text} | Red lights {self._red_light_events} | Target {completion}")
        return lines

    def elapsed_times(self):
        """Return tuple (sim_time_seconds, wallclock_seconds_since_start)."""
        return self.sim_time, time.time() - self._wallclock_start

    def get_lane_crossings(self):
        return int(self._lane_invasion_marks)

    def get_collision_counts(self):
        return dict(self._collision_counts)

    def get_red_light_events(self):
        return self._red_light_events

    def collision_block_active(self):
        return self._collision_block_active

    def clear_collision_block_if_recovered(self, speed_ms, recovery_speed=1.0):
        if self._collision_block_active and speed_ms > recovery_speed:
            self._collision_block_active = False

    def _check_red_light(self, vehicle, frame):
        light = vehicle.get_traffic_light()
        if not light:
            self._last_red_light_id = None
            return
        if light.get_state() != carla.TrafficLightState.Red:
            self._last_red_light_id = None
            return
        if self._last_red_light_id == light.id:
            return

        speed = vehicle.get_velocity().length()
        if speed < 0.3:
            return

        if hasattr(light, "check_if_vehicle_has_crossed_stop_line"):
            try:
                if light.check_if_vehicle_has_crossed_stop_line(vehicle):
                    self._register_red_light(light, frame)
                return
            except RuntimeError:
                pass

        stop_lines = self._get_stop_waypoints(light)
        if not stop_lines:
            return

        tail_close, tail_far = self._vehicle_tail_segment(vehicle)
        tail_wp = self._map.get_waypoint(tail_far, project_to_road=True, lane_type=carla.LaneType.Driving)
        vehicle_forward = vehicle.get_transform().get_forward_vector()

        for wp in stop_lines:
            if tail_wp is None or tail_wp.road_id != wp.road_id or tail_wp.lane_id != wp.lane_id:
                continue
            wp_forward = wp.transform.get_forward_vector()
            dot = vehicle_forward.x * wp_forward.x + vehicle_forward.y * wp_forward.y + vehicle_forward.z * wp_forward.z
            if dot <= 0:
                continue

            left_pt, right_pt = self._stop_line_segment(wp)
            if self._segments_intersect(tail_close, tail_far, left_pt, right_pt):
                self._register_red_light(light, frame)
                break

    def _register_red_light(self, light, frame):
        self._red_light_events += 1
        event = TrafficEvent(
            TrafficEventType.TRAFFIC_LIGHT_INFRACTION,
            frame=frame,
            message=f"Red light at id {light.id}",
        )
        event.set_dict({'id': light.id, 'location': light.get_transform().location})
        self.criteria[0].events.append(event)
        self._last_red_light_id = light.id

    def _vehicle_tail_segment(self, vehicle):
        tf = vehicle.get_transform()
        extent = vehicle.bounding_box.extent.x
        tail_close = tf.transform(carla.Location(x=-0.8 * extent))
        tail_far = tf.transform(carla.Location(x=-(extent + 1.5)))
        return tail_close, tail_far

    def _stop_line_segment(self, waypoint):
        yaw = waypoint.transform.rotation.yaw
        lane_width = waypoint.lane_width
        base = waypoint.transform.location
        left_vec = self._rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw + 90)
        right_vec = self._rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw - 90)
        left_pt = base + carla.Location(left_vec)
        right_pt = base + carla.Location(right_vec)
        return left_pt, right_pt

    def _get_stop_waypoints(self, traffic_light):
        cached = self._stopline_cache.get(traffic_light.id)
        if cached is not None:
            return cached
        wps = []
        if hasattr(traffic_light, "get_stop_waypoints"):
            try:
                wps = traffic_light.get_stop_waypoints()
            except RuntimeError:
                wps = []
        if not wps:
            wps = self._compute_stop_waypoints(traffic_light)
        self._stopline_cache[traffic_light.id] = wps
        return wps

    def _compute_stop_waypoints(self, traffic_light):
        trigger = traffic_light.trigger_volume
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(trigger.location)
        area_ext = trigger.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)

        area_points = []
        for x in x_values:
            point = self._rotate_point(carla.Vector3D(x, 0, trigger.extent.z), base_rot)
            area_points.append(area_loc + carla.Location(x=point.x, y=point.y))

        initial_wps = []
        for pt in area_points:
            wp = self._map.get_waypoint(pt, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not initial_wps or initial_wps[-1].road_id != wp.road_id or initial_wps[-1].lane_id != wp.lane_id:
                initial_wps.append(wp)

        result_wps = []
        for wp in initial_wps:
            temp_wp = wp
            while temp_wp and not temp_wp.is_intersection:
                next_wp = temp_wp.next(0.5)
                if next_wp and not next_wp[0].is_intersection:
                    temp_wp = next_wp[0]
                else:
                    break
            result_wps.append(temp_wp)
        return result_wps

    @staticmethod
    def _rotate_point(point, angle):
        rad = math.radians(angle)
        x_ = math.cos(rad) * point.x - math.sin(rad) * point.y
        y_ = math.sin(rad) * point.x + math.cos(rad) * point.y
        return carla.Vector3D(x_, y_, point.z)

    @staticmethod
    def _segments_intersect(p1, p2, q1, q2):
        def ccw(a, b, c):
            return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


class OverlayHUD:
    """Small pygame window that displays overlay text independent of the spectator."""

    def __init__(self, width=560, height=200):
        self.enabled = False
        if pygame is None:
            print("Pygame not available; HUD overlay disabled.")
            return
        try:
            pygame.display.init()
            pygame.font.init()
            self.surface = pygame.display.set_mode((width, height))
            pygame.display.set_caption("PCLA HUD")
            self.font = pygame.font.SysFont("monospace", 18)
            self.enabled = True
        except pygame.error as exc:
            print(f"Failed to init pygame HUD ({exc}); overlay disabled.")
            self.enabled = False

    def update(self, lines):
        if not self.enabled:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.enabled = False
                pygame.display.quit()
                return
        self.surface.fill((10, 10, 10))
        if not lines:
            lines = ["No data"]
        for idx, line in enumerate(lines):
            text = self.font.render(line, True, (0, 191, 255))
            self.surface.blit(text, (10, 10 + idx * 22))
        pygame.display.flip()

    def close(self):
        if self.enabled:
            pygame.display.quit()
            pygame.font.quit()
            self.enabled = False


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


class SweepLogger:
    """CSV helper that appends sweep rows when a file path is provided."""

    FIELDNAMES = [
        "timestamp",
        "route_id",
        "route_xml",
        "agent",
        "control_period",
        "control_latency",
        "control_steps",
        "latency_steps",
        "vehicle_density",
        "pedestrian_density",
        "status",
        "route_completion",
        "sim_time",
        "wall_time",
        "lane_crossings",
        "collisions_vehicle",
        "collisions_pedestrian",
        "collisions_static",
        "red_lights",
        "message",
    ]

    def __init__(self, csv_path):
        self._path = csv_path
        self._file = None
        self._writer = None
        if not csv_path:
            return
        directory = os.path.dirname(csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_exists = os.path.isfile(csv_path)
        self._file = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if not file_exists or os.path.getsize(csv_path) == 0:
            self._writer.writeheader()

    def log(self, row):
        if not self._writer:
            return
        sanitized = {key: row.get(key) for key in self.FIELDNAMES}
        self._writer.writerow(sanitized)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

# ---------- helpers ----------

def make_sync(world, traffic_manager, fixed_dt=0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)
    return settings  # keep original to restore later if you want

def spawn_tm_vehicles(client, world, tm_port, n_vehicles=60, seed=42):
    import random
    random.seed(seed)

    bp_lib = world.get_blueprint_library()
    blacklist = {"carlamotors", "carlacola", "bus", "mitsubishi", "ambulance"}
    v_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels").as_int()) >= 4
        and not any(token in bp.id for token in blacklist)
    ]

    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    batch = []
    for tr in spawns[:n_vehicles]:
        bp = random.choice(v_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        batch.append(
            carla.command.SpawnActor(bp, tr).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)
            )
        )

    resp = client.apply_batch_sync(batch, True)
    ids = [r.actor_id for r in resp if not r.error]
    vehicles = world.get_actors(ids)

    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    tm.set_global_distance_to_leading_vehicle(3.0)
    tm.global_percentage_speed_difference(0.0)  # global baseline
    tm.set_respawn_dormant_vehicles(True)

    # 0.9.16: per-vehicle speed diff is (actor, float)
    dormant_tracker = {}
    def nudge_vehicle(actor):
        actor.set_autopilot(False, tm_port)
        ctrl = carla.VehicleControl(throttle=0.5, brake=0.0)
        actor.apply_control(ctrl)
        world.tick()
        actor.set_autopilot(True, tm_port)

    for v in vehicles:
        tm.vehicle_percentage_speed_difference(v, random.uniform(-20.0, 10.0))
        dormant_tracker[v.id] = 0
        # Optional conservative behavior:
        # tm.set_percentage_ignore_lights(v, 0.0)
        # tm.distance_to_leading_vehicle(v, 2.5)
    def monitor():
        for actor_id in list(dormant_tracker.keys()):
            actor = world.get_actor(actor_id)
            if not actor:
                dormant_tracker.pop(actor_id, None)
                continue
            speed = actor.get_velocity().length()
            if speed < 0.5:
                dormant_tracker[actor_id] += 1
            else:
                dormant_tracker[actor_id] = 0
            if dormant_tracker[actor_id] > int(15 / (world.get_settings().fixed_delta_seconds or 0.05)):
                nudge_vehicle(actor)
                dormant_tracker[actor_id] = 0
    world._tm_monitor = monitor  # stash callable for main loop
    return vehicles


def spawn_walkers(client, world, n_walkers=80, seed=42):
    random.seed(seed)
    walker_bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    spawn_points = []
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_points.append(carla.Transform(loc))

    # spawn the pedestrians
    walker_batch = []
    for i in range(len(spawn_points)):
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("speed"):
            # let controller pick speed; still set a default
            bp.set_attribute("speed", str(bp.get_attribute("speed").recommended_values[1]))
        walker_batch.append(carla.command.SpawnActor(bp, spawn_points[i]))

    results = client.apply_batch_sync(walker_batch, True)
    walker_ids = [r.actor_id for r in results if not r.error]

    # spawn AI controllers for each walker
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    controller_batch = []
    for wid in walker_ids:
        controller_batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), wid))
    results_ctrl = client.apply_batch_sync(controller_batch, True)
    controller_ids = [r.actor_id for r in results_ctrl if not r.error]

    walkers = world.get_actors(walker_ids)
    controllers = world.get_actors(controller_ids)

    # start the controllers and set random destinations
    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.2 + random.random() * 0.6)  # ~1.2–1.8 m/s

    return walkers, controllers

def destroy(actors):
    try:
        for a in actors:
            if not a:
                continue
            if getattr(a, 'is_alive', False):
                a.destroy()
    except RuntimeError:
        pass


def destroy_sensors(sensors):
    for sensor in sensors:
        if not sensor:
            continue
        try:
            stop_fn = getattr(sensor, 'stop', None)
            if callable(stop_fn):
                stop_fn()
        except RuntimeError:
            pass
        try:
            destroy_fn = getattr(sensor, 'destroy', None)
            if callable(destroy_fn):
                destroy_fn()
        except RuntimeError:
            pass


def _load_world_and_wait(client, town):
    client.load_world(town)
    world = client.get_world()
    try:
        world.wait_for_tick()
    except RuntimeError:
        time.sleep(0.5)
    return world


def _apply_route_weather(world, route_config):
    weather = getattr(route_config, 'weather', None)
    if weather is None:
        return
    try:
        world.set_weather(weather)
    except RuntimeError as exc:
        print(f"Failed to apply route weather ({exc}); continuing with current weather settings.")


def _materialize_route_subset(route_xml, route_id):
    """
    Create a temporary route file that only contains the selected <route id>.
    """
    tree = ET.parse(route_xml)
    root = tree.getroot()
    target = None
    for route_elem in root.findall("route"):
        if route_elem.get("id") == str(route_id):
            target = copy.deepcopy(route_elem)
            break
    if target is None:
        raise ValueError(f"Route id {route_id} not found in {route_xml}")

    new_root = ET.Element(root.tag)
    for child in root:
        if child.tag != "route":
            new_root.append(copy.deepcopy(child))
    new_root.append(target)

    out_dir = os.path.join(CURRENT_DIR, "tmp", "routes")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(route_xml))[0]
    out_path = os.path.join(out_dir, f"{base}_route{route_id}.xml")
    ET.ElementTree(new_root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _clone_control(control):
    clone = carla.VehicleControl()
    clone.throttle = control.throttle
    clone.steer = control.steer
    clone.brake = control.brake
    clone.hand_brake = control.hand_brake
    clone.reverse = control.reverse
    clone.manual_gear_shift = control.manual_gear_shift
    clone.gear = control.gear
    return clone


# ---------- main ----------

def run_configuration(args, client, world, tm_port, route_xml, preview_config,
                      control_latency, vehicle_density, pedestrian_density):
    if control_latency < 0:
        raise ValueError("Control latency must be non-negative")

    FIXED_DT = 0.025
    MIN_CONTROL_PERIOD = 0.05
    STUCK_SPEED_THRESHOLD = 0.2  # m/s
    STUCK_TIME_SECONDS = 8.0
    MAX_LOW_SPEED_SECONDS = 40.0

    _apply_route_weather(world, preview_config)
    tm = client.get_trafficmanager(tm_port)

    if control_latency >= MIN_CONTROL_PERIOD:
        latency_steps = max(1, int(round(control_latency / FIXED_DT)))
        control_period_steps = latency_steps
    else:
        latency_steps = 0
        control_period_steps = max(1, int(round(MIN_CONTROL_PERIOD / FIXED_DT)))

    effective_control_latency = latency_steps * FIXED_DT
    effective_control_period = control_period_steps * FIXED_DT

    if control_latency == 0:
        print(
            f"Using zero-latency mode with control period {effective_control_period:.3f}s "
            f"(steps={control_period_steps})."
        )
    elif abs(effective_control_latency - control_latency) > 1e-6:
        print(
            f"Requested control latency {control_latency:.3f}s rounded to "
            f"{effective_control_latency:.3f}s (steps={latency_steps})."
        )

    dense_route_preview = interpolate_trajectory(world, preview_config.trajectory)[1]

    make_sync(world, tm, fixed_dt=FIXED_DT)

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    ego_bp.set_attribute("role_name", "ego")
    start_location = preview_config.trajectory[0]
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
        raise RuntimeError("Failed to spawn ego vehicle at the route start; ensure the area is clear")

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(carla.Location(x=-8, y=108, z=7),
                                            carla.Rotation(pitch=-19, yaw=0, roll=0)))

    def follow(veh):
        tf = veh.get_transform()
        cam_loc = tf.transform(carla.Location(x=-3.0, z=2.5))
        cam_rot = carla.Rotation(pitch=-12.0, yaw=tf.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def vehicle_overlay_lines(ego_actor, control, route_tracker, route_transforms, red_count):
        lines = []
        spd_ms = ego_actor.get_velocity().length()
        lines.append(
            f"Speed {spd_ms*3.6:5.1f} km/h | steer {control.steer:+.2f} "
            f"thr {control.throttle:.2f} brk {control.brake:.2f}"
        )
        tf = ego_actor.get_transform()
        lines.append(
            f"Pos ({tf.location.x:7.1f}, {tf.location.y:7.1f}, {tf.location.z:5.1f}) "
            f"Yaw {tf.rotation.yaw:6.1f}"
        )
        if route_tracker and route_transforms:
            idx = route_tracker.current_index
            tf_loc = tf.location
            for offset in (5, 15, 30):
                target_idx = min(idx + offset, len(route_transforms) - 1)
                wp_loc = route_transforms[target_idx][0].location
                dist = tf_loc.distance(wp_loc)
                lines.append(
                    f"Next +{offset:02d}: ({wp_loc.x:7.1f}, {wp_loc.y:7.1f}) "
                    f"dist={dist:5.1f}m"
                )
        else:
            lines.append("Route info unavailable")

        lines.append(f"Red-light infractions: {red_count}")
        return lines

    world.tick()

    vehicles = []
    walkers = []
    walker_controllers = []
    aux_sensors = []
    hud = OverlayHUD()
    pcla = None
    scorer = None
    status = 'aborted'
    message = 'manual_stop'

    try:
        vehicles = spawn_tm_vehicles(client, world, tm_port, n_vehicles=vehicle_density, seed=42)
        walkers, walker_controllers = spawn_walkers(client, world, n_walkers=pedestrian_density, seed=42)

        agent_name = args.agent
        pcla = PCLA(agent_name, ego, route_xml, client, agent_config_override=args.agent_config)
        scorer = LeaderboardLikeScorer(world, route_xml)

        collision_bp = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
        collision_sensor.listen(scorer.on_collision)
        aux_sensors.append(collision_sensor)

        lane_bp = bp_lib.find('sensor.other.lane_invasion')
        lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)
        lane_sensor.listen(scorer.on_lane_invasion)
        aux_sensors.append(lane_sensor)

        print(
            f"Spawned ego and {len(vehicles)} vehicles, {len(walkers)} walkers "
            f"(targets {vehicle_density}/{pedestrian_density})."
        )

        stuck_frames = 0
        low_speed_frames = 0
        pending_control = _clone_control(pcla.get_action())
        applied_control = carla.VehicleControl()
        if latency_steps == 0:
            applied_control = _clone_control(pending_control)
            next_apply_tick = control_period_steps
        else:
            next_apply_tick = latency_steps
        sim_ticks = 0
        terminal_message = ""

        while True:
            apply_before_tick = latency_steps == 0
            if apply_before_tick:
                # Zero-latency path: apply the most recent control before stepping the world.
                ego.apply_control(applied_control)

            world.tick()
            sim_ticks += 1
            monitor = getattr(world, "_tm_monitor", None)
            if monitor:
                monitor()
            snapshot = world.get_snapshot()
            frame = snapshot.frame if snapshot else 0
            scorer.tick(ego, FIXED_DT, frame)
            follow(ego)

            spd_ms = ego.get_velocity().length()
            scorer.clear_collision_block_if_recovered(spd_ms)
            low_speed_frames = low_speed_frames + 1 if spd_ms < STUCK_SPEED_THRESHOLD else 0
            if (
                spd_ms < STUCK_SPEED_THRESHOLD
                and (
                    (applied_control.throttle > 0.3 and applied_control.brake < 0.1)
                    or scorer.collision_block_active()
                )
            ):
                stuck_frames += 1
            else:
                stuck_frames = 0
            if stuck_frames * FIXED_DT >= STUCK_TIME_SECONDS:
                terminal_message = "vehicle_stuck"
                print(
                    f"Aborting run: vehicle stuck (speed {spd_ms:.2f} m/s for "
                    f"{stuck_frames * FIXED_DT:.1f}s)."
                )
                break

            overlay_lines = scorer.get_overlay_lines()
            overlay_lines.extend(
                vehicle_overlay_lines(
                    ego,
                    applied_control,
                    scorer.route_tracker,
                    dense_route_preview,
                    scorer.get_red_light_events(),
                )
            )
            hud.update(overlay_lines)

            if scorer.route_tracker.transforms:
                idx = min(scorer.route_tracker.current_index, len(dense_route_preview) - 1)
                next_loc = dense_route_preview[idx][0].location
                distance_to_wp = ego.get_transform().location.distance(next_loc)
                if distance_to_wp > 10.0:
                    terminal_message = "route_deviation"
                    print(
                        f"Aborting run: distance to next waypoint exceeded 10m "
                        f"(current {distance_to_wp:.2f} m at index {idx})."
                    )
                    break
            if low_speed_frames * FIXED_DT >= MAX_LOW_SPEED_SECONDS:
                terminal_message = "low_speed_timeout"
                print(
                    f"Aborting run: vehicle stayed below {STUCK_SPEED_THRESHOLD:.2f} m/s for "
                    f"{low_speed_frames * FIXED_DT:.1f}s."
                )
                break

            if scorer.completed_event_sent:
                terminal_message = "route_completed"
                sim_t, wall_t = scorer.elapsed_times()
                print(f"Route completed in {sim_t:.1f}s (sim) / {wall_t:.1f}s (wall).")
                break

            if random.random() < 0.02:
                for controller in walker_controllers:
                    controller.go_to_location(world.get_random_location_from_navigation())

            if sim_ticks >= next_apply_tick:
                if latency_steps == 0:
                    # Zero-latency: sample now so it is applied on the next tick.
                    pending_control = _clone_control(pcla.get_action())
                    applied_control = _clone_control(pending_control)
                else:
                    applied_control = _clone_control(pending_control)
                    pending_control = _clone_control(pcla.get_action())
                next_apply_tick = sim_ticks + control_period_steps

            if not apply_before_tick:
                # Non-zero latency: apply after the tick to model the delay.
                ego.apply_control(applied_control)

        status = "completed" if scorer.completed_event_sent else "aborted"
        message = terminal_message or ("route_completed" if status == "completed" else "manual_stop")

    except KeyboardInterrupt:
        message = "keyboard_interrupt"
        raise
    finally:
        for controller in walker_controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass

        destroy_sensors(aux_sensors)
        aux_sensors.clear()

        if pcla is not None:
            try:
                pcla.cleanup()
            except Exception:
                pass

        if ego is not None:
            destroy([ego])
        destroy(vehicles)
        destroy(walkers)
        destroy(walker_controllers)

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        hud.close()

        if scorer is not None:
            try:
                scorer.finalize()
            except Exception as exc:
                print(f"Failed to compute leaderboard-like stats: {exc}")

    if scorer is None:
        raise RuntimeError("Leaderboard scorer did not initialize; run aborted before start")

    sim_time, wall_time = scorer.elapsed_times()
    collisions = scorer.get_collision_counts()
    lane_crossings = scorer.get_lane_crossings()
    red_lights = scorer.get_red_light_events()
    print(
        f"Run summary -> sim_time {sim_time:.1f}s, wall_time {wall_time:.1f}s, "
        f"lane_crossings {lane_crossings}, "
        f"collisions v/p/s {collisions['vehicle']}/{collisions['pedestrian']}/{collisions['static']} "
        f"red_lights {red_lights} | route_completion {scorer.route_percent:.1f}% | status {status}"
    )

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'route_id': scorer.route_id,
        'route_xml': route_xml,
        'agent': args.agent,
        'control_period': float(effective_control_period),
        'control_latency': float(effective_control_latency),
        'control_steps': control_period_steps,
        'latency_steps': latency_steps,
        'vehicle_density': vehicle_density,
        'pedestrian_density': pedestrian_density,
        'status': status,
        'route_completion': scorer.route_percent,
        'sim_time': sim_time,
        'wall_time': wall_time,
        'lane_crossings': lane_crossings,
        'collisions_vehicle': collisions['vehicle'],
        'collisions_pedestrian': collisions['pedestrian'],
        'collisions_static': collisions['static'],
        'red_lights': red_lights,
        'message': message,
    }


def _route_id_from_config(route_config):
    name = getattr(route_config, "name", "route") or "route"
    repetition = getattr(route_config, "repetition_index", 0)
    return f"{name}_rep{repetition}"


def main():
    args = parse_args()
    route_xml = os.path.abspath(args.route)
    if not os.path.isfile(route_xml):
        raise FileNotFoundError(f"Route file {route_xml} not found")
    if args.route_id is not None:
        route_xml = _materialize_route_subset(route_xml, args.route_id)
        print(f"Using only route id {args.route_id} from {args.route} -> {route_xml}")
    if args.agent_config:
        args.agent_config = os.path.abspath(args.agent_config)
        if not os.path.isfile(args.agent_config):
            raise FileNotFoundError(f"Agent config override {args.agent_config} not found")
    scenario_json = os.path.join(CURRENT_DIR, "leaderboard_codes", "no_scenarios.json")
    preview_indexer = LegacyRouteIndexer(route_xml, scenario_json, 1)
    preview_config = preview_indexer.next()
    if not preview_config or not getattr(preview_config, "trajectory", None):
        raise RuntimeError(f"Route file {route_xml} does not contain any waypoints")

    route_town = getattr(preview_config, "town", None)
    if not route_town or route_town == "_":
        raise RuntimeError("Route file does not specify a town.")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    tm_port = DEFAULT_TM_PORT

    vehicle_counts = args.sweep_vehicle_density or [DEFAULT_VEHICLE_DENSITY]
    pedestrian_counts = args.sweep_pedestrian_density or [DEFAULT_PEDESTRIAN_DENSITY]
    control_latencies = args.sweep_control_latencies or [DEFAULT_CONTROL_LATENCY]

    if any(lat < 0 for lat in control_latencies):
        raise ValueError("All control latencies must be non-negative")

    def _build_density_pairs(vehicles, pedestrians):
        vehicle_sweep = bool(args.sweep_vehicle_density)
        pedestrian_sweep = bool(args.sweep_pedestrian_density)
        if vehicle_sweep and pedestrian_sweep:
            if len(vehicles) != len(pedestrians):
                raise ValueError(
                    "--sweep-vehicle-density and --sweep-pedestrian-density must have the same length "
                    "when both are provided."
                )
            return list(zip(vehicles, pedestrians))
        if vehicle_sweep:
            return [(vehicle, pedestrians[0]) for vehicle in vehicles]
        if pedestrian_sweep:
            return [(vehicles[0], ped) for ped in pedestrians]
        return [(vehicles[0], pedestrians[0])]

    density_pairs = _build_density_pairs(vehicle_counts, pedestrian_counts)
    combos = [
        (latency, veh_count, ped_count)
        for latency in control_latencies
        for (veh_count, ped_count) in density_pairs
    ]

    print(
        f"Preparing {len(combos)} configuration{'s' if len(combos) != 1 else ''} "
        f"(control_latencies={control_latencies}, density_pairs={density_pairs})."
    )

    sweep_logger = SweepLogger(args.sweep_log)
    route_fallback_id = _route_id_from_config(preview_config)
    world = _load_world_and_wait(client, route_town)

    try:
        for idx, (control_latency, vehicle_count, pedestrian_count) in enumerate(combos, start=1):
            print(
                f"\n=== Run {idx}/{len(combos)}: latency={control_latency:.3f}s | "
                f"vehicles={vehicle_count} | pedestrians={pedestrian_count} ==="
            )

            if idx == 1:
                current_world = world
            else:
                current_world = client.get_world()
                try:
                    current_world.wait_for_tick()
                except RuntimeError:
                    time.sleep(0.1)

            try:
                result = run_configuration(
                    args,
                    client,
                    current_world,
                    tm_port,
                    route_xml,
                    preview_config,
                    control_latency,
                    vehicle_count,
                    pedestrian_count,
                )
            except KeyboardInterrupt:
                print("Sweep interrupted by user. Exiting...")
                raise
            except Exception as exc:
                print(f"Configuration failed: {exc}")
                try:
                    world = _load_world_and_wait(client, route_town)
                except Exception as reload_exc:
                    print(f"Failed to reload world after error: {reload_exc}")
                sweep_logger.log({
                    'timestamp': datetime.utcnow().isoformat(),
                    'route_id': route_fallback_id,
                    'route_xml': route_xml,
                    'agent': args.agent,
                    'control_period': float(control_latency),
                    'control_latency': float(control_latency),
                    'control_steps': None,
                    'latency_steps': None,
                    'vehicle_density': vehicle_count,
                    'pedestrian_density': pedestrian_count,
                    'status': 'error',
                    'route_completion': None,
                    'sim_time': None,
                    'wall_time': None,
                    'lane_crossings': None,
                    'collisions_vehicle': None,
                    'collisions_pedestrian': None,
                    'collisions_static': None,
                    'red_lights': None,
                    'message': str(exc),
                })
                continue

            sweep_logger.log(result)
            world = current_world
    finally:
        sweep_logger.close()


if __name__ == "__main__":
    main()
