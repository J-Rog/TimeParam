import carla
import os
import random
import time
from datetime import datetime

from PCLA import PCLA
from leaderboard_codes.route_indexer import RouteIndexer as LegacyRouteIndexer
from leaderboard_codes.route_manipulation import interpolate_trajectory

from wor_common import (
    DEFAULT_TM_PORT,
    LeaderboardLikeScorer,
    OverlayHUD,
    SweepLogger,
    _apply_route_weather,
    _clone_control,
    _load_world_and_wait,
    _materialize_route_subset,
    _route_id_from_config,
    destroy,
    destroy_sensors,
    make_sync,
    spawn_tm_vehicles,
    spawn_walkers,
    vehicle_overlay_lines,
)


class TimingAdapter:
    """Interface for control timing and optional overlay/result hooks."""

    def __init__(self, control_latency, fixed_dt):
        self.latency_steps = 0
        self.control_period_steps = 1

    def attach(self, pcla, scorer, args):
        return None

    def update(self, ego):
        return None

    def overlay_lines(self):
        return []

    def result_fields(self, control_latency):
        return {}

    def summary_extra(self):
        return ""


class FixedLatencyAdapter(TimingAdapter):
    """Fixed latency/period model used by test_wor."""

    def __init__(self, control_latency, fixed_dt, min_control_period):
        super().__init__(control_latency, fixed_dt)
        if control_latency >= min_control_period:
            self.latency_steps = max(1, int(round(control_latency / fixed_dt)))
            self.control_period_steps = self.latency_steps
        else:
            self.latency_steps = 0
            self.control_period_steps = max(1, int(round(min_control_period / fixed_dt)))

        self.effective_control_latency = self.latency_steps * fixed_dt
        self.effective_control_period = self.control_period_steps * fixed_dt

        if control_latency == 0:
            print(
                f"Using zero-latency mode with control period {self.effective_control_period:.3f}s "
                f"(steps={self.control_period_steps})."
            )
        elif abs(self.effective_control_latency - control_latency) > 1e-6:
            print(
                f"Requested control latency {control_latency:.3f}s rounded to "
                f"{self.effective_control_latency:.3f}s (steps={self.latency_steps})."
            )

    def result_fields(self, control_latency):
        return {
            'control_period': float(self.effective_control_period),
            'control_latency': float(self.effective_control_latency),
            'control_steps': self.control_period_steps,
            'latency_steps': self.latency_steps,
        }


def run_configuration(
    args,
    client,
    world,
    tm_port,
    route_xml,
    preview_config,
    control_latency,
    vehicle_density,
    pedestrian_density,
    timing_factory,
):
    FIXED_DT = 0.025
    STUCK_SPEED_THRESHOLD = 0.2
    STUCK_TIME_SECONDS = 8.0
    MAX_LOW_SPEED_SECONDS = 40.0

    _apply_route_weather(world, preview_config)
    tm = client.get_trafficmanager(tm_port)
    timing = timing_factory(args, control_latency, FIXED_DT)

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
        timing.attach(pcla, scorer, args)

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
        latency_steps = timing.latency_steps
        control_period_steps = timing.control_period_steps
        pending_control = _clone_control(pcla.get_action())
        applied_control = carla.VehicleControl()
        if latency_steps == 0:
            applied_control = _clone_control(pending_control)
        next_apply_tick = control_period_steps
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
            overlay_lines.extend(timing.overlay_lines())
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
                timing.update(ego)
                latency_steps = timing.latency_steps
                control_period_steps = timing.control_period_steps
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
    summary_extra = timing.summary_extra()
    extra_segment = f" | {summary_extra}" if summary_extra else ""
    print(
        f"Run summary -> sim_time {sim_time:.1f}s, wall_time {wall_time:.1f}s, "
        f"lane_crossings {lane_crossings}, "
        f"collisions v/p/s {collisions['vehicle']}/{collisions['pedestrian']}/{collisions['static']} "
        f"red_lights {red_lights}{extra_segment} | "
        f"route_completion {scorer.route_percent:.1f}% | status {status}"
    )

    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'route_id': scorer.route_id,
        'route_xml': route_xml,
        'agent': args.agent,
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
    result.update(timing.result_fields(control_latency))
    return result


def run_sweep(
    args,
    timing_factory,
    default_control_latency,
    default_vehicle_density,
    default_pedestrian_density,
    tm_port=DEFAULT_TM_PORT,
):
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

    scenario_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard_codes", "no_scenarios.json")
    preview_indexer = LegacyRouteIndexer(route_xml, scenario_json, 1)
    preview_config = preview_indexer.next()
    if not preview_config or not getattr(preview_config, "trajectory", None):
        raise RuntimeError(f"Route file {route_xml} does not contain any waypoints")

    route_town = getattr(preview_config, "town", None)
    if not route_town or route_town == "_":
        raise RuntimeError("Route file does not specify a town.")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    vehicle_counts = args.sweep_vehicle_density or [default_vehicle_density]
    pedestrian_counts = args.sweep_pedestrian_density or [default_pedestrian_density]
    control_latencies = args.sweep_control_latencies or [default_control_latency]

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
                    timing_factory,
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
