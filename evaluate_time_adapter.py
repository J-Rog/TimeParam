#!/usr/bin/env python3
"""Compare frozen WoR, delayed WoR and delayed+adapter on complete held-out routes."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml

from carla_delay_common import (ROOT, CarlaEpisode, add_common_arguments, gps_position,
                                live_state, read_routes, validate_common, write_route)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['arm'], row['delay_ms'], row['vehicle_density'])].append(row)
    output = []
    for (arm, delay, density), group in groups.items():
        output.append(dict(arm=arm, delay_ms=delay, vehicle_density=density, runs=len(group),
            completion_rate=sum(row['completed'] for row in group) / len(group),
            route_completion_mean=float(np.mean([row['route_completion'] for row in group])),
            collisions_total=sum(sum(row['collisions'].values()) for row in group),
            red_lights_total=sum(row['red_lights'] for row in group),
            lane_invasion_events_total=sum(row['lane_invasion_events'] for row in group),
            sensing_to_actuation_sim_mean_s=float(np.mean([row['sensing_to_actuation_sim_mean_s'] for row in group])),
            receive_to_apply_wall_mean_s=float(np.mean([row['receive_to_apply_wall_mean_s'] for row in group]))))
    return output


def evaluate_run(args, route, config, density, seed, arm, delay, output):
    import carla
    from pcla_agents.wor.image_agent import ImageAgent
    from wor_common import LeaderboardLikeScorer
    output.mkdir(parents=True, exist_ok=False)
    route_path = output / 'route.xml'
    write_route(route, route_path)
    run_config = dict(config, sensor_tick=1 / args.fps, observation_latency_s=delay,
                      time_parameterized_control=arm == 'adapter',
                      time_adapter_path=str(Path(args.adapter).resolve()))
    # Config paths are always project-relative unless absolute.
    run_config['main_model_dir'] = str((ROOT / run_config['main_model_dir']).resolve())
    config_path = output / 'config.yaml'
    config_path.write_text(yaml.safe_dump(run_config))
    agent = ImageAgent(str(config_path))
    times, sensors = [], []
    lane_events = []
    try:
        with CarlaEpisode(args, route, config, density, seed) as episode:
            agent.set_global_plan(episode.gps_route, episode.dense_route)
            scorer = LeaderboardLikeScorer(episode.world, str(route_path))
            try:
                for sensor_type, callback in [('sensor.other.collision', scorer.on_collision),
                                             ('sensor.other.lane_invasion', lambda event: (lane_events.append(event.frame), scorer.on_lane_invasion(event)))]:
                    sensor = episode.world.spawn_actor(episode.world.get_blueprint_library().find(sensor_type),
                                                       carla.Transform(), attach_to=episode.ego)
                    sensors.append(sensor)
                    sensor.listen(callback)
                pending = None
                stopped_steps = 0
                reason = 'time_limit'
                with (output / 'timing.jsonl').open('w') as stream:
                    for _ in range(max(2, round(args.seconds * args.fps))):
                        frame, sim_time, images, camera_time, received = episode.tick()
                        if pending is not None:
                            # First world tick after apply_control: this command has now actuated.
                            pending['actuation_timestamp_s'] = sim_time
                            pending['sensing_to_actuation_sim_s'] = sim_time - pending['camera_timestamp_s']
                            times.append(pending)
                            stream.write(json.dumps(pending) + '\n')
                        scorer.tick(episode.ego, 1 / args.fps, frame)
                        state = live_state(episode.ego)
                        stopped_steps = stopped_steps + 1 if state['spd'] < .1 else 0
                        if scorer.completed_event_sent:
                            reason = 'route_completed'
                            break
                        if stopped_steps / args.fps > args.stuck_seconds:
                            reason = 'stuck_timeout'
                            break
                        tracker = scorer.route_tracker
                        if tracker.transforms and episode.ego.get_location().distance(tracker.transforms[tracker.current_index].location) > 15:
                            reason = 'route_deviation'
                            break
                        images.update(EGO=(frame, dict(speed=state['spd'])), GPS=(frame, gps_position(episode.world, episode.ego)))
                        inference_start = time.monotonic()
                        control = agent.run_step(images, sim_time, episode.ego,
                            camera_timestamp_s=camera_time, received_monotonic_s=received)
                        inference_s = time.monotonic() - inference_start
                        episode.ego.apply_control(control)
                        applied = time.monotonic()
                        pending = dict(agent.latest_timing, command_frame=frame, command_timestamp_s=sim_time,
                            inference_wall_s=inference_s,
                            receive_to_apply_wall_s=applied - agent.latest_timing['received_monotonic_s'])
                valid = [row for row in times if not row['warmup']]
                if not valid:
                    raise RuntimeError('Run ended before any full-delay control reached actuation')
                result = dict(arm=arm, delay_ms=delay * 1000, route_id=route['route_id'], town=route['town'],
                    source_route_id=route['xml_id'], seed=seed, weather=args.weather,
                    vehicle_density=density, actual_vehicle_density=episode.actual_density,
                    completed=scorer.completed_event_sent, route_completion=scorer.route_percent,
                    reason=reason, collisions=scorer.get_collision_counts(), red_lights=scorer.get_red_light_events(),
                    lane_invasion_events=len(lane_events), lane_markings_crossed=scorer.get_lane_crossings(),
                    measured_controls=len(valid), warmup_controls=len(times) - len(valid),
                    sensing_to_actuation_sim_mean_s=float(np.mean([r['sensing_to_actuation_sim_s'] for r in valid])),
                    sensing_to_actuation_sim_p95_s=float(np.percentile([r['sensing_to_actuation_sim_s'] for r in valid], 95)),
                    receive_to_apply_wall_mean_s=float(np.mean([r['receive_to_apply_wall_s'] for r in valid])),
                    inference_wall_mean_s=float(np.mean([r['inference_wall_s'] for r in valid])))
                (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                return result
            finally:
                for sensor in sensors:
                    try:
                        sensor.stop()
                        sensor.destroy()
                    except RuntimeError:
                        pass
    finally:
        agent.destroy()


def evaluate(args):
    validate_common(args)
    if args.stuck_seconds <= 0 or any(not math.isfinite(d) or d < 0 for d in args.delays_ms):
        raise ValueError('Invalid stuck timeout or delay')
    import torch
    payload = torch.load(args.adapter, map_location='cpu', weights_only=True)
    config = yaml.safe_load(Path(args.config).read_text())
    routes = read_routes(args.routes, args.route_ids)
    if args.validation_routes_only:
        validation = set(payload.get('val_routes', []))
        routes = [r for r in routes if f"{r['town']}:{r['route_id']}" in validation]
        if not routes:
            raise ValueError('No checkpoint validation routes found in the supplied XML')
    trained = set(payload.get('train_routes', []))
    if not trained:
        raise ValueError('Evaluation requires a self-distilled checkpoint with train_routes provenance')
    overlap = {f"{r['town']}:{r['route_id']}" for r in routes} & trained
    if overlap:
        raise ValueError(f'Evaluation routes overlap training: {sorted(overlap)}')
    max_delay = payload['adapter_config']['max_latency_s']
    if max(args.delays_ms) / 1000 > max_delay + 1e-6:
        raise ValueError('Evaluation delay exceeds adapter training range')
    for delay in args.delays_ms:
        if not math.isclose(delay / 1000 * args.fps, round(delay / 1000 * args.fps), abs_tol=1e-6):
            raise ValueError('Evaluation delays must align to camera ticks; choose multiples of 1000/fps ms')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    arms = [('no_delay', 0.)] + [(arm, d / 1000) for d in sorted(set(args.delays_ms)) for arm in ('delayed', 'adapter')]
    for route in routes:
        for density in args.vehicle_densities:
            for seed in args.seeds:
                for arm, delay in arms:
                    directory = root / f"{route['town']}_{route['route_id']}_v{density}_s{seed}_{arm}_{delay * 1000:g}ms"
                    try:
                        result = evaluate_run(args, route, config, density, seed, arm, delay, directory)
                    except Exception as exc:
                        directory.mkdir(parents=True, exist_ok=True)
                        (directory / 'error.json').write_text(json.dumps(dict(error=str(exc), arm=arm, delay_s=delay), indent=2))
                        raise
                    rows.append(result)
                    (root / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
                    (root / 'summary.json').write_text(json.dumps(summarize(rows), indent=2) + '\n')
                    print(json.dumps(result), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(p)
    p.add_argument('--adapter', default='outputs/time_adapter.th')
    p.add_argument('--output', required=True, help='New directory; existing runs are never overwritten')
    p.add_argument('--delays-ms', nargs='+', type=float, default=[0, 50, 100, 150, 200])
    p.add_argument('--stuck-seconds', type=float, default=60, help='Allow red-light waiting before declaring stuck')
    p.add_argument('--validation-routes-only', action='store_true',
                   help='Select only checkpoint validation routes from the supplied XML')
    return p.parse_args(argv)


if __name__ == '__main__':
    evaluate(parse_args())
