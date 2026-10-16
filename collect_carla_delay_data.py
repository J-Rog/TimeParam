#!/usr/bin/env python3
"""Collect label-free, frame-aligned WoR camera trajectories using Traffic Manager."""
import argparse
import json
from pathlib import Path

from PIL import Image
import yaml

from carla_delay_common import (CarlaEpisode, add_common_arguments, gps_position, live_state,
                                read_routes, validate_common, write_route)
from pcla_agents.wor.rails.camera_io import camera_specs, stitch_rgb


def collect(args):
    validate_common(args)
    from leaderboard_codes.route_manipulation import downsample_route
    from pcla_agents.wor.waypointer import Waypointer
    from wor_common import _RouteCompletionTracker
    config = yaml.safe_load(Path(args.config).read_text())
    for route in read_routes(args.routes, args.route_ids):
        for density in args.vehicle_densities:
            for seed in args.seeds:
                name = f"{route['town']}_{route['route_id']}_v{density}_s{seed}_{args.weather}"
                output = Path(args.output) / name
                output.mkdir(parents=True, exist_ok=False)
                (output / 'rgb').mkdir()
                write_route(route, output / 'route.xml')
                frames = []
                metadata = dict(schema_version=1, route_id=route['route_id'], town=route['town'],
                                source_route_id=route['xml_id'], seed=seed, vehicle_density=density,
                                fps=args.fps, weather=args.weather, frames=frames,
                                camera_layout='wor_stitched_rgb_v1',
                                camera_config={key: config[key] for key in ('camera_x', 'camera_z')},
                                camera_specs=camera_specs(config, 1 / args.fps), status='incomplete')
                try:
                    with CarlaEpisode(args, route, config, density, seed) as episode:
                        metadata['actual_vehicle_density'] = episode.actual_density
                        # Same route command generator as deployment, evaluated at current time.
                        ids = downsample_route(episode.dense_route, 50)
                        plan = [episode.gps_route[i] for i in ids]
                        waypointer = Waypointer(plan, gps_position(episode.world, episode.ego))
                        tracker = _RouteCompletionTracker(episode.dense_route)
                        episode.ego.set_autopilot(True, args.tm_port)
                        episode.tm.set_path(episode.ego, episode.path)
                        for index in range(round(args.seconds * args.fps)):
                            frame, timestamp, images, _, _ = episode.tick()
                            _, _, command = waypointer.tick(gps_position(episode.world, episode.ego))
                            wide, narrow = stitch_rgb([images[f'Wide_RGB_{i}'][1] for i in range(3)], images['Narrow_RGB'][1])
                            wide_path, narrow_path = f'rgb/{index:06d}_wide.png', f'rgb/{index:06d}_narr.png'
                            Image.fromarray(wide).save(output / wide_path)
                            Image.fromarray(narrow).save(output / narrow_path)
                            frames.append(dict(frame=frame, timestamp=timestamp, wide_rgb=wide_path,
                                narr_rgb=narrow_path, cmd=command.value - 1 if command.value > 0 else 3,
                                **live_state(episode.ego)))
                            completion = tracker.update(episode.ego.get_location())
                            if completion >= 99:
                                break
                        metadata.update(status='collected', route_completion=completion)
                finally:
                    # Interrupted runs retain their successfully captured frames for inspection.
                    (output / 'data.json').write_text(json.dumps(metadata, indent=2) + '\n')
                print(f"Collected {len(frames)} frames: {output}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(p)
    p.add_argument('--output', required=True)
    return p.parse_args(argv)


if __name__ == '__main__':
    collect(parse_args())
