"""Synchronous CARLA utilities for matched collection/evaluation camera packets."""
import hashlib
import json
import math
import os
import queue
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pcla_agents.wor.rails.camera_io import camera_specs

ROOT = Path(__file__).resolve().parent


def setup_carla_paths():
    carla_root = Path(os.environ.get('CARLA_ROOT', Path.home() / 'CARLA_0.9.16'))
    for path in (ROOT / 'scenario_runner', ROOT / 'leaderboard_codes',
                 carla_root / 'PythonAPI', carla_root / 'PythonAPI/carla'):
        if str(path) not in sys.path:
            sys.path.append(str(path))


def read_routes(path, selected=None):
    root = ET.parse(path).getroot()
    nodes = [root] if root.tag == 'route' else list(root.findall('route'))
    if selected:
        absent = set(selected) - {n.get('id') for n in nodes}
        if absent:
            raise ValueError(f'Unknown route IDs: {absent}')
        nodes = [n for n in nodes if n.get('id') in selected]
    routes = []
    for node in nodes:
        town = node.get('town')
        waypoints = node.findall('waypoint') or node.findall('waypoints/position')
        if not town or town == '_' or len(waypoints) < 2:
            raise ValueError('Each route needs a concrete town and at least two waypoints')
        points = [tuple(float(w.get(k, 0)) for k in ('x', 'y', 'z')) for w in waypoints]
        fingerprint = hashlib.sha256(json.dumps([town, points]).encode()).hexdigest()[:16]
        routes.append(dict(xml=node, town=town, xml_id=node.get('id'), route_id=fingerprint, points=points))
    if not routes:
        raise ValueError('No routes selected')
    return routes


def write_route(route, path):
    root = ET.Element('routes')
    node = ET.SubElement(root, 'route', id=route['xml_id'] or route['route_id'], town=route['town'])
    # Normalize both supported XML layouts for the repository's legacy scorer.
    for x, y, z in route['points']:
        ET.SubElement(node, 'waypoint', x=str(x), y=str(y), z=str(z))
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def live_state(ego):
    control = ego.get_control()
    return dict(spd=ego.get_velocity().length(), yaw_rate=math.radians(ego.get_angular_velocity().z),
                previous_control=[control.steer, control.throttle, control.brake])


def gps_position(world, ego):
    geo = world.get_map().transform_to_geolocation(ego.get_location())
    return np.array([geo.latitude, geo.longitude, geo.altitude])


class CameraRig:
    def __init__(self, world, ego, config, dt):
        import carla
        self.sensors, self.queues = [], {}
        try:
            for spec in camera_specs(config, dt):
                bp = world.get_blueprint_library().find('sensor.camera.rgb')
                attrs = dict(image_size_x=spec['width'], image_size_y=spec['height'], fov=spec['fov'],
                             sensor_tick=dt, lens_circle_multiplier=3.0, lens_circle_falloff=3.0,
                             chromatic_aberration_intensity=.5, chromatic_aberration_offset=0)
                for key, value in attrs.items():
                    bp.set_attribute(key, str(value))
                transform = carla.Transform(carla.Location(x=spec['x'], z=spec['z']), carla.Rotation(yaw=spec['yaw']))
                sensor = world.spawn_actor(bp, transform, attach_to=ego)
                self.sensors.append(sensor)
                inbox = queue.Queue()
                self.queues[spec['id']] = inbox
                sensor.listen(lambda image, inbox=inbox: inbox.put((image, time.monotonic())))
        except BaseException:
            self.close()
            raise

    def read(self, frame, timeout=20):
        deadline = time.monotonic() + timeout
        images, timestamps, arrivals = {}, [], []
        for name, inbox in self.queues.items():
            while True:
                try:
                    image, received = inbox.get(timeout=max(.001, deadline - time.monotonic()))
                except queue.Empty as exc:
                    raise RuntimeError(f'Timed out waiting for {name} frame {frame}') from exc
                if image.frame > frame:
                    raise RuntimeError(f'{name} skipped requested frame {frame}')
                if image.frame == frame:
                    break
            images[name] = (frame, np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4).copy())
            timestamps.append(image.timestamp)
            arrivals.append(received)
        if max(timestamps) - min(timestamps) > 1e-6:
            raise RuntimeError('Camera timestamps are not aligned')
        # Earliest camera arrival measures time until all views are available too.
        return images, timestamps[0], min(arrivals)

    def close(self):
        for sensor in self.sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self.sensors.clear()


class CarlaEpisode:
    """Own actors and settings; reload the town for each matched experimental arm."""
    def __init__(self, args, route, config, density, seed):
        self.args, self.route, self.config = args, route, config
        self.density, self.seed = density, seed
        self.actors, self.rig, self.world, self.tm, self.original_settings = [], None, None, None, None

    def __enter__(self):
        import carla
        from leaderboard_codes.route_manipulation import interpolate_trajectory
        try:
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(self.args.timeout)
            self.world = self.client.load_world(self.route['town'])
            self.original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1 / self.args.fps
            settings.substepping = True
            settings.max_substep_delta_time = .01
            settings.max_substeps = max(10, math.ceil(settings.fixed_delta_seconds / .01))
            self.world.apply_settings(settings)
            self.tm = self.client.get_trafficmanager(self.args.tm_port)
            self.tm.set_synchronous_mode(True)
            self.tm.set_random_device_seed(self.seed)
            self.world.set_weather(getattr(carla.WeatherParameters, self.args.weather))
            locations = [carla.Location(x=p[0], y=p[1], z=p[2]) for p in self.route['points']]
            _, self.dense_route = interpolate_trajectory(self.world, locations)
            self.gps_route = []
            for transform, option in self.dense_route:
                geo = self.world.get_map().transform_to_geolocation(transform.location)
                self.gps_route.append((dict(lat=geo.latitude, lon=geo.longitude, z=geo.altitude), option))
            library = self.world.get_blueprint_library()
            bp = library.find('vehicle.tesla.model3')
            bp.set_attribute('role_name', 'hero')
            spawn = self.world.get_map().get_waypoint(locations[0]).transform
            spawn.location.z += .5
            self.ego = self.world.spawn_actor(bp, spawn)
            self.actors.append(self.ego)
            rng = random.Random(self.seed)
            spawns = self.world.get_map().get_spawn_points()
            rng.shuffle(spawns)
            blueprints = sorted([b for b in library.filter('vehicle.*')
                                 if b.has_attribute('number_of_wheels') and b.get_attribute('number_of_wheels').as_int() == 4], key=lambda b: b.id)
            self.actual_density = 0
            for transform in spawns:
                if self.actual_density >= self.density:
                    break
                if transform.location.distance(spawn.location) < 10:
                    continue
                other = self.world.try_spawn_actor(rng.choice(blueprints), transform)
                if other:
                    self.actors.append(other)
                    other.set_autopilot(True, self.args.tm_port)
                    self.actual_density += 1
            self.rig = CameraRig(self.world, self.ego, self.config, 1 / self.args.fps)
            self.path = locations[1:]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def tick(self):
        frame = self.world.tick(self.args.timeout)
        snapshot = self.world.get_snapshot()
        images, sensor_time, received = self.rig.read(frame, self.args.timeout)
        if abs(snapshot.timestamp.elapsed_seconds - sensor_time) > 1e-6:
            raise RuntimeError('Camera capture does not match current simulator state')
        return frame, snapshot.timestamp.elapsed_seconds, images, sensor_time, received

    def __exit__(self, *_):
        if self.rig:
            self.rig.close()
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        if self.tm:
            self.tm.set_synchronous_mode(False)
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)


def add_common_arguments(parser):
    parser.add_argument('--routes', required=True, help='Route XML with concrete towns; all selected routes run')
    parser.add_argument('--route-ids', nargs='+')
    parser.add_argument('--config', default='outputs/config_timeparam.yaml')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000)
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--vehicle-densities', nargs='+', type=int, default=[0, 20, 50])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--weather', default='ClearNoon')
    parser.add_argument('--timeout', type=float, default=30)


def validate_common(args):
    if not all(math.isfinite(v) and v > 0 for v in (args.fps, args.seconds, args.timeout)):
        raise ValueError('fps, seconds and timeout must be finite and positive')
    if min(args.vehicle_densities) < 0:
        raise ValueError('Vehicle densities must be nonnegative')
    if args.seconds * args.fps < 2:
        raise ValueError('Episode must span at least two simulator ticks')
    setup_carla_paths()
