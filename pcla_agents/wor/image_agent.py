"""Frozen WoR driving policy with optional self-distilled observation compensation."""
import hashlib
import math
import os
from collections import deque

import carla
import numpy as np
import torch
import yaml

from leaderboard_codes.autonomous_agent1 import AutonomousAgent, Track
from leaderboard_codes.local_planner import RoadOption
from pcla_agents.wor.rails.camera_io import camera_specs, crop_rgb, stitch_rgb
from pcla_agents.wor.rails.models.main_model import CameraModel
from pcla_agents.wor.rails.models.time_adapter import load_delay_adapter
from pcla_agents.wor.waypointer import Waypointer


def get_entry_point():
    return 'ImageAgent'


class ImageAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self.num_frames = 0
        self.latest_lane_follow = True
        self.latest_route_command = None
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(path_to_conf_file) as stream:
            config = yaml.safe_load(stream)
        self.config = config.copy()
        for key, value in config.items():
            setattr(self, key, value)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.sensor_tick = config.get('sensor_tick', .05)
        config['imagenet_pretrained'] = False
        self.image_model = CameraModel(config).to(self.device)
        model_path = os.path.join(project_root, self.main_model_dir)
        payload = torch.load(model_path, map_location=self.device, weights_only=True)
        self.image_model.load_state_dict(payload.get('state_dict', payload), strict=True)
        self.image_model.eval().requires_grad_(False)
        self.delay_adapter = None
        if config.get('time_parameterized_control', False):
            adapter_path = os.path.join(project_root, self.time_adapter_path)
            # A missing checkpoint must not silently turn the adapter arm into a baseline.
            payload = torch.load(adapter_path, map_location=self.device, weights_only=True)
            self.delay_adapter = load_delay_adapter(payload, self.image_model.feature_dim, self.device)
            if 'base_checkpoint_sha256' in payload:
                digest = hashlib.sha256()
                with open(model_path, 'rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != payload['base_checkpoint_sha256']:
                    raise ValueError('Adapter was trained against a different frozen WoR checkpoint')
            trained = payload.get('model_config', {})
            for key in ('use_narr_cam', 'all_speeds', 'num_speeds', 'min_speeds', 'max_speeds',
                        'num_steers', 'num_throts', 'wide_crop_top', 'narr_crop_bottom', 'camera_x', 'camera_z'):
                if key in trained and trained[key] != config[key]:
                    raise ValueError(f'Adapter training/runtime config mismatch: {key}')
        self.max_observation_latency_s = (self.delay_adapter.max_latency_s if self.delay_adapter else
                                          config.get('max_observation_latency_s', .2))
        self.frame_history = deque()
        self.set_observation_latency(config.get('observation_latency_s', 0))
        self.waypointer = None
        self.steers = torch.linspace(-self.max_steers, self.max_steers, self.num_steers, device=self.device)
        self.throts = torch.linspace(0, self.max_throts, self.num_throts, device=self.device)
        self.lane_change_counter = 0
        self.lane_changed = None
        self.latest_timing = {}

    def set_observation_latency(self, latency_s):
        latency_s = float(latency_s)
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError('Observation latency must be finite and nonnegative')
        if self.delay_adapter is not None and latency_s > self.delay_adapter.max_latency_s + 1e-6:
            raise ValueError('Requested delay exceeds adapter training range')
        self.observation_latency_s = latency_s

    def sensors(self):
        return [dict(type='sensor.speedometer', id='EGO'),
                dict(type='sensor.other.gnss', x=0., y=0., z=self.camera_z,
                     sensor_tick=self.sensor_tick, id='GPS')] + camera_specs(self.config, self.sensor_tick)

    @torch.no_grad()
    def run_step(self, input_data, timestamp, vehicle=None, camera_timestamp_s=None,
                 received_monotonic_s=None):
        frames = [input_data[f'Wide_RGB_{i}'][0] for i in range(3)] + [input_data['Narrow_RGB'][0]]
        if len(set(frames)) != 1:
            raise ValueError('WoR requires frame-aligned wide/narrow cameras')
        if camera_timestamp_s is None:
            # PCLA's timestamp may have an episode-local origin. Use the frame age
            # relative to its current snapshot, not an absolute CARLA epoch.
            if vehicle is None:
                raise ValueError('Need vehicle or explicit camera timestamp for delay measurement')
            snapshot = vehicle.get_world().get_snapshot()
            camera_timestamp_s = timestamp - (snapshot.frame - frames[0]) * snapshot.timestamp.delta_seconds
        if not math.isfinite(camera_timestamp_s) or camera_timestamp_s > timestamp + 1e-6:
            raise ValueError('Camera capture timestamp must not be in the future')
        if self.frame_history and frames[0] < self.frame_history[-1]['frame']:
            raise ValueError('Camera frames must arrive in chronological order')
        if not self.frame_history or frames[0] != self.frame_history[-1]['frame']:
            wide, narrow = stitch_rgb([input_data[f'Wide_RGB_{i}'][1] for i in range(3)], input_data['Narrow_RGB'][1])
            self.frame_history.append(dict(timestamp=camera_timestamp_s, frame=frames[0],
                wide=wide, narrow=narrow, received_monotonic_s=received_monotonic_s))
        target = timestamp - self.observation_latency_s
        selected = self.frame_history[0]
        for packet in reversed(self.frame_history):
            if packet['timestamp'] <= target + 1e-6:
                selected = packet
                break
        age = max(0., timestamp - selected['timestamp'])
        warmup = selected['timestamp'] > target + 1e-6
        if self.delay_adapter is not None and age > self.delay_adapter.max_latency_s + 1e-6:
            raise ValueError('Measured observation age exceeds adapter training range')
        # Retain one packet before the entire supported delay window.
        keep_after = timestamp - max(self.max_observation_latency_s, self.observation_latency_s) - self.sensor_tick
        while len(self.frame_history) > 1 and self.frame_history[1]['timestamp'] < keep_after:
            self.frame_history.popleft()
        wide, narrow = crop_rgb(selected['wide'], selected['narrow'], self.config)
        wide = torch.as_tensor(wide, device=self.device).permute(2, 0, 1)[None].float()
        narrow = torch.as_tensor(narrow, device=self.device).permute(2, 0, 1)[None].float()
        _, ego = input_data['EGO']
        _, gps = input_data['GPS']
        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
        _, _, cmd = self.waypointer.tick(gps)
        self.latest_route_command = cmd
        self.latest_lane_follow = cmd == RoadOption.LANEFOLLOW
        spd = float(ego['speed'])
        cmd_value = max(0, cmd.value - 1) if cmd.value > 0 else 3
        if cmd_value in (4, 5):
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0
            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > 200 else None
        else:
            self.lane_change_counter, self.lane_changed = 0, None
        if cmd_value == self.lane_changed:
            cmd_value = 3
        ego_state = None
        if self.delay_adapter is not None and self.delay_adapter.ego_dim:
            state = [spd]
            if self.delay_adapter.ego_dim == 5:
                if vehicle is None:
                    raise ValueError('Full-state adapter requires live vehicle telemetry')
                control = vehicle.get_control()
                state += [math.radians(vehicle.get_angular_velocity().z), control.steer, control.throttle, control.brake]
            ego_state = torch.tensor([state], dtype=torch.float32, device=self.device)
        steer, throttle, brake = self.image_model.policy(wide, narrow, cmd_value,
            spd=torch.tensor([spd], device=self.device) if not self.all_speeds else None,
            delay_adapter=self.delay_adapter, latency_s=torch.tensor([age], device=self.device), ego_state=ego_state)
        if self.all_speeds:
            steer, throttle, brake = [self._lerp(v, spd) for v in (steer, throttle, brake)]
        brake_prob = float(self.action_prob(steer, throttle, brake)[-1])
        steer = float(self.steers @ torch.softmax(steer, dim=0))
        throttle = float(self.throts @ torch.softmax(throttle, dim=0))
        steer, throttle, brake = self.post_process(steer, throttle, brake_prob, spd, cmd_value)
        self.latest_timing = dict(camera_frame=selected['frame'], camera_timestamp_s=selected['timestamp'],
            received_monotonic_s=selected['received_monotonic_s'], observation_age_sim_s=age, warmup=warmup)
        self.num_frames += 1
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    def _lerp(self, values, speed):
        position = np.clip((speed - self.min_speeds) / (self.max_speeds - self.min_speeds), 0, 1) * (values.shape[0] - 1)
        lower, upper = math.floor(position), math.ceil(position)
        weight = position - lower
        return (1 - weight) * values[lower] + weight * values[upper]

    def action_prob(self, steer_logit, throt_logit, brake_logit):
        # Preserve the existing deployed driver, including its legacy layout.
        # Self-distillation uses CameraModel.driving_logits_from_features to match.
        steer = steer_logit.repeat(self.num_throts)
        throttle = throt_logit.repeat_interleave(self.num_steers)
        return torch.softmax(torch.cat([steer, throttle, brake_logit[None]]), dim=0)

    def post_process(self, steer, throt, brake_prob, spd, cmd):
        if brake_prob > 0.5:
            steer, throt, brake = 0, 0, 1
        else:
            brake, throt = 0, max(0.4, throt)
        if spd > {0: 10, 1: 10}.get(cmd, 20) / 3.6:
            throt = 0
        return steer, throt, brake

    def destroy(self):
        self.frame_history.clear()
        self.waypointer = None
        self.delay_adapter = None
        self.image_model = None
