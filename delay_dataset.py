"""Label-free, timestamp-paired trajectories and route-disjoint splits."""
import bisect
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from pcla_agents.wor.rails.camera_io import crop_rgb


def discover_sequences(root):
    root = Path(root).resolve()
    sequences = []
    for path in sorted(root.rglob('data.json')):
        with path.open() as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or not payload.get('route_id') or not payload.get('town'):
            raise ValueError(f'{path}: require town, route_id and frames (see docs/SELF_DISTILLATION.md)')
        frames = payload.get('frames', [])
        if not frames:
            raise ValueError(f'{path}: empty trajectory')
        times = [float(frame['timestamp']) for frame in frames]
        if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f'{path}: timestamps must be finite and strictly increasing')
        for frame in frames:
            if not math.isfinite(float(frame['spd'])) or float(frame['spd']) < 0:
                raise ValueError(f'{path}: invalid speed')
            if frame['cmd'] not in range(6):
                raise ValueError(f'{path}: cmd must use zero-based WoR commands 0..5')
            for key in ('wide_rgb', 'narr_rgb'):
                asset = (path.parent / frame[key]).resolve()
                if path.parent not in asset.parents or not asset.is_file():
                    raise ValueError(f'{path}: missing or nonlocal image {frame[key]}')
        sequences.append(dict(id=path.parent.relative_to(root).as_posix(), directory=path.parent,
                              route=f"{payload['town']}:{payload['route_id']}",
                              camera_config=payload.get('camera_config', {}),
                              town=payload['town'], frames=frames, times=times))
    if not sequences:
        raise ValueError(f'No data.json trajectories under {root}')
    return sequences


def split_sequences(sequences, seed=42, val_fraction=0.2, heldout_towns=(), manifest=None):
    """Group repeated runs of the same route before splitting, never split frames."""
    ids = {s['id'] for s in sequences}
    if manifest is not None:
        train_ids, val_ids = set(manifest['train']), set(manifest['val'])
        if train_ids & val_ids or train_ids | val_ids != ids:
            raise ValueError('Split manifest must partition every trajectory exactly once')
    else:
        routes = sorted({s['route'] for s in sequences})
        if heldout_towns:
            unknown = set(heldout_towns) - {s['town'] for s in sequences}
            if unknown:
                raise ValueError(f'Held-out towns absent from data: {unknown}')
            val_routes = {s['route'] for s in sequences if s['town'] in heldout_towns}
        else:
            if not 0 < val_fraction < 1 or len(routes) < 2:
                raise ValueError('Need at least two distinct routes and 0 < val_fraction < 1')
            random.Random(seed).shuffle(routes)
            count = min(len(routes) - 1, max(1, round(len(routes) * val_fraction)))
            val_routes = set(routes[:count])
        val_ids = {s['id'] for s in sequences if s['route'] in val_routes}
        train_ids = ids - val_ids
    train = [s for s in sequences if s['id'] in train_ids]
    val = [s for s in sequences if s['id'] in val_ids]
    if not train or not val:
        raise ValueError('Train and validation must both contain whole trajectories')
    if {s['route'] for s in train} & {s['route'] for s in val}:
        raise ValueError('Route leakage between train and validation')
    return train, val, dict(train=sorted(train_ids), val=sorted(val_ids))


def stale_index(times, current_index, delay_s, tolerance_s=1e-6):
    """Newest observation at/before t-delay. None during insufficient history."""
    index = bisect.bisect_right(times, times[current_index] - delay_s + tolerance_s,
                               hi=current_index + 1) - 1
    return index if index >= 0 else None


class DelayedJSONActionDataset(Dataset):
    """Name retained for the old entry point; no action/segmentation labels are read.

    Every eligible current frame is paired with every requested delay. Actual
    timestamp differences, not nominal frame counts, condition the adapter.
    """
    def __init__(self, sequences, config, delays_s=(0, .05, .1, .15, .2),
                 max_latency_s=.2, ego_dim=5):
        self.sequences, self.config, self.ego_dim = sequences, config, ego_dim
        self.delays_s = tuple(sorted(set(float(d) for d in delays_s)))
        if (not self.delays_s or any(not math.isfinite(d) or d < 0 for d in self.delays_s)
                or self.delays_s[-1] > max_latency_s or max_latency_s <= 0):
            raise ValueError('Delays must be finite and within [0, max_latency_s]')
        self.samples = []
        for sequence_index, seq in enumerate(sequences):
            for key, value in seq.get('camera_config', {}).items():
                if value != config.get(key):
                    raise ValueError(f'Collection/training camera mismatch: {key}')
            if ego_dim == 5:
                for frame in seq['frames']:
                    state = [frame['yaw_rate'], *frame['previous_control']]
                    if len(state) != 4 or not all(math.isfinite(float(x)) for x in state):
                        raise ValueError('Need yaw_rate (rad/s) and previous_control [steer, throttle, brake]')
            # Equal current-frame support across all delays; never clamp warmup to frame 0.
            for current_index in range(len(seq['frames'])):
                pairs = []
                for delay in self.delays_s:
                    stale = stale_index(seq['times'], current_index, delay)
                    if stale is None:
                        break
                    actual = seq['times'][current_index] - seq['times'][stale]
                    if actual > max_latency_s + 1e-6:
                        break  # missing sensor frames: do not silently train outside range
                    pairs.append((sequence_index, current_index, stale, delay, actual))
                if len(pairs) == len(self.delays_s):
                    self.samples.extend(pairs)
        if not self.samples:
            raise ValueError('No complete delay windows; collect longer sequences or reduce delay range')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence_index, current, stale, requested, actual = self.samples[index]
        seq = self.sequences[sequence_index]
        def load(frame_index):
            frame = seq['frames'][frame_index]
            with Image.open(seq['directory'] / frame['wide_rgb']) as image:
                wide = np.array(image.convert('RGB'))
            with Image.open(seq['directory'] / frame['narr_rgb']) as image:
                narrow = np.array(image.convert('RGB'))
            return crop_rgb(wide, narrow, self.config)
        sw, sn = load(stale)
        cw, cn = (sw.copy(), sn.copy()) if stale == current else load(current)
        frame = seq['frames'][current]
        state = [float(frame['spd'])]
        if self.ego_dim == 5:
            state += [float(frame['yaw_rate']), *map(float, frame['previous_control'])]
        if self.ego_dim == 0:
            state = []
        return dict(stale_wide=sw, stale_narr=sn, current_wide=cw, current_narr=cn,
                    speed=float(frame['spd']), command=int(frame['cmd']),
                    ego_state=np.asarray(state, dtype=np.float32), latency_s=actual,
                    requested_delay_s=requested, timestamp=seq['times'][current])
