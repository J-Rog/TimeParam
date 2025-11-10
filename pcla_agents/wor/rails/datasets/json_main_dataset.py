import glob
import json
import os
import random

import cv2
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from ...common.augmenter import augment
from ...utils import filter_sem


def _read_rgb(path):
    """Load an RGB image (uint8, HxWx3)."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im[:, :, ::-1].copy()  # BGR -> RGB with positive strides


def _read_sem_gray(path):
    """Load a grayscale semantic mask (uint8, HxW)."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


class JSONMainDataset(Dataset):
    """
    Drop-in replacement for MainDataset that loads from an image+JSON dump
    produced by the conversion scripts (one folder per sequence).

    Each sequence directory must contain:
      - data.json: list of frame dicts (see conversion script)
      - rgbs/, segs/, etc: referenced by the JSON paths.
    """

    def __init__(
        self,
        data_dir,
        config_path,
        use_augment=None,
        augment_prob=0.5,
        wide_scale=1.0,
        narr_scale=1.0,
    ):
        import yaml

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.T = cfg["num_plan"]
        self.camera_yaws = cfg["camera_yaws"]
        self.wide_crop_top = cfg["wide_crop_top"]
        self.narr_crop_bottom = cfg["narr_crop_bottom"]
        self.seg_channels = cfg["seg_channels"]

        self.num_speeds = cfg["num_speeds"]
        self.num_steers = cfg["num_steers"]
        self.num_throts = cfg["num_throts"]
        self.multi_cam = cfg.get("multi_cam", False)
        self.wide_scale = wide_scale
        self.narr_scale = narr_scale

        # Augmentation mirrors LabeledMainDataset behavior
        cfg_use_aug = cfg.get("use_augment", False)
        self.use_augment = cfg_use_aug if use_augment is None else use_augment
        aug_prob = cfg.get("augment_prob", augment_prob)
        self.augmenter = augment(aug_prob) if self.use_augment else None

        self.num_cmds = cfg.get("num_cmds", 6)

        self.seq_meta = []  # [{dir, frames, n}]
        self.index = []  # list of (seq_idx, base_idx, cam_idx)
        self.cmd_labels = []
        self.cmd_buckets = defaultdict(list)

        seq_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
        for seq_dir in seq_dirs:
            json_path = os.path.join(seq_dir, "data.json")
            if not os.path.isfile(json_path):
                continue
            frames, num_frames = self._load_frames(json_path)
            if num_frames < self.T + 1:
                continue
            self.seq_meta.append({"dir": seq_dir, "frames": frames, "n": num_frames})

        dataset_counter = 0
        for seq_idx, seq in enumerate(self.seq_meta):
            n = seq["n"]
            for base_idx in range(n - self.T):
                cmd_val = int(self._scalarize(seq["frames"][base_idx].get("cmd", 0)))
                if not self.multi_cam:
                    self.cmd_labels.append(cmd_val)
                    self.cmd_buckets[cmd_val].append(dataset_counter)
                    dataset_counter += 1
                for cam_idx in range(len(self.camera_yaws)):
                    self.index.append((seq_idx, base_idx, cam_idx))
                    if self.multi_cam:
                        sample_idx = len(self.index) - 1
                        self.cmd_labels.append(cmd_val)
                        self.cmd_buckets[cmd_val].append(sample_idx)

        if self.multi_cam:
            self._len = len(self.index)
        else:
            cam_count = max(1, len(self.camera_yaws))
            self._len = len(self.index) // cam_count

        print(
            f"{data_dir}: {self._len} frames (x{len(self.camera_yaws)}) from {len(self.seq_meta)} sequences"
        )

    def __len__(self):
        return self._len

    def command_histogram(self):
        """Return a dict {cmd: count} describing the dataset distribution."""
        return {cmd: len(indices) for cmd, indices in self.cmd_buckets.items()}

    def balanced_indices(self, total_count=None, fraction=None, seed=None):
        """
        Return dataset indices sampled to balance command categories.
        """
        if fraction is not None:
            if not (0 < fraction <= 1):
                raise ValueError("fraction must lie in (0, 1]")
            total_count = max(1, int(round(self._len * fraction)))
        if total_count is None:
            raise ValueError("total_count or fraction must be provided")
        if total_count <= 0:
            raise ValueError("total_count must be positive")
        if not self.cmd_buckets:
            return []

        rng = random.Random(seed)
        cmds = list(self.cmd_buckets.keys())
        per_cmd = max(1, total_count // len(cmds))
        selected = []

        for cmd in cmds:
            pool = self.cmd_buckets[cmd]
            if not pool:
                continue
            if len(pool) <= per_cmd:
                selected.extend(pool)
            else:
                selected.extend(rng.sample(pool, per_cmd))

        while len(selected) < total_count:
            cmd = rng.choice(cmds)
            pool = self.cmd_buckets[cmd]
            selected.append(pool[rng.randrange(len(pool))])

        rng.shuffle(selected)
        return selected[:total_count]

    def __getitem__(self, idx):
        seq, base_idx, cam_idx = self._resolve_indices(idx)
        frames = seq["frames"]
        seq_dir = seq["dir"]

        locs = np.stack(
            [np.asarray(frames[t]["loc"], dtype=np.float32) for t in range(base_idx, base_idx + self.T + 1)],
            axis=0,
        )
        rots = np.stack(
            [np.asarray(frames[t]["rot"], dtype=np.float32) for t in range(base_idx, base_idx + self.T)],
            axis=0,
        )
        spds = np.stack(
            [np.asarray(frames[t]["spd"], dtype=np.float32) for t in range(base_idx, base_idx + self.T)],
            axis=0,
        ).flatten()
        cmd_val = frames[base_idx].get("cmd", 0)
        cmd = int(self._scalarize(cmd_val))

        lbl_list = []
        for t in range(base_idx + 1, base_idx + self.T + 1):
            channels = []
            for k in range(12):
                rel = frames[t][f"lbl_{k:02d}"].lstrip("/")
                channels.append(_read_sem_gray(os.path.join(seq_dir, rel)))
            lbl = np.stack(channels, axis=-1)
            lbl_list.append(lbl.astype(np.uint8))
        lbls = np.stack(lbl_list, axis=0)

        def _resolve_path(path_str):
            rel = path_str.lstrip("/")
            return os.path.join(seq_dir, rel)

        frame = frames[base_idx]
        wide_rgb_path, narr_rgb_path = self._resolve_rgb_paths(frame, cam_idx)
        wide_sem_path = frame[f"wide_sem_{cam_idx}"]

        wide_rgb = _read_rgb(_resolve_path(wide_rgb_path))
        wide_sem = _read_sem_gray(_resolve_path(wide_sem_path))
        narr_rgb = _read_rgb(_resolve_path(narr_rgb_path))

        wide_sem = filter_sem(wide_sem, self.seg_channels)

        wide_rgb = wide_rgb[self.wide_crop_top :, :, :]
        wide_sem = wide_sem[self.wide_crop_top :, :]
        narr_rgb = narr_rgb[: -self.narr_crop_bottom, :, :]

        wide_rgb, wide_sem = self._resize_pair(wide_rgb, wide_sem, self.wide_scale)
        narr_rgb = self._resize_rgb(narr_rgb, self.narr_scale)

        if self.augmenter is not None:
            wide_rgb = self.augmenter(images=wide_rgb[None])[0]
            narr_rgb = self.augmenter(images=narr_rgb[None])[0]

        return wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, cmd

    def _resolve_indices(self, idx):
        if not self.multi_cam:
            idx *= len(self.camera_yaws)

        seq_idx, base_idx, cam_idx = self.index[idx]
        seq = self.seq_meta[seq_idx]
        return seq, base_idx, cam_idx

    @staticmethod
    def _load_frames(json_path):
        with open(json_path, "r") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            frames = raw
        elif isinstance(raw, dict):
            numeric_items = sorted(
                [(int(k), v) for k, v in raw.items() if isinstance(k, str) and k.isdigit()],
                key=lambda kv: kv[0],
            )
            frames = [v for _, v in numeric_items]
        else:
            raise ValueError(f"Unsupported JSON structure in {json_path}")

        return frames, len(frames)

    @staticmethod
    def _scalarize(value, default=0):
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            return value[0]
        if value is None:
            return default
        return value

    def _resize_rgb(self, rgb, scale):
        if scale is None or abs(scale - 1.0) < 1e-6:
            return rgb
        h, w = rgb.shape[:2]
        new_size = (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )
        return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)

    def _resize_sem(self, sem, scale):
        if scale is None or abs(scale - 1.0) < 1e-6:
            return sem
        h, w = sem.shape[:2]
        new_size = (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )
        return cv2.resize(sem, new_size, interpolation=cv2.INTER_NEAREST)

    def _resize_pair(self, rgb, sem, scale):
        if scale is None or abs(scale - 1.0) < 1e-6:
            return rgb, sem
        return self._resize_rgb(rgb, scale), self._resize_sem(sem, scale)

    @staticmethod
    def _swap_camera_prefix(path, swaps):
        dir_name, base = os.path.split(path)
        for src, dst in swaps:
            if base.startswith(src):
                return os.path.join(dir_name, dst + base[len(src):])
        for src, dst in swaps:
            if src in base:
                return os.path.join(dir_name, base.replace(src, dst, 1))
        return None

    @staticmethod
    def _normalize_camera_filename(path, target_prefix):
        if path is None:
            return None

        dir_name, base = os.path.split(path)
        replacements = [
            (f"{target_prefix}_rgb_", f"{target_prefix}_"),
            (f"{target_prefix}_rgb", f"{target_prefix}_"),
        ]

        for src, dst in replacements:
            if base.startswith(src):
                return os.path.join(dir_name, dst + base[len(src):])
            if src in base:
                return os.path.join(dir_name, base.replace(src, dst, 1))

        return path

    @staticmethod
    def _resolve_rgb_paths(frame, cam_idx):
        """
        Handle datasets where narrow-camera filenames accidentally live under the
        `wide_rgb_*` keys by swapping them and inferring the missing wide paths.
        """

        swaps_narr_to_wide = [
            ("narr_rgb_", "wide_rgb_"),
            ("narr_rgb", "wide_rgb"),
            ("narr_", "wide_"),
        ]
        wide_key = f"wide_rgb_{cam_idx}"
        narr_key = f"narr_rgb_{cam_idx}"

        wide_val = frame.get(wide_key)
        narr_val = frame.get(narr_key)

        if narr_val is None and wide_val is not None and "narr" in os.path.basename(wide_val):
            narr_val = wide_val
            swapped = JSONMainDataset._swap_camera_prefix(wide_val, swaps_narr_to_wide)
            if swapped is not None:
                wide_val = swapped

        if wide_val is None and narr_val is not None:
            swapped = JSONMainDataset._swap_camera_prefix(narr_val, swaps_narr_to_wide)
            if swapped is not None:
                wide_val = swapped

        if wide_val is None or narr_val is None:
            missing = []
            if wide_val is None:
                missing.append(wide_key)
            if narr_val is None:
                missing.append(narr_key)
            raise KeyError(f"Missing camera assets: {', '.join(missing)}")

        wide_val = JSONMainDataset._normalize_camera_filename(wide_val, "wide")
        narr_val = JSONMainDataset._normalize_camera_filename(narr_val, "narr")

        return wide_val, narr_val


class JSONLabeledMainDataset(JSONMainDataset):
    """
    JSON-backed equivalent of LabeledMainDataset that returns action-value labels
    and narrow segmentation masks for supervised fine-tuning.
    """

    def __getitem__(self, idx):
        seq, base_idx, cam_idx = self._resolve_indices(idx)
        frames = seq["frames"]
        seq_dir = seq["dir"]
        frame = frames[base_idx]

        def _resolve(path_str):
            rel = path_str.lstrip("/")
            return os.path.join(seq_dir, rel)

        wide_rgb_path, narr_rgb_path = self._resolve_rgb_paths(frame, cam_idx)
        wide_sem_path = frame[f"wide_sem_{cam_idx}"]
        narr_sem_key = f"narr_sem_{cam_idx}"
        if narr_sem_key not in frame:
            raise KeyError(f"Missing {narr_sem_key} for frame {base_idx}")

        wide_rgb = _read_rgb(_resolve(wide_rgb_path))
        narr_rgb = _read_rgb(_resolve(narr_rgb_path))
        wide_sem = _read_sem_gray(_resolve(wide_sem_path))
        narr_sem = _read_sem_gray(_resolve(frame[narr_sem_key]))

        wide_sem = filter_sem(wide_sem, self.seg_channels)
        narr_sem = filter_sem(narr_sem, self.seg_channels)

        wide_rgb = wide_rgb[self.wide_crop_top :, :, :]
        wide_sem = wide_sem[self.wide_crop_top :, :]
        narr_rgb = narr_rgb[: -self.narr_crop_bottom, :, :]
        narr_sem = narr_sem[: -self.narr_crop_bottom, :]

        wide_rgb, wide_sem = self._resize_pair(wide_rgb, wide_sem, self.wide_scale)
        narr_rgb, narr_sem = self._resize_pair(narr_rgb, narr_sem, self.narr_scale)

        if self.augmenter is not None:
            wide_rgb = self.augmenter(images=wide_rgb[None])[0]
            narr_rgb = self.augmenter(images=narr_rgb[None])[0]

        act_val = self._read_action_values(frame, cam_idx)

        spd = float(self._scalarize(frame.get("spd", 0.0)))
        cmd = int(self._scalarize(frame.get("cmd", 0)))

        return wide_rgb, wide_sem, narr_rgb, narr_sem, act_val, spd, cmd

    def _read_action_values(self, frame, cam_idx):
        candidates = [
            f"act{cam_idx}",
            f"act_{cam_idx}",
            f"act_val_{cam_idx}",
        ]
        data = None
        for key in candidates:
            if key in frame:
                data = frame[key]
                break
        if data is None:
            raise KeyError(f"Missing action values for camera {cam_idx}")

        act = np.asarray(data, dtype=np.float32)
        expected = self.num_cmds * (self.num_steers * self.num_throts + 1) * self.num_speeds
        if act.size != expected:
            raise ValueError(f"Unexpected action value size {act.size}, expected {expected}")
        act = act.reshape(self.num_cmds, self.num_steers * self.num_throts + 1, self.num_speeds)
        return act
