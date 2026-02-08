#!/usr/bin/env python3
"""Fine-tune CameraModel with resolution-aware BatchNorm on multi-resolution data."""

import os
import math
import yaml

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from finetune_common import (
    _forward_with_features,
    _load_balanced_cache,
    _resize_images,
    _resize_labels,
    _save_balanced_cache,
    enable_backbone_bn_training,
    spd_lerp,
)
from pcla_agents.wor.rails.models import CameraModel
from pcla_agents.wor.rails.models.resaware_bn import (
    ResolutionAwareBatchNorm2d,
    convert_bn_to_resaware,
    state_dict_has_resaware_stats,
)
from pcla_agents.wor.rails.datasets.json_main_dataset import JSONLabeledMainDataset


###############################
# Editable hyper-parameters
###############################
CONFIG_PATH = "pcla_agents/wor_pretrained/nocrash_weights/config_nocrash.yaml"
DEFAULT_DATA_DIR = "data/main_trajs_converted"
DATA_DIR = os.environ.get("WOR_DATA_DIR", DEFAULT_DATA_DIR)
CHECKPOINT_PATH = "pcla_agents/wor_pretrained/nocrash_weights/main_model_16.th"
TEACHER_CHECKPOINT = CHECKPOINT_PATH
OUTPUT_PATH = "outputs/resaware_bn_finetuned.th"

DEVICE = "cuda"
BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 2
LEARNING_RATE = 1.5e-4
USE_AUGMENT = True
WIDE_SCALE = 1.0
NARR_SCALE = 1.0
RESOLUTION_SCALES = [
    (1.0, 1.0),
    (0.75, 0.75),
]
TRAIN_BACKBONE = True
TRAIN_BACKBONE_BN = True
BACKBONE_UNFREEZE_SCHEDULE = [
    {"stage": "layer4", "steps": 200},
    {"stage": "layer3", "steps": 200},
    {"stage": "layer2", "steps": 200},
    {"stage": "layer1", "steps": 200},
]
TRAIN_BOTTLENECK = False
TRAIN_SEG_HEADS = True
TRAIN_ACT_HEAD = False
CALIBRATE_BN = True
CALIBRATION_MAX_BATCHES = 200
USE_BALANCED_SAMPLING = True
TEACHER_KL_WEIGHT = 1.0
TEACHER_FEAT_WEIGHT = 1.0
BALANCED_CACHE_PATH = "outputs/balanced_indices.json"

SAMPLE_SIZE = 50000  # set to None to use all frames
SAMPLE_SEED = 0
MAX_STEPS = None  # set to an int to cap optimizer steps
###############################


def _normalize_resolution_scales():
    normalized = []
    for entry in RESOLUTION_SCALES:
        if isinstance(entry, (int, float)):
            normalized.append((float(entry), float(entry)))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            normalized.append((float(entry[0]), float(entry[1])))
        else:
            raise ValueError("RESOLUTION_SCALES entries must be floats or len-2 iterables")
    if not normalized:
        normalized.append((1.0, 1.0))
    return normalized


def resolution_scale_pairs():
    return _normalize_resolution_scales()


def _load_checkpoint(path, device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
        return payload["state_dict"], metadata
    return payload, {}


def _adapt_state_dict_for_resaware(state_dict):
    """
    Make plain BN checkpoints load cleanly into ResolutionAwareBatchNorm2d modules.
    """
    if state_dict_has_resaware_stats(state_dict):
        return state_dict

    adapted = {}
    for key, value in state_dict.items():
        if key.endswith(".running_mean"):
            adapted[key[: -len(".running_mean")] + ".default_running_mean"] = value
        elif key.endswith(".running_var"):
            adapted[key[: -len(".running_var")] + ".default_running_var"] = value
        elif key.endswith(".num_batches_tracked"):
            adapted[key[: -len(".num_batches_tracked")] + ".default_num_batches_tracked"] = value
        else:
            adapted[key] = value
    return adapted


def calibrate_batch_norm(model, dataset, cfg, device, max_batches=None):
    if len(dataset) == 0:
        return

    scale_pairs = resolution_scale_pairs()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    bn_layers = [
        module
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, ResolutionAwareBatchNorm2d))
    ]
    original_momenta = [layer.momentum for layer in bn_layers]
    for layer in bn_layers:
        if hasattr(layer, "reset_running_stats"):
            layer.reset_running_stats()
        layer.momentum = None

    was_training = model.training
    model.train()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            for w_scale, n_scale in scale_pairs:
                _forward_for_bn(model, batch, cfg, device, w_scale, n_scale)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    for layer, momentum in zip(bn_layers, original_momenta):
        layer.momentum = momentum
    model.train(was_training)


def _forward_for_bn(model, batch, cfg, device, w_scale=1.0, n_scale=1.0):
    wide_rgb, _, narr_rgb, _, _, spd, _ = batch

    wide_imgs = _resize_images(wide_rgb, w_scale)
    narr_imgs = _resize_images(narr_rgb, n_scale)

    wide_rgbs = (
        torch.as_tensor(wide_imgs, dtype=torch.float32, device=device)
        .permute(0, 3, 1, 2)
    )
    narr_rgbs = (
        torch.as_tensor(narr_imgs, dtype=torch.float32, device=device)
        .permute(0, 3, 1, 2)
    )
    spds = torch.as_tensor(spd, dtype=torch.float32, device=device)

    model(
        wide_rgbs,
        narr_rgbs,
        spd=None if cfg["all_speeds"] else spds,
    )


def _prepare_scaled_tensors(wide_rgb, narr_rgb, wide_sem, narr_sem, w_scale, n_scale, device):
    wide_imgs = _resize_images(wide_rgb, w_scale)
    narr_imgs = _resize_images(narr_rgb, n_scale)
    wide_labels = _resize_labels(wide_sem, w_scale)
    narr_labels = _resize_labels(narr_sem, n_scale)

    wide_rgbs = (
        torch.as_tensor(wide_imgs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        if wide_imgs is not None
        else None
    )
    narr_rgbs = (
        torch.as_tensor(narr_imgs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        if narr_imgs is not None
        else None
    )
    wide_sems = (
        torch.as_tensor(wide_labels, dtype=torch.long, device=device)
        if wide_labels is not None
        else None
    )
    narr_sems = (
        torch.as_tensor(narr_labels, dtype=torch.long, device=device)
        if narr_labels is not None
        else None
    )

    return wide_rgbs, narr_rgbs, wide_sems, narr_sems


def train_step(batch, model, teacher, cfg, optimizer, device):
    wide_rgb, wide_sem, narr_rgb, narr_sem, act_val, spd, cmd = batch

    act_vals = torch.as_tensor(act_val, dtype=torch.float32, device=device).permute(0, 1, 3, 2)
    spds = torch.as_tensor(spd, dtype=torch.float32, device=device)
    cmds = torch.as_tensor(cmd, dtype=torch.long, device=device)

    act_probs = F.softmax(act_vals / cfg["temperature"], dim=3)
    act_probs_interp = None
    if not cfg["all_speeds"]:
        act_probs_interp = spd_lerp(act_probs, spds, cfg["min_speeds"], cfg["max_speeds"])

    teacher_out = None
    if teacher is not None and (TEACHER_KL_WEIGHT > 0 or TEACHER_FEAT_WEIGHT > 0):
        with torch.no_grad():
            teacher_wide_rgbs = (
                torch.as_tensor(wide_rgb, dtype=torch.float32, device=device)
                .permute(0, 3, 1, 2)
            )
            teacher_narr_rgbs = (
                torch.as_tensor(narr_rgb, dtype=torch.float32, device=device)
                .permute(0, 3, 1, 2)
            )
            teacher_out = _forward_with_features(teacher, teacher_wide_rgbs, teacher_narr_rgbs, spds, cfg)

    scale_pairs = resolution_scale_pairs()
    num_scales = len(scale_pairs)
    total_loss = 0.0
    total_act = 0.0
    total_seg = 0.0
    total_teacher = 0.0

    for w_scale, n_scale in scale_pairs:
        wide_rgbs, narr_rgbs, wide_sems, narr_sems = _prepare_scaled_tensors(
            wide_rgb,
            narr_rgb,
            wide_sem,
            narr_sem,
            w_scale,
            n_scale,
            device,
        )

        student_out = _forward_with_features(model, wide_rgbs, narr_rgbs, spds, cfg)
        act_outputs = student_out["act"]
        wide_seg_outputs = student_out["wide_seg"]
        narr_seg_outputs = student_out["narr_seg"]

        teacher_loss = torch.tensor(0.0, device=device)
        if teacher_out is not None:
            if TEACHER_KL_WEIGHT > 0:
                student_log = F.log_softmax(act_outputs, dim=-1)
                teacher_prob = F.softmax(teacher_out["act"], dim=-1)
                teacher_loss = teacher_loss + TEACHER_KL_WEIGHT * F.kl_div(
                    student_log, teacher_prob, reduction="batchmean"
                )

            if TEACHER_FEAT_WEIGHT > 0:
                teacher_loss = teacher_loss + TEACHER_FEAT_WEIGHT * F.mse_loss(
                    student_out["pooled"], teacher_out["pooled"].detach()
                )

        if cfg["all_speeds"]:
            act_loss = F.kl_div(
                F.log_softmax(act_outputs, dim=3),
                act_probs,
                reduction="none",
            ).mean(dim=[2, 3])
        else:
            act_loss = F.kl_div(
                F.log_softmax(act_outputs, dim=2),
                act_probs_interp,
                reduction="none",
            ).mean(dim=2)

        turn_loss = (act_loss[:, 0] + act_loss[:, 1] + act_loss[:, 2] + act_loss[:, 3]) / 4
        lane_loss = (act_loss[:, 4] + act_loss[:, 5] + act_loss[:, 3]) / 3
        foll_loss = act_loss[:, 3]

        is_turn = (cmds == 0) | (cmds == 1) | (cmds == 2)
        is_lane = (cmds == 4) | (cmds == 5)

        action_loss = torch.mean(
            torch.where(is_turn, turn_loss, foll_loss)
            + torch.where(is_lane, lane_loss, foll_loss)
        )

        target_hw = wide_sems.shape[-2:]
        pred_wide_seg = F.interpolate(wide_seg_outputs, size=target_hw, mode="nearest")
        seg_loss = F.cross_entropy(pred_wide_seg, wide_sems)

        if cfg["use_narr_cam"] and narr_seg_outputs is not None and narr_sems is not None:
            narr_target_hw = narr_sems.shape[-2:]
            pred_narr_seg = F.interpolate(
                narr_seg_outputs, size=narr_target_hw, mode="nearest"
            )
            seg_loss = (seg_loss + F.cross_entropy(pred_narr_seg, narr_sems)) / 2

        scale_loss = action_loss + cfg["seg_weight"] * seg_loss + teacher_loss

        total_loss = total_loss + scale_loss
        total_act = total_act + action_loss
        total_seg = total_seg + seg_loss
        total_teacher = total_teacher + teacher_loss

    avg_loss = total_loss / num_scales
    avg_act = total_act / num_scales
    avg_seg = total_seg / num_scales
    avg_teacher = total_teacher / num_scales

    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    return {
        "loss": float(avg_loss),
        "act_loss": float(avg_act),
        "seg_loss": float(avg_seg),
        "teacher_loss": float(avg_teacher),
    }


def main():
    device = torch.device(DEVICE)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {DATA_DIR}\n"
            "Set WOR_DATA_DIR to your dataset root, e.g.\n"
            "  export WOR_DATA_DIR='/path/to/main_trajs_converted'"
        )

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    train_dataset = JSONLabeledMainDataset(
        DATA_DIR,
        CONFIG_PATH,
        use_augment=USE_AUGMENT,
        wide_scale=WIDE_SCALE,
        narr_scale=NARR_SCALE,
    )

    base_dataset_len = len(train_dataset)
    subset_indices = None
    if SAMPLE_SIZE is not None:
        if SAMPLE_SIZE <= 0:
            raise ValueError("SAMPLE_SIZE must be positive when provided")
        if SAMPLE_SIZE > len(train_dataset):
            raise ValueError(
                f"SAMPLE_SIZE {SAMPLE_SIZE} exceeds dataset length {len(train_dataset)}"
            )
        if USE_BALANCED_SAMPLING and hasattr(train_dataset, "balanced_indices"):
            cached_indices = _load_balanced_cache(
                BALANCED_CACHE_PATH,
                base_dataset_len,
                SAMPLE_SIZE,
                SAMPLE_SEED,
                DATA_DIR,
                CONFIG_PATH,
            )
            if cached_indices is not None:
                subset_indices = cached_indices
                print(
                    f"Loaded {len(subset_indices)} balanced indices from cache {BALANCED_CACHE_PATH}"
                )
            else:
                subset_indices = train_dataset.balanced_indices(
                    total_count=SAMPLE_SIZE, seed=SAMPLE_SEED
                )
                print(
                    f"Sampling {len(subset_indices)} balanced frames from the JSON dataset"
                )
                _save_balanced_cache(
                    BALANCED_CACHE_PATH,
                    subset_indices,
                    base_dataset_len,
                    SAMPLE_SIZE,
                    SAMPLE_SEED,
                    DATA_DIR,
                    CONFIG_PATH,
                )
        else:
            generator = torch.Generator().manual_seed(SAMPLE_SEED)
            subset_indices = (
                torch.randperm(len(train_dataset), generator=generator)[: SAMPLE_SIZE].tolist()
            )
            print(f"Sampling {len(subset_indices)} random frames from the JSON dataset")
        train_dataset = Subset(train_dataset, subset_indices)

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    model = CameraModel(cfg).to(device)
    convert_bn_to_resaware(model)
    student_state_dict, _ = _load_checkpoint(CHECKPOINT_PATH, device)
    student_state_dict = _adapt_state_dict_for_resaware(student_state_dict)
    model.load_state_dict(student_state_dict, strict=False)
    model.train()

    teacher_model = None
    if TEACHER_KL_WEIGHT > 0 or TEACHER_FEAT_WEIGHT > 0:
        teacher_model = CameraModel(cfg).to(device)
        convert_bn_to_resaware(teacher_model)
        teacher_state_dict, _ = _load_checkpoint(TEACHER_CHECKPOINT, device)
        teacher_state_dict = _adapt_state_dict_for_resaware(teacher_state_dict)
        teacher_model.load_state_dict(teacher_state_dict, strict=False)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    steps_per_epoch = len(loader)
    raw_schedule = BACKBONE_UNFREEZE_SCHEDULE or [{"stage": None, "epochs": EPOCHS}]
    stage_entries = []
    for cfg_stage in raw_schedule:
        stage = cfg_stage.get("stage")
        steps = cfg_stage.get("steps")
        if steps is None:
            stage_epochs = cfg_stage.get("epochs", 0)
            steps = int(math.ceil(stage_epochs * steps_per_epoch))
        if steps <= 0:
            continue
        stage_entries.append({"stage": stage, "steps": steps})

    if not stage_entries:
        stage_entries = [{"stage": None, "steps": EPOCHS * steps_per_epoch}]

    def activate_stage(entry, idx):
        stage_name = entry["stage"]
        print(
            f"Stage {idx+1}/{len(stage_entries)}: unfreezing "
            f"{stage_name or 'no backbone layers'} for {entry['steps']} step(s)"
        )
        params = enable_backbone_bn_training(
            model,
            train_backbone=TRAIN_BACKBONE and stage_name is not None,
            train_backbone_bn=TRAIN_BACKBONE_BN,
            backbone_unfreeze_stage=stage_name,
            train_bottleneck=TRAIN_BOTTLENECK,
            train_seg_heads=TRAIN_SEG_HEADS,
            train_act_head=TRAIN_ACT_HEAD,
            bn_types=(nn.BatchNorm2d, nn.SyncBatchNorm, ResolutionAwareBatchNorm2d),
        )
        return torch.optim.Adam(params, lr=LEARNING_RATE)

    stage_idx = 0
    stage_steps_remaining = stage_entries[0]["steps"]
    optimizer = activate_stage(stage_entries[0], 0)
    global_step = 0
    stop_training = False

    for epoch in range(EPOCHS):
        if stop_training:
            break
        for batch in loader:
            stats = train_step(batch, model, teacher_model, cfg, optimizer, device)
            global_step += 1
            stage_steps_remaining -= 1

            if global_step % 20 == 0:
                print(
                    f"[epoch {epoch+1}] step {global_step} "
                    f"loss={stats['loss']:.4f} act={stats['act_loss']:.4f} seg={stats['seg_loss']:.4f}"
                    f" teacher={stats['teacher_loss']:.4f}"
                )

            if stage_steps_remaining == 0:
                stage_idx += 1
                if stage_idx >= len(stage_entries):
                    stop_training = True
                    break
                optimizer = activate_stage(stage_entries[stage_idx], stage_idx)
                stage_steps_remaining = stage_entries[stage_idx]["steps"]

            if MAX_STEPS is not None and global_step >= MAX_STEPS:
                stop_training = True
                break

    if CALIBRATE_BN:
        calib_dataset = JSONLabeledMainDataset(
            DATA_DIR,
            CONFIG_PATH,
            use_augment=False,
            wide_scale=WIDE_SCALE,
            narr_scale=NARR_SCALE,
        )
        if subset_indices is not None:
            calib_dataset = Subset(calib_dataset, subset_indices)
        print("Running BN calibration sweep...")
        calibrate_batch_norm(
            model,
            calib_dataset,
            cfg,
            device,
            max_batches=CALIBRATION_MAX_BATCHES,
        )

    checkpoint_payload = {
        "state_dict": model.state_dict(),
        "resaware_bn": True,
        "resolution_scales": resolution_scale_pairs(),
        "wide_scale": WIDE_SCALE,
        "narr_scale": NARR_SCALE,
    }
    torch.save(checkpoint_payload, OUTPUT_PATH)
    print(f"Saved fine-tuned weights to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
