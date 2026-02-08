import json
import os

import cv2
import numpy as np
import torch
from torch import nn


def enable_backbone_bn_training(
    model,
    train_backbone=True,
    train_backbone_bn=True,
    backbone_unfreeze_stage=None,
    train_bottleneck=True,
    train_seg_heads=True,
    train_act_head=False,
    bn_types=(nn.BatchNorm2d, nn.SyncBatchNorm),
):
    """
    Freeze most of the network, optionally enabling specific regions to train.
    """
    for param in model.parameters():
        param.requires_grad = False

    trainable = []
    stages = ["layer1", "layer2", "layer3", "layer4"]
    module_types = dict(model.named_modules())

    if backbone_unfreeze_stage is not None and backbone_unfreeze_stage not in stages:
        raise ValueError(f"Invalid backbone stage {backbone_unfreeze_stage}")

    def should_train_backbone_param(param_name):
        if not train_backbone:
            return False

        if param_name.startswith("backbone_wide."):
            suffix = param_name[len("backbone_wide.") :]
        elif param_name.startswith("backbone_narr."):
            suffix = param_name[len("backbone_narr.") :]
        else:
            return False

        if backbone_unfreeze_stage is None:
            return True

        target_idx = stages.index(backbone_unfreeze_stage)
        for stage in stages[target_idx:]:
            if suffix.startswith(stage):
                return True
        return False

    def is_bn_param(param_name):
        if "." not in param_name:
            return False
        module_name = param_name.rsplit(".", 1)[0]
        module = module_types.get(module_name)
        return isinstance(module, bn_types)

    for name, param in model.named_parameters():
        is_trainable = False
        if should_train_backbone_param(name):
            if train_backbone_bn or not is_bn_param(name):
                is_trainable = True
        elif train_bottleneck and name.startswith("bottleneck"):
            is_trainable = True
        elif train_seg_heads and name.startswith("seg_head"):
            is_trainable = True
        elif train_act_head and name.startswith("act_head"):
            is_trainable = True

        if is_trainable:
            param.requires_grad = True
            trainable.append(param)

    if not trainable:
        raise RuntimeError("No parameters were marked trainable. Check the model structure.")

    return trainable


def _load_balanced_cache(
    cache_path,
    expected_len,
    sample_size,
    sample_seed,
    data_dir,
    config_path,
):
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return None

    if payload.get("dataset_len") != expected_len:
        return None
    if payload.get("sample_size") != sample_size:
        return None
    if payload.get("seed") != sample_seed:
        return None
    if payload.get("data_dir") != os.path.abspath(data_dir):
        return None
    if payload.get("config_path") != os.path.abspath(config_path):
        return None

    indices = payload.get("indices")
    if not isinstance(indices, list):
        return None
    return indices


def _save_balanced_cache(
    cache_path,
    indices,
    dataset_len,
    sample_size,
    sample_seed,
    data_dir,
    config_path,
):
    if not cache_path:
        return
    payload = {
        "dataset_len": dataset_len,
        "sample_size": sample_size,
        "seed": sample_seed,
        "data_dir": os.path.abspath(data_dir),
        "config_path": os.path.abspath(config_path),
        "indices": indices,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f)


def _to_numpy_batch(batch):
    if batch is None:
        return None
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    return batch


def _resize_images(batch, scale):
    if batch is None or abs(scale - 1.0) < 1e-6:
        return batch
    np_batch = _to_numpy_batch(batch)
    resized = [
        cv2.resize(
            img,
            (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        for img in np_batch
    ]
    return np.stack(resized, axis=0)


def _resize_labels(batch, scale):
    if batch is None or abs(scale - 1.0) < 1e-6:
        return batch
    np_batch = _to_numpy_batch(batch)
    resized = [
        cv2.resize(
            lbl,
            (int(round(lbl.shape[1] * scale)), int(round(lbl.shape[0] * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
        for lbl in np_batch
    ]
    return np.stack(resized, axis=0)


def spd_lerp(v, x, min_val, max_val):
    """
    Vectorized version of RAILS.spd_lerp to interpolate over the speed dimension.
    v: (B, num_cmds, num_speeds, num_actions)
    x: (B,) tensor of speeds
    """
    D = v.shape[2]
    x = (x - min_val) / (max_val - min_val + 1e-8) * (D - 1)

    x0 = torch.floor(x).long().clamp(min=0, max=D - 1)
    x1 = torch.ceil(x).long().clamp(min=0, max=D - 1)
    w = (x - x0.float()).unsqueeze(-1).unsqueeze(-1)

    gather_shape = (v.shape[0], v.shape[1], 1, v.shape[3])
    idx0 = x0.view(v.shape[0], 1, 1, 1).expand(gather_shape)
    idx1 = x1.view(v.shape[0], 1, 1, 1).expand(gather_shape)

    v0 = torch.gather(v, 2, idx0).squeeze(2)
    v1 = torch.gather(v, 2, idx1).squeeze(2)
    return (1 - w) * v0 + w * v1


def action_logits(raw_logits, num_steers, num_throts):
    steer_logits = raw_logits[..., :num_steers]
    throt_logits = raw_logits[..., num_steers : num_steers + num_throts]
    brake_logits = raw_logits[..., -1:]

    steer_logits = steer_logits.repeat(1, 1, 1, num_throts)
    throt_logits = throt_logits.repeat_interleave(num_steers, -1)

    act_logits = torch.cat([steer_logits + throt_logits, brake_logits], dim=-1)

    return act_logits


def _forward_with_features(model, wide_rgbs, narr_rgbs, spds, cfg):
    wide_embed = model.backbone_wide(model.normalize(wide_rgbs / 255.0))
    wide_seg = model.seg_head_wide(wide_embed)

    if model.two_cam:
        narr_embed = model.backbone_narr(model.normalize(narr_rgbs / 255.0))
        narr_seg = model.seg_head_narr(narr_embed)
        pooled = torch.cat(
            [
                wide_embed.mean(dim=[2, 3]),
                model.bottleneck_narr(narr_embed.mean(dim=[2, 3])),
            ],
            dim=1,
        )
    else:
        narr_embed = None
        narr_seg = None
        pooled = wide_embed.mean(dim=[2, 3])

    if cfg["all_speeds"]:
        act = model.act_head(pooled).view(
            -1,
            model.num_cmds,
            model.num_speeds,
            model.num_steers + model.num_throts + 1,
        )
        act = action_logits(act, model.num_steers, model.num_throts)
    else:
        speed_feat = model.spd_encoder(spds[:, None])
        act = model.act_head(torch.cat([pooled, speed_feat], dim=1)).view(
            -1,
            model.num_cmds,
            1,
            model.num_steers + model.num_throts + 1,
        )
        act = action_logits(act, model.num_steers, model.num_throts).squeeze(2)

    return dict(
        act=act,
        wide_seg=wide_seg,
        narr_seg=narr_seg,
        pooled=pooled,
        wide_embed=wide_embed,
        narr_embed=narr_embed,
    )
