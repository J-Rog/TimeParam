import torch
import torch.nn.functional as F
from torch import nn


class ResolutionAwareBatchNorm2d(nn.Module):
    """BatchNorm2d variant that keeps per-resolution running statistics."""

    def __init__(self, bn_module: nn.BatchNorm2d):
        super().__init__()
        if isinstance(bn_module, ResolutionAwareBatchNorm2d):
            raise TypeError("Module is already ResolutionAwareBatchNorm2d")
        if not isinstance(bn_module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            raise TypeError("Expected a BatchNorm module")

        self.num_features = bn_module.num_features
        self.eps = bn_module.eps
        self.momentum = bn_module.momentum
        self.affine = bn_module.affine
        self.track_running_stats = bn_module.track_running_stats

        if self.affine:
            self.weight = nn.Parameter(bn_module.weight.detach().clone())
            self.bias = nn.Parameter(bn_module.bias.detach().clone())
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer(
                "default_running_mean", bn_module.running_mean.detach().clone()
            )
            self.register_buffer(
                "default_running_var", bn_module.running_var.detach().clone()
            )
            self.register_buffer(
                "default_num_batches_tracked",
                bn_module.num_batches_tracked.detach().clone()
                if bn_module.num_batches_tracked is not None
                else torch.tensor(0, dtype=torch.long),
            )
        else:
            self.register_buffer("default_running_mean", None)
            self.register_buffer("default_running_var", None)
            self.register_buffer("default_num_batches_tracked", None)

    def forward(self, x):
        if not self.track_running_stats:
            return F.batch_norm(
                x,
                None,
                None,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )

        key = f"{x.shape[-2]}x{x.shape[-1]}"
        running_mean, running_var, num_batches = self._stats_for_key(key)
        momentum = self.momentum
        if momentum is None:
            if self.training and self.track_running_stats:
                if num_batches is not None:
                    momentum = 1.0 / float(num_batches.item() + 1)
                else:
                    momentum = 0.0
            else:
                momentum = 0.0

        out = F.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            self.training,
            momentum,
            self.eps,
        )

        if self.training and num_batches is not None:
            num_batches.add_(1)
        return out

    def _stats_for_key(self, key):
        mean_name = f"running_mean_{key}"
        var_name = f"running_var_{key}"
        num_name = f"num_batches_tracked_{key}"

        if mean_name not in self._buffers:
            self.register_buffer(
                mean_name,
                self.default_running_mean.detach().clone()
                if self.default_running_mean is not None
                else None,
            )
            self.register_buffer(
                var_name,
                self.default_running_var.detach().clone()
                if self.default_running_var is not None
                else None,
            )
            self.register_buffer(
                num_name,
                self.default_num_batches_tracked.detach().clone()
                if self.default_num_batches_tracked is not None
                else torch.tensor(0, dtype=torch.long),
            )

        return getattr(self, mean_name), getattr(self, var_name), getattr(self, num_name)

    def reset_running_stats(self):
        if not self.track_running_stats:
            return
        if self.default_running_mean is not None:
            self.default_running_mean.zero_()
        if self.default_running_var is not None:
            self.default_running_var.fill_(1)
        if self.default_num_batches_tracked is not None:
            self.default_num_batches_tracked.zero_()
        for name, buffer in list(self._buffers.items()):
            if name.startswith("running_mean_") and buffer is not None:
                buffer.zero_()
            elif name.startswith("running_var_") and buffer is not None:
                buffer.fill_(1)
            elif name.startswith("num_batches_tracked_") and buffer is not None:
                buffer.zero_()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        dyn_keys = []
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            if suffix.startswith("running_mean_") or suffix.startswith("running_var_"):
                dyn_keys.append((suffix, key))
            elif suffix.startswith("num_batches_tracked_"):
                dyn_keys.append((suffix, key))

        for suffix, full_key in dyn_keys:
            value = state_dict.pop(full_key)
            if not hasattr(self, suffix):
                self.register_buffer(suffix, value.clone())
            else:
                setattr(self, suffix, value.clone())

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def convert_bn_to_resaware(module):
    """Recursively swap BatchNorm2d layers for ResolutionAwareBatchNorm2d."""

    for name, child in list(module.named_children()):
        if isinstance(child, ResolutionAwareBatchNorm2d):
            continue
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            setattr(module, name, ResolutionAwareBatchNorm2d(child))
        else:
            convert_bn_to_resaware(child)


def state_dict_has_resaware_stats(state_dict):
    return any(
        ".running_mean_" in key
        or ".running_var_" in key
        or ".num_batches_tracked_" in key
        for key in state_dict.keys()
    )


__all__ = [
    "ResolutionAwareBatchNorm2d",
    "convert_bn_to_resaware",
    "state_dict_has_resaware_stats",
]
