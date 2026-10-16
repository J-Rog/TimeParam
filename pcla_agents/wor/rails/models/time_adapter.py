import torch
from torch import nn


class DelayFeatureAdapter(nn.Module):
    """
    Learns a bounded, latency-dependent residual correction to a frozen
    visual feature vector.
    """

    def __init__(
        self,
        feature_dim=512,
        hidden_dim=128,
        max_latency_s=0.25,
        correction_scale=0.10,
        ego_dim=0,
    ):
        super().__init__()
        if max_latency_s <= 0 or correction_scale <= 0 or ego_dim not in (0, 1, 5):
            raise ValueError("Positive latency/scale and ego_dim in {0, 1, 5} required")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.ego_dim = ego_dim
        self.max_latency_s = max_latency_s
        self.correction_scale = correction_scale

        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim + 1 + ego_dim),
            nn.Linear(feature_dim + 1 + ego_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Ensures the adapter initially changes nothing.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, latency_s, ego_state=None):
        if latency_s.ndim == 1:
            latency_s = latency_s[:, None]

        normalized_latency = torch.clamp(
            latency_s / self.max_latency_s,
            min=0.0,
            max=1.0,
        )

        adapter_input = torch.cat([features, normalized_latency], dim=1)
        if self.ego_dim:
            if ego_state is None or ego_state.shape != (features.shape[0], self.ego_dim):
                raise ValueError("Adapter requires current ego state matching ego_dim")
            # m/s, rad/s, steer, throttle, brake. Same units at train and runtime.
            scales = features.new_tensor([10.0, 1.0, 1.0, 1.0, 1.0][:self.ego_dim])
            adapter_input = torch.cat([adapter_input, ego_state / scales], dim=1)
        residual = torch.tanh(self.network(adapter_input))

        # At latency 0, this is exactly the original feature vector.
        return features + (
            self.correction_scale * normalized_latency * residual
        )

    def config(self):
        return dict(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim,
                    max_latency_s=self.max_latency_s, correction_scale=self.correction_scale,
                    ego_dim=self.ego_dim)


def load_delay_adapter(payload, feature_dim, device):
    """Load new self-describing checkpoints and the original visual-only format."""
    config = payload.get("adapter_config", dict(
        feature_dim=payload.get("feature_dim", 512),
        max_latency_s=payload.get("max_latency_s", 0.25)))
    if config["feature_dim"] != feature_dim:
        raise ValueError("Adapter feature dimension does not match the frozen driver")
    adapter = DelayFeatureAdapter(**config).to(device)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    return adapter
