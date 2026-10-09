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
    ):
        super().__init__()
        self.max_latency_s = max_latency_s
        self.correction_scale = correction_scale

        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim + 1),
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Ensures the adapter initially changes nothing.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, latency_s):
        if latency_s.ndim == 1:
            latency_s = latency_s[:, None]

        normalized_latency = torch.clamp(
            latency_s / self.max_latency_s,
            min=0.0,
            max=1.0,
        )

        adapter_input = torch.cat([features, normalized_latency], dim=1)
        residual = torch.tanh(self.network(adapter_input))

        # At latency 0, this is exactly the original feature vector.
        return features + (
            self.correction_scale * normalized_latency * residual
        )
