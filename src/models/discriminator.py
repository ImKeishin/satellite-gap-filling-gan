from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Conditional PatchGAN discriminator.

    The temporal condition and reconstructed target can have different channel
    counts. For example, with three Sentinel-2 dates:
        condition: 3 * 13 = 39 channels
        target:              13 channels
    The discriminator learns to distinguish real from fake targets, both
    conditioned by the masked input image."""

    def __init__(self,
                 condition_channels: int = 1,
                 target_channels: int = 1,
                 base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels

        def block(in_filters: int, out_filters: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(condition_channels + target_channels, b, normalize=False),
            *block(b, b * 2),
            *block(b * 2, b * 4),
            *block(b * 4, b * 8),
            nn.Conv2d(b * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, target], dim=1)
        return self.model(x)
