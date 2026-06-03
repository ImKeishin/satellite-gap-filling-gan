from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.block(x)
        if skip is None:
            return x
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


class GeneratorUNet(nn.Module):
    """Pix2Pix-style U-Net generator for satellite image gap filling.

    Input:  masked image tensor, shape (N, C, H, W)
    Output: reconstructed image tensor, shape (N, out_channels, H, W)

    The depth is intentionally moderate, so it works with common patch sizes
    such as 64x64, 128x128 and 256x256.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels

        self.down1 = DownBlock(in_channels, b, normalize=False)
        self.down2 = DownBlock(b, b * 2)
        self.down3 = DownBlock(b * 2, b * 4)
        self.down4 = DownBlock(b * 4, b * 8)
        self.down5 = DownBlock(b * 8, b * 8, normalize=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(b * 8, b * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = UpBlock(b * 8, b * 8, dropout=0.5)
        self.up2 = UpBlock(b * 16, b * 8, dropout=0.5)
        self.up3 = UpBlock(b * 16, b * 4)
        self.up4 = UpBlock(b * 8, b * 2)
        self.up5 = UpBlock(b * 4, b)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(b * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        z = self.bottleneck(d5)

        u1 = self.up1(z, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        out = self.final(u5)

        if out.shape[-2:] != original_size:
            out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)
        return out
