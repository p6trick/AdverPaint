"""Generator architecture used by the released AdverPaint checkpoint.

The network implementation is based on AOT-GAN, which AdverPaint builds on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InpaintGenerator(nn.Module):
    def __init__(self, rates=(1, 2, 4, 8), block_num=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(4, 64, 7), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(True),
        )
        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(block_num)])
        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
        )

    def forward(self, image, mask):
        return torch.tanh(self.decoder(self.middle(self.encoder(torch.cat([image, mask], dim=1)))))


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, channels, rates):
        super().__init__()
        self.rates = rates
        # These attribute names intentionally match the original training code,
        # so the released checkpoint can be loaded with strict=True.
        for index, rate in enumerate(rates):
            setattr(self, f"block{index:02d}", nn.Sequential(
                nn.ReflectionPad2d(rate),
                nn.Conv2d(channels, channels // 4, 3, dilation=rate),
                nn.ReLU(True),
            ))
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3))

    def forward(self, x):
        transformed = self.fuse(torch.cat([getattr(self, f"block{index:02d}")(x) for index in range(len(self.rates))], dim=1))
        gate = torch.sigmoid(_layer_norm(self.gate(x)))
        return x * (1 - gate) + transformed * gate


def _layer_norm(features):
    mean = features.mean((2, 3), keepdim=True)
    std = features.std((2, 3), keepdim=True) + 1e-9
    return 5 * (2 * (features - mean) / std - 1)
