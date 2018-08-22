"""3D U-Net for volumetric segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):  # can try (16, 32, 64, 128) for less memory
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(ConvBlock3D(prev, f))
            self.pools.append(nn.MaxPool3d(2))
            prev = f
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(f * 2, f, 2, stride=2))
            self.decoders.append(ConvBlock3D(f * 2, f))
        self.final = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.final(x))

class UNet2D(nn.Module):
    """Slice-by-slice version."""
    def __init__(self, in_ch=1, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        prev = in_ch
        for f in features:
            self.encoders.append(self._block(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(self._block(f * 2, f))
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def _block(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.final(x))
