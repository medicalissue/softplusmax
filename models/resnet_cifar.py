"""ResNet for CIFAR (He et al. 2016, "Deep Residual Learning for Image
Recognition", Section 4.2).

ResNet-{20, 32, 44, 56, 110}: 6n+2 layers with three stages of n blocks
(channel widths 16/32/64). Standard CIFAR variant — NOT to be confused
with ImageNet ResNets which start from a 7×7 stem and 4 stages.

Used widely as the standard CIFAR benchmark. Numbers in the original
paper:
  ResNet-20: 91.25%  CIFAR-10
  ResNet-56: 93.03%
  (CIFAR-100 numbers vary by recipe; ResNet-56 ≈ 71-72% with strong aug.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = _conv3x3(out_c, out_c, 1)
        self.bn2 = nn.BatchNorm2d(out_c)

        # 1x1 projection if stride or channel mismatch.
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCIFAR(nn.Module):
    """6n+2 CIFAR ResNet.

    Args:
        n: blocks per stage. n=3 → 20-layer; n=9 → 56-layer.
        num_classes: 10 for CIFAR-10, 100 for CIFAR-100.
        return_features: if True, forward returns (logits, penultimate_features).
                         Useful for projection heads in SimCLR.
    """

    def __init__(self, n: int, num_classes: int = 100, return_features: bool = False):
        super().__init__()
        self.n = int(n)
        self.return_features = bool(return_features)
        self.num_classes = int(num_classes)

        # Stem: 3×3 conv from 3 → 16, no maxpool (CIFAR is 32×32).
        self.stem = nn.Sequential(
            _conv3x3(3, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_stage(16, 16, n, stride=1)   # 32×32
        self.layer2 = self._make_stage(16, 32, n, stride=2)   # 16×16
        self.layer3 = self._make_stage(32, 64, n, stride=2)   # 8×8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_c: int, out_c: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)        # [B, 64]
        logits = self.fc(x)
        if self.return_features:
            return logits, x
        return logits


def resnet20(num_classes: int = 100, **kw) -> ResNetCIFAR:
    return ResNetCIFAR(n=3, num_classes=num_classes, **kw)


def resnet56(num_classes: int = 100, **kw) -> ResNetCIFAR:
    return ResNetCIFAR(n=9, num_classes=num_classes, **kw)


if __name__ == "__main__":
    for name, fn in [("resnet20", resnet20), ("resnet56", resnet56)]:
        m = fn(num_classes=100)
        n_p = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"{name}: params={n_p:,}  out={tuple(y.shape)}")
