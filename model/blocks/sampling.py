import torch.nn as nn
from torch import Tensor


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.conv(x)
