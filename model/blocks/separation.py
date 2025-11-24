import torch
import torch.nn as nn
from torch import Tensor


class SeparationBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        sigma: float,
    ) -> None:
        super().__init__()

        pad_size = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding=pad_size)

        kernel_1d = self._get_gaussian_kernel_1d(kernel_size=kernel_size, sigma=sigma)

        self.conv_y = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(kernel_size, 1), bias=False
        )
        self.conv_x = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(1, kernel_size), bias=False
        )

        with torch.no_grad():
            self.conv_y.weight.copy_(other=kernel_1d.view(1, 1, kernel_size, 1))
            self.conv_x.weight.copy_(other=kernel_1d.view(1, 1, 1, kernel_size))

        self.conv_y.weight.requires_grad = False
        self.conv_x.weight.requires_grad = False

        self.EPS = 1e-6

    def _get_gaussian_kernel_1d(
        self,
        kernel_size: int,
        sigma: float,
    ) -> Tensor:
        x_cord = torch.arange(end=kernel_size).float()
        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0
        kernel_1d = torch.exp(input=-((x_cord - mean) ** 2) / (2 * variance))
        kernel_1d = kernel_1d / kernel_1d.sum()
        return kernel_1d

    def forward(
        self,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        y_log = torch.log(input=y.float().clamp(min=self.EPS))

        low_log = self.pad(y_log)
        low_log = self.conv_y(low_log)
        low_log = self.conv_x(low_log)

        re_log = y_log - low_log

        il = torch.exp(input=low_log)
        re = torch.exp(input=re_log).clamp(min=0.0, max=1.0)

        min_val = il.amin(dim=(2, 3), keepdim=True)
        max_val = il.amax(dim=(2, 3), keepdim=True)

        il = (il - min_val) / (max_val - min_val + self.EPS)

        return il.type_as(other=y), re.type_as(other=y)
