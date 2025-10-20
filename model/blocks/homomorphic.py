import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class HomomorphicSeparate(nn.Module):
    def __init__(
        self,
        cutoff: float,
    ) -> None:
        super().__init__()
        cutoff_tensor = torch.tensor(data=float(cutoff), dtype=torch.float32)
        self.register_buffer(name="cutoff", tensor=cutoff_tensor)
        self._sigma_denom: float = math.sqrt(math.log(2.0))
        self._filter_cache: dict[
            tuple[int, int, torch.device, torch.dtype], Tensor
        ] = {}

    def _gaussian_lpf(
        self,
        size: tuple[int, int],
        reference: Tensor,
    ) -> Tensor:
        key = (size[0], size[1], reference.device, reference.dtype)
        cached = self._filter_cache.get(key)
        if cached is not None:
            return cached

        height, width = size
        device = reference.device
        dtype = reference.dtype

        fy: Tensor = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
        fx: Tensor = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        y, x = torch.meshgrid(fy, fx, indexing="ij")
        radius: Tensor = torch.hypot(input=x, other=y)

        cutoff = self.cutoff.to(device=device, dtype=dtype)
        sigma = cutoff / reference.new_tensor(data=self._sigma_denom)
        denominator = 2.0 * sigma * sigma
        h: Tensor = torch.exp(input=-(radius * radius) / denominator)
        h = h.unsqueeze(dim=0).unsqueeze(dim=0)
        self._filter_cache[key] = h
        return h

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        height, width = x.shape[-2:]

        x_log: Tensor = torch.log(input=torch.clamp(input=x, min=1e-5))

        x_fft: Tensor = torch.fft.fft2(x_log, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft)

        h: Tensor = self._gaussian_lpf(size=(height, width), reference=x)
        h = h.to(dtype=x_fft.dtype)
        low_fft: Tensor = x_fft * h

        low_log: torch.Tensor = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft), norm="ortho"
        ).real
        high_log: torch.Tensor = x_log - low_log

        il: Tensor = torch.exp(input=low_log)
        re: Tensor = torch.exp(input=high_log)
        return il, re
