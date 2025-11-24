from typing import Any

import torch.nn as nn
from torch import Tensor


class MeanAbsoluteError(nn.L1Loss):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        return super().forward(input=input, target=target)


class MeanSquaredError(nn.MSELoss):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        return super().forward(input=input, target=target)
