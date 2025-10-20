import torch.nn as nn
from torch import Tensor

from model.blocks.singlestage import SingleStage


class Enhancer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        embed_head: int,
        num_resolution: int,
        num_level: int,
        dropout_ratio: float,
        offset: float,
        curoff: float,
    ) -> None:
        super().__init__()
        self.num_level = num_level

        stages = [
            SingleStage(
                embed_dim=embed_dim,
                embed_head=embed_head,
                num_resolution=num_resolution,
                dropout_ratio=dropout_ratio,
                offset=offset,
                curoff=curoff,
            )
            for _ in range(num_level)
        ]
        self.net = nn.Sequential(*stages)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
