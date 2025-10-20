import math

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.flattener = nn.Flatten(
            start_dim=2,
            end_dim=3,
        )
        self.register_parameter(name="pos_embed", param=None)

    def _reset_positional_embedding(
        self,
        num_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        pos_embed = torch.zeros(
            size=(1, num_patches, self.embed_dim),
            device=device,
            dtype=dtype,
        )
        nn.init.trunc_normal_(tensor=pos_embed, std=0.02)
        self.pos_embed = nn.Parameter(data=pos_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patcher(x)
        x = self.flattener(x)
        x = x.permute(0, 2, 1)

        num_patches = x.shape[1]
        if self.pos_embed is None:
            self._reset_positional_embedding(
                num_patches=num_patches,
                device=x.device,
                dtype=x.dtype,
            )

        return x + self.pos_embed


class UnPatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int, out_channels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim

        self.unpatcher = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        num_patches = x.shape[1]
        h_patch = w_patch = int(math.sqrt(num_patches))
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], self.embed_dim, h_patch, w_patch)
        x = self.unpatcher(x)

        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int, dropout_ratio: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=embed_dim * mlp_ratio,
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(
                in_features=embed_dim * mlp_ratio,
                out_features=embed_dim * mlp_ratio,
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(
                in_features=embed_dim * mlp_ratio,
                out_features=embed_dim,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_ratio: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_ratio,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        return attn


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout_ratio: float,
    ):
        super().__init__()
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_ratio=dropout_ratio,
        )
        self.mlp = MultiLayerPerceptron(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x) + x
        x = self.mlp(x) + x
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_ratio: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_ratio,
            batch_first=True,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.layer_norm(x)
        attn, _ = self.attn(query=x, key=c, value=c, need_weights=False)
        return attn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout_ratio: float,
    ):
        super().__init__()
        self.attn = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_ratio=dropout_ratio,
        )
        self.mlp = MultiLayerPerceptron(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.attn(x, c) + x
        x = self.mlp(x) + x
        return x
