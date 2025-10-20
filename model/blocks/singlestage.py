import torch
import torch.nn as nn
from torch import Tensor

from model.blocks.attention import (
    CrossAttentionBlock,
    PatchEmbedding,
    SelfAttentionBlock,
    UnPatchEmbedding,
)
from model.blocks.homomorphic import HomomorphicSeparate


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
        dropout_ratio: float,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.separate = HomomorphicSeparate(cutoff=cutoff)

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.il_sttn = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )
        self.re_sttn = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

        self.cttn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

        self.unpatch_embed = UnPatchEmbedding(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x1, x2, x3 = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        x1_il, x1_re = self.separate(x1)
        x2_il, x2_re = self.separate(x2)
        x3_il, x3_re = self.separate(x3)

        il_concat = torch.concat(tensors=[x1_il, x2_il, x3_il], dim=1)
        re_concat = torch.concat(tensors=[x1_re, x2_re, x3_re], dim=1)

        il_patch = self.patch_embed(il_concat)
        re_patch = self.patch_embed(re_concat)

        il_fea = self.il_sttn(il_patch)
        re_fea = self.re_sttn(re_patch)
        out_fea = self.cttn(il_fea, re_fea)

        out = self.unpatch_embed(out_fea)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
        num_resolution: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        dim_level = embed_dim
        self.down: nn.ModuleList = nn.ModuleList()
        for level in range(num_resolution):
            self.down.append(
                module=nn.ModuleList(
                    modules=[
                        CrossAttentionBlock(
                            embed_dim=dim_level,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout_ratio=dropout_ratio,
                        ),
                        nn.Conv2d(
                            in_channels=dim_level,
                            out_channels=dim_level * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.Conv2d(
                            in_channels=dim_level,
                            out_channels=dim_level * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ]
                )
            )
            dim_level *= 2

        self.mid = CrossAttentionBlock(
            embed_dim=dim_level,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

        self.up: nn.ModuleList = nn.ModuleList()
        for level in range(num_resolution):
            self.up.append(
                module=nn.ModuleList(
                    modules=[
                        nn.ConvTranspose2d(
                            in_channels=dim_level,
                            out_channels=dim_level // 2,
                            kernel_size=2,
                            stride=2,
                            bias=False,
                        ),
                        nn.Conv2d(
                            in_channels=dim_level,
                            out_channels=dim_level // 2,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                        CrossAttentionBlock(
                            embed_dim=dim_level,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout_ratio=dropout_ratio,
                        ),
                    ]
                )
            )
            dim_level //= 2

        self.unpatch_embed = UnPatchEmbedding(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_channels=out_channels,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.patch_embed(x)
        c = self.patch_embed(c)

        x_lst = []
        c_lst = []
        for attn, x_down, c_down in self.down:
            x = attn(x, c)
            x_lst.append(x)
            c_lst.append(c)
            x = x_down(x)
            c = c_down(c)

        x = self.mid(x, c)

        for up, fusion, attn in self.up:
            x = up(x)
            x = fusion(torch.cat(tensors=[x, x_lst.pop()], dim=1))
            x = attn(x, c_lst.pop())

        return self.unpatch_embed(x)


class SingleStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
        num_resolution: int,
        dropout_ratio: float,
        cutoff: float,
    ) -> None:
        super().__init__()

        self.separater = Encoder(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            cutoff=cutoff,
            dropout_ratio=dropout_ratio,
        )
        self.denoiser = Decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            num_resolution=num_resolution,
            dropout_ratio=dropout_ratio,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.separater(x)
        x = self.denoiser(x)
        return x
