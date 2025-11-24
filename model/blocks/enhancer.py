import torch
import torch.nn as nn
from torch import Tensor

from model.blocks.attention import CrossAttentionBlock, SelfAttentionBlock
from model.blocks.flatten import Flatten, Unflatten
from model.blocks.sampling import Downsampling, Upsampling
from model.blocks.separation import SeparationBlock


class Encoder(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        sigma: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.separate = SeparationBlock(
            kernel_size=kernel_size,
            sigma=sigma,
        )

        self.in_conv: nn.Conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
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

        self.out_conv: nn.Conv2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        b, c, h, w = x.shape
        x1, x2, x3 = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        x1_il, x1_re = self.separate(x1)
        x2_il, x2_re = self.separate(x2)
        x3_il, x3_re = self.separate(x3)

        il_concat = torch.concat(tensors=[x1_il, x2_il, x3_il], dim=1)
        re_concat = torch.concat(tensors=[x1_re, x2_re, x3_re], dim=1)

        il_conv = self.in_conv(il_concat)
        re_conv = self.in_conv(re_concat)

        il_flat = Flatten(x=il_conv)
        re_flat = Flatten(x=re_conv)

        il_attn = self.il_sttn(il_flat)
        re_attn = self.re_sttn(re_flat)
        out_cttn = self.cttn(il_attn, re_attn)

        out_unflat = Unflatten(x=out_cttn, h=h, w=w)

        out = self.out_conv(out_unflat)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_resolution: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()

        self.in_conv: nn.Conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        dim_level = embed_dim
        self.down: nn.ModuleList = nn.ModuleList(modules=[])
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
                        Downsampling(
                            in_channels=dim_level,
                            out_channels=dim_level * 2,
                        ),
                        Downsampling(
                            in_channels=dim_level,
                            out_channels=dim_level * 2,
                        ),
                    ]
                )
            )
            dim_level *= 2

        self.mid: nn.ModuleList = nn.ModuleList(modules=[])
        for _ in range(num_resolution // 2):
            self.mid.append(
                module=CrossAttentionBlock(
                    embed_dim=dim_level,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_ratio=dropout_ratio,
                )
            )

        self.up: nn.ModuleList = nn.ModuleList(modules=[])
        for level in range(num_resolution):
            self.up.append(
                module=nn.ModuleList(
                    modules=[
                        Upsampling(
                            in_channels=dim_level,
                            out_channels=dim_level // 2,
                        ),
                        nn.Conv2d(
                            in_channels=dim_level,
                            out_channels=dim_level // 2,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                        CrossAttentionBlock(
                            embed_dim=dim_level // 2,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout_ratio=dropout_ratio,
                        ),
                    ]
                )
            )
            dim_level //= 2

        self.out_conv: nn.Conv2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x_res = x
        c_res = c
        x = self.in_conv(x)
        c = self.in_conv(c)

        x_lst = []
        c_lst = []
        for cttn, x_down, c_down in self.down:
            _, _, h, w = x.shape
            x, c = Flatten(x=x), Flatten(x=c)
            x = cttn(x, c)
            x, c = Unflatten(x=x, h=h, w=w), Unflatten(x=c, h=h, w=w)

            x_lst.append(x)
            c_lst.append(c)

            x = x_down(x)
            c = c_down(c)

        for cttn in self.mid:
            _, _, h, w = x.shape
            x, c = Flatten(x=x), Flatten(x=c)
            x = cttn(x, c)
            x, c = Unflatten(x=x, h=h, w=w), Unflatten(x=c, h=h, w=w)

        for x_up, fusion, cttn in self.up:
            x = x_up(x)
            x = fusion(torch.cat(tensors=[x, x_lst.pop()], dim=1))
            c = c_lst.pop()

            _, _, h, w = x.shape
            x, c = Flatten(x=x), Flatten(x=c)
            x = cttn(x, c)
            x, c = Unflatten(x=x, h=h, w=w), Unflatten(x=c, h=h, w=w)

        out = self.out_conv(x) + x_res * c_res

        return out


class Enhancer(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        sigma: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_resolution: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            channels=channels,
            kernel_size=kernel_size,
            sigma=sigma,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

        self.decoder = Decoder(
            channels=channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_resolution=num_resolution,
            dropout_ratio=dropout_ratio,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        c = self.encoder(x)
        out = self.decoder(x, c)

        return out
