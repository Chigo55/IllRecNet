from torch import Tensor


def Flatten(x: Tensor) -> Tensor:
    return x.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)


def Unflatten(x: Tensor, h: int, w: int):
    return x.permute(0, 2, 1).unflatten(dim=2, sizes=(h, w))
