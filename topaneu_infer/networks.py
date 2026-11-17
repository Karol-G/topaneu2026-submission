"""The two segmentation networks, built from the public dynamic-network-architectures package.

    Primus                  transformer encoder + lightweight decoder (CT model)
    ResidualEncoderUNet     nnU-Net ResEnc-L (MR model)

Both are plain modules: every exported state dict loads with `strict=True`.
"""
from typing import Sequence

import torch
from torch import nn


def _torch_op(name):
    if name is None:
        return None
    op = getattr(nn, name, None)
    if op is None:
        raise ValueError(f"unknown torch.nn op {name!r}")
    return op


def build_network(architecture: dict, num_input_channels: int, num_classes: int,
                  patch_size: Sequence[int]) -> nn.Module:
    name = architecture["name"]
    kw = dict(architecture["kwargs"])

    if name == "Primus":
        from dynamic_network_architectures.architectures.primus import Primus
        return Primus(
            num_input_channels,
            kw["embed_dim"],
            tuple(kw["patch_embed_size"]),
            num_classes,
            kw["eva_depth"],
            kw["eva_numheads"],
            list(patch_size),
            drop_path_rate=kw["drop_path_rate"],
            scale_attn_inner=kw["scale_attn_inner"],
            init_values=kw["init_values"],
        )

    if name == "ResidualEncoderUNet":
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        for k in ("conv_op", "norm_op", "dropout_op", "nonlin"):
            kw[k] = _torch_op(kw.get(k))
        return ResidualEncoderUNet(input_channels=num_input_channels, num_classes=num_classes,
                                   deep_supervision=False, **kw)

    raise ValueError(f"unknown architecture {name!r}")
