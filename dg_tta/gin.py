# MIT License

# Copyright (c) 2022 Cheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# GIN Code based on:
# Ouyang, C., Chen, C., Li, S., Li, Z., Qin, C., Bai, W., & Rueckert, D. (2022). Causality-inspired single-source domain generalization for medical image segmentation. IEEE Transactions on Medical Imaging, 42(4), 1095-1106.
# DOI: 10.1109/TMI.2022.3224067
# https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization
# https://ieeexplore.ieee.org/document/9961940

import torch
import torch.nn as nn
import torch.nn.functional as F

from dg_tta.utils import get_internal_augmentation_enabled


class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(
        self,
        out_channel=32,
        in_channel=3,
        scale_pool=[1, 3],
        layer_id=0,
        use_act=True,
        requires_grad=False,
    ):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        assert requires_grad == False

    def forward(self, x_in, requires_grad=False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
        k = self.scale_pool[idx_k[0]]

        if x_in.dim() == 5:
            mode = "3d"
        elif x_in.dim() == 4:
            mode = "2d"
        else:
            raise ValueError()

        if mode == "2d":
            nb, nc, nx, ny = x_in.shape
            ker = torch.randn(
                [self.out_channel * nb, self.in_channel, k, k],
                requires_grad=self.requires_grad,
            ).to(x_in.device)
            shift = (
                torch.randn(
                    [self.out_channel * nb, 1, 1], requires_grad=self.requires_grad
                ).to(x_in.device)
                * 1.0
            )
            x_in = x_in.reshape(1, nb * nc, nx, ny)
            x_conv = F.conv2d(
                x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb
            )

        elif mode == "3d":
            nb, nc, nx, ny, nz = x_in.shape
            ker = torch.randn(
                [self.out_channel * nb, self.in_channel, k, k, k],
                requires_grad=self.requires_grad,
            ).to(x_in.device)
            shift = (
                torch.randn(
                    [self.out_channel * nb, 1, 1, 1], requires_grad=self.requires_grad
                ).to(x_in.device)
                * 1.0
            )
            x_in = x_in.reshape(1, nb * nc, nx, ny, nz)
            x_conv = F.conv3d(
                x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb
            )
        else:
            raise ValueError()

        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        if mode == "2d":
            x_conv = x_conv.reshape(nb, self.out_channel, nx, ny)
        elif mode == "3d":
            x_conv = x_conv.reshape(nb, self.out_channel, nx, ny, nz)
        else:
            raise ValueError()

        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, cfg):
        """
        GIN
        """
        super(GINGroupConv, self).__init__()
        self.scale_pool = [1, 3]  # don't make it tool large as we have multiple layers
        self.n_layer = cfg["N_LAYER"]
        self.layers = []
        self.out_norm = "frob"
        self.out_channel = cfg["IN_CHANNELS"]
        in_channel = cfg["IN_CHANNELS"]
        interm_channel = cfg["INTERM_CHANNELS"]

        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=interm_channel,
                in_channel=in_channel,
                scale_pool=self.scale_pool,
                layer_id=0,
            )
        )
        for ii in range(self.n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock(
                    out_channel=interm_channel,
                    in_channel=interm_channel,
                    scale_pool=self.scale_pool,
                    layer_id=ii + 1,
                )
            )
        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=self.out_channel,
                in_channel=interm_channel,
                scale_pool=self.scale_pool,
                layer_id=self.n_layer - 1,
                use_act=False,
            )
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        if x_in.dim() == 5:
            mode = "3d"
        elif x_in.dim() == 4:
            mode = "2d"
        else:
            raise ValueError()

        nb, nc = x_in.shape[:2]

        if mode == "2d":
            alphas = torch.rand(nb, device=x_in.device)[
                :, None, None, None
            ]  # nb, 1, 1, 1
            alphas = alphas.repeat(1, nc, 1, 1)  # nb, nc, 1, 1
        elif mode == "3d":
            alphas = torch.rand(nb, device=x_in.device)[
                :, None, None, None, None
            ]  # nb, 1, 1, 1, 1
            alphas = alphas.repeat(1, nc, 1, 1, 1)  # nb, nc, 1, 1, 1
        else:
            raise ValueError()

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == "frob":
            _in_frob = torch.norm(
                x_in.reshape(nb, nc, -1), dim=(-1, -2), p="fro", keepdim=False
            )
            if mode == "2d":
                _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
                _self_frob = torch.norm(
                    mixed.reshape(nb, self.out_channel, -1),
                    dim=(-1, -2),
                    p="fro",
                    keepdim=False,
                )
                _self_frob = _self_frob[:, None, None, None].repeat(
                    1, self.out_channel, 1, 1
                )
            elif mode == "3d":
                _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
                _self_frob = torch.norm(
                    mixed.reshape(nb, self.out_channel, -1),
                    dim=(-1, -2),
                    p="fro",
                    keepdim=False,
                )
                _self_frob = _self_frob[:, None, None, None, None].repeat(
                    1, self.out_channel, 1, 1, 1
                )
            else:
                raise ValueError()

            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed


def gin_aug(input):
    cfg = dict(
        IN_CHANNELS=1,
        N_LAYER=4,
        INTERM_CHANNELS=2,
    )
    gin_group_conv = GINGroupConv(cfg)
    input = gin_group_conv(input)
    return input


def gin_hook(module, input):
    if get_internal_augmentation_enabled():
        return gin_aug(*input)
    return input
