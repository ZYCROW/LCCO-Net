"""
ConvNeXt-tiny backbone for LCCO-Net (ours_C variant).

ConvNeXt (Liu et al., CVPR 2022) is a pure convolutional network whose building
block is a residual block:

    x -> depthwise 7x7 conv -> LayerNorm -> pointwise 1x1 conv (4x expansion)
      -> GELU -> pointwise 1x1 conv -> layer scale -> drop path -> + x (residual)

This file provides a self-contained implementation (no dependency on timm) of
ConvNeXt-tiny (depths=[3,3,9,3], dims=[96,192,384,768]). When used inside
LCCO-Net it is instantiated with features_only=True and out_indices=(0, 2), so
that it returns:
  - shallow feature = stage 0 output (96 channels, Process 1)
  - deep feature    = stage 2 output (384 channels, Process 2)

Reference: Section 3.2 / Table 1-2 of the paper (ours_C = ConvNeXt backbone).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for NCHW tensors (channels-last is applied then transposed back)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C] -> norm -> [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) per sample, used in residual blocks."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """ConvNeXt residual block.

    Architecture:
        depthwise 7x7 conv -> LayerNorm -> pointwise 1x1 conv (4x expansion)
        -> GELU -> pointwise 1x1 conv -> layer scale -> drop path -> + residual
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise convolution (groups = channels).
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        # Pointwise convolutions with 4x intermediate expansion.
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Layer scale: per-channel learnable scaling initialized to a small value.
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.dwconv(x)
        out = self.norm(out)
        # [B, C, H, W] -> [B, H, W, C] for pointwise (Linear) operations.
        out = out.permute(0, 2, 3, 1)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = self.gamma * out
        out = out.permute(0, 3, 1, 2)  # back to [B, C, H, W]
        out = identity + self.drop_path(out)  # residual connection
        return out


class ConvNeXt(nn.Module):
    """ConvNeXt backbone.

    Args:
        in_chans: number of input image channels (default 3).
        depths: number of blocks in each of the 4 stages.
        dims: channel width of each stage.
        drop_path_rate: stochastic depth rate (linearly increased across stages).
        layer_scale_init_value: initial value of layer scale (0 disables it).
        features_only: if True, forward returns a list of stage outputs selected
            by out_indices instead of the classification logits.
        out_indices: indices of stages to return when features_only=True.
    """

    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 drop_path_rate=0.0, layer_scale_init_value=1e-6,
                 features_only=False, out_indices=(0, 2)):
        super().__init__()
        self.features_only = features_only
        self.out_indices = out_indices

        # Stem: 4x4 conv stride 4 + LayerNorm.
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # Downsampling layers between stages: 2x2 conv stride 2 + LayerNorm.
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Four stages, each built from ConvNeXt residual blocks.
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i],
                                drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final norm (only used by the classification head, not features_only).
        if not features_only:
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
            self.head = nn.Linear(dims[-1], 1000)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.features_only and i in self.out_indices:
                outs.append(x)
        if self.features_only:
            return outs
        # Classification path (unused inside LCCO-Net).
        x = self.norm(x.mean([-2, -1]))  # global average pool + norm
        x = self.head(x)
        return x


def convnext_tiny(pretrained=False, **kwargs):
    """Build a ConvNeXt-tiny backbone (depths=[3,3,9,3], dims=[96,192,384,768])."""
    model = ConvNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    if pretrained:
        # Official ImageNet-1k weights can be loaded here if available.
        raise NotImplementedError(
            "Pretrained weight loading is not bundled; load via torch.load manually."
        )
    return model
