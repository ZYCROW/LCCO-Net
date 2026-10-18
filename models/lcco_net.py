"""
LCCO-Net: Layer-twice Cross-scale Complementary Optimization Network.

A difference-guided and frequency-adaptive dual-layer (dual-branch) framework
for AI-generated image detection. The overall pipeline is:

    Input X
      │
      ▼
    LHFE  ──►  enhanced X'  and  per-sample high-frequency energy E
      │
      ├──► Branch Upper ── Process 1 (shallow) ──► f_u,s
      │                    └── Process 2 (deep)  ──► f_u,d
      │
      └──► Branch Lower ── Process 1 (shallow) ──► f_l,s
                           └── Process 2 (deep)  ──► f_l,d

    Difference-guided CBAM refines both shallow and deep feature pairs using
    attention maps derived from |f_u - f_l|.

    FAD losses (shallow + deep) and the CCO constraint form the CSDD loss;
    the refined deep features are compared (difference) and classified.

Reference: Sections 3.1-3.7 of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lhfe import LocalHighFreqEnhance
from models.cbam import CBAM
from models.resnet import Bottleneck
from models.convnext import convnext_tiny


class ResNetBranch(nn.Module):
    """A single ResNet branch returning shallow (layer1) and deep (layer2) maps.

    Adapted from the lightweight AIDE backbone (Li et al., KDD 2025):
    3x3 stride-2 stem + maxpool, then layer1 (256 ch, shallow / Process 1)
    and layer2 (512 ch, deep / Process 2).
    """

    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        shallow = self.layer1(x)   # Process 1 / shallow
        deep = self.layer2(shallow)  # Process 2 / deep
        return shallow, deep


class ConvNeXtBranch(nn.Module):
    """A single ConvNeXt-tiny branch (self-contained implementation in models/convnext.py).

    Shallow = stage 0 output (96 channels), deep = stage 2 output (384 channels).
    """

    def __init__(self):
        super().__init__()
        self.backbone = convnext_tiny(features_only=True, out_indices=(0, 2))

    def forward(self, x):
        feats = self.backbone(x)
        shallow, deep = feats[0], feats[1]
        return shallow, deep


class TwoBranchLCCONet(nn.Module):
    """LCCO-Net dual-branch model.

    Args:
        backbone: 'resnet' or 'convnext'.
        num_classes: number of classification classes (2 for real/fake).
        alpha: LHFE high-frequency amplification coefficient (default 1.0).
        base_margin: initial base discrepancy margin for fake samples.
        margin_coeff: coefficient of the normalized high-frequency energy in
            the adaptive margin (default 0.30, as in the paper).
        ema_beta: momentum for the EMA of the natural real-sample discrepancy.
    """

    def __init__(self, backbone='resnet', num_classes=2, alpha=1.0,
                 base_margin=0.5, margin_coeff=0.30, ema_beta=0.99):
        super().__init__()
        self.backbone_name = backbone

        # Shared LHFE front-end (Section 3.1).
        self.lhfe = LocalHighFreqEnhance(in_channels=3, alpha=alpha)

        # Two parallel branches with independent parameters (Section 3.2).
        if backbone == 'resnet':
            self.branch_upper = ResNetBranch()
            self.branch_lower = ResNetBranch()
            shallow_ch, deep_ch = 256, 512
        elif backbone == 'convnext':
            self.branch_upper = ConvNeXtBranch()
            self.branch_lower = ConvNeXtBranch()
            shallow_ch, deep_ch = 96, 384
        else:
            raise ValueError("backbone must be 'resnet' or 'convnext', got %s" % backbone)

        # Difference-guided CBAM at shallow and deep levels (Section 3.3).
        # Weights are shared between the two branches and applied to each separately.
        self.cbam_shallow = CBAM(in_planes=shallow_ch)
        self.cbam_deep = CBAM(in_planes=deep_ch)

        # Classifier on the compared (difference of) refined deep features.
        self.classifier = nn.Linear(deep_ch, num_classes)

        # Learnable scale-balancing coefficients (gamma1, gamma2) for the
        # shallow/deep FAD losses; softmax-normalized so gamma1 + gamma2 = 1.
        self.scale_weights = nn.Parameter(torch.zeros(2))

        # Adaptive margin (Section 3.5): M = real_dist_ema + margin_coeff * E_norm.
        self.base_margin = base_margin
        self.margin_coeff = margin_coeff
        self.ema_beta = ema_beta
        self.register_buffer('real_dist_ema', torch.tensor(base_margin))

    # ------------------------------------------------------------------ #
    # Scale weights and adaptive margin
    # ------------------------------------------------------------------ #
    def get_scale_weights(self):
        """Return (w_deep, w_shallow), softmax-normalized so they sum to 1."""
        w = F.softmax(self.scale_weights, dim=0)
        return w[0], w[1]

    @torch.no_grad()
    def update_real_dist_ema(self, real_dist_mean):
        """Update the EMA of the natural cross-scale discrepancy of real samples.

        This EMA serves as the baseline of the adaptive margin.
        """
        if self.training:
            self.real_dist_ema = (
                self.ema_beta * self.real_dist_ema
                + (1 - self.ema_beta) * real_dist_mean.detach()
            )

    def get_adaptive_margins(self, high_freq_energy):
        """Sample-adaptive discrepancy margin (Eq. 5).

        M(x) = real_dist_ema + margin_coeff * normalized_E(x),
        where normalized_E is the min-max normalized high-frequency energy
        within the current batch. Samples with stronger high-frequency
        anomalies receive a larger required discrepancy.
        """
        if high_freq_energy.numel() > 1:
            e_min = high_freq_energy.min()
            e_max = high_freq_energy.max()
            e_norm = (high_freq_energy - e_min) / (e_max - e_min + 1e-8)
        else:
            e_norm = torch.zeros_like(high_freq_energy)
        return self.real_dist_ema + self.margin_coeff * e_norm

    # ------------------------------------------------------------------ #
    # Difference-guided CBAM refinement
    # ------------------------------------------------------------------ #
    @staticmethod
    def _discrepancy_guided_refine(feat_a, feat_b, cbam):
        """Refine a pair of branch features with difference-guided CBAM.

        D = |f_a - f_b|  ->  attention = CBAM(D)  ->  f_a * attention, f_b * attention.
        """
        diff_map = (feat_a - feat_b).abs()
        attn = cbam(diff_map)
        return feat_a * attn, feat_b * attn

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x):
        # LHFE: enhanced image and per-sample high-frequency energy.
        x_enh, hf_energy = self.lhfe(x)

        # Two branches: shallow (Process 1) and deep (Process 2).
        shallow_u, deep_u = self.branch_upper(x_enh)
        shallow_l, deep_l = self.branch_lower(x_enh)

        # Difference-guided CBAM refinement at both levels.
        shallow_u, shallow_l = self._discrepancy_guided_refine(
            shallow_u, shallow_l, self.cbam_shallow)
        deep_u, deep_l = self._discrepancy_guided_refine(
            deep_u, deep_l, self.cbam_deep)

        # Global average pool to obtain feature vectors for losses / classification.
        shallow_u_vec = F.adaptive_avg_pool2d(shallow_u, 1).flatten(1)
        shallow_l_vec = F.adaptive_avg_pool2d(shallow_l, 1).flatten(1)
        deep_u_vec = F.adaptive_avg_pool2d(deep_u, 1).flatten(1)
        deep_l_vec = F.adaptive_avg_pool2d(deep_l, 1).flatten(1)

        # Compare refined deep features (difference as the discriminative cue).
        diff_deep = deep_u_vec - deep_l_vec
        logits = self.classifier(diff_deep)

        # Sample-adaptive margins for the FAD loss.
        margins = self.get_adaptive_margins(hf_energy)

        return (logits, deep_u_vec, deep_l_vec, shallow_u_vec, shallow_l_vec,
                hf_energy, margins)


# ---------------------------------------------------------------------- #
# Factory functions
# ---------------------------------------------------------------------- #
def lcco_net_resnet(**kwargs):
    """Build LCCO-Net with the ResNet (AIDE-style) backbone."""
    return TwoBranchLCCONet(backbone='resnet', **kwargs)


def lcco_net_convnext(**kwargs):
    """Build LCCO-Net with the ConvNeXt-tiny backbone."""
    return TwoBranchLCCONet(backbone='convnext', **kwargs)
