"""
LHFE: Local High-Frequency Enhancement.

Placed at the input end of LCCO-Net, before the two parallel branches and
shared by both branches. It decomposes the input into a low-frequency component
with a learnable blur kernel, obtains the high-frequency residual
    H = X - LowPass(X),                                  (Eq. 2)
and enhances the input as
    X' = X + alpha * H.                                  (Eq. 3)
The global high-frequency energy
    E = mean(|H|)                                        (Eq. 4)
is returned together with the enhanced feature, and is used to build the
sample-adaptive discrepancy margin in the FAD loss (Eq. 5).

Reference: Section 3.1 of the paper.
"""
import torch
import torch.nn as nn


class LocalHighFreqEnhance(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3, alpha=1.0):
        super().__init__()
        # Learnable depthwise blur kernel (parameterized low-pass filter).
        self.blur_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        # Initialize as an average filter.
        nn.init.constant_(self.blur_conv.weight, 1.0 / (kernel_size ** 2))

        # High-frequency amplification coefficient (alpha in Eq. 3).
        # The hyper-parameter study in Section 4.4 shows alpha=1 is optimal.
        self.alpha = alpha

    def forward(self, x):
        # Low-frequency component via the learnable blur.
        low_freq = self.blur_conv(x)
        # High-frequency residual: H = X - LowPass(X).
        high_freq = x - low_freq
        # Enhanced representation: X' = X + alpha * H.
        enhanced = x + self.alpha * high_freq
        # Per-sample global high-frequency energy: E = mean(|H|).
        high_freq_energy = high_freq.abs().mean(dim=[1, 2, 3])
        return enhanced, high_freq_energy
