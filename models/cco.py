"""
CCO: Class-conditional Orthogonal (Complementary) Constraint.

Imposed between the deep features of the two branches. Unlike a plain
orthogonality loss that pushes every pair of features apart, CCO is
class-conditional:

  * Real samples (y = 0): minimize the squared Euclidean distance between the
    two branch features, encouraging consistency (authentic images have stable
    feature patterns).
  * Fake samples (y = 1): minimize the squared Frobenius norm of the
    cross-inner-product matrix, encouraging low-correlation / complementary
    representations so that the two branches capture distinct synthetic traces.

This avoids the conflict between a conventional orthogonality constraint and
the consistency objective for real samples.

Reference: Section 3.4 of the paper.
"""
import torch
import torch.nn.functional as F


def cco_loss(feat_upper, feat_lower, targets, eps=1e-8):
    """
    Args:
        feat_upper: [B, D] deep features from the upper branch.
        feat_lower: [B, D] deep features from the lower branch.
        targets:    [B] integer labels, 0 = real, 1 = fake.
        eps:        small constant for numerical stability.
    Returns:
        Scalar CCO loss.
    """
    real_mask = (targets == 0)
    fake_mask = (targets == 1)
    loss = torch.zeros((), device=feat_upper.device)

    # Real samples: encourage feature consistency via squared Euclidean distance.
    if real_mask.any():
        fu = F.normalize(feat_upper[real_mask], dim=1, eps=eps)
        fl = F.normalize(feat_lower[real_mask], dim=1, eps=eps)
        dist_sq = (fu - fl).pow(2).sum(dim=1).mean()
        loss = loss + dist_sq

    # Fake samples: encourage complementarity via the squared Frobenius norm of
    # the cross-inner-product matrix (fu^T @ fl).
    if fake_mask.any():
        fu = F.normalize(feat_upper[fake_mask], dim=1, eps=eps)
        fl = F.normalize(feat_lower[fake_mask], dim=1, eps=eps)
        cross = fu.t() @ fl  # [D, D]
        loss = loss + cross.pow(2).sum() / fu.shape[0]

    return loss
