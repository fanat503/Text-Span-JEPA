# Copyright (c) Text-Span JEPA Authors
# Collapse prevention: variance, covariance, centering, diagnostics
# Following VICReg (Bardes et al., ICLR 2022), I-JEPA (Assran et al., CVPR 2023),
# data2vec 2.0 (Baevski et al., 2023), C-JEPA (NeurIPS 2024)

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceRegularization(nn.Module):
    """VICReg variance term: prevents dimension collapse.

    Each dimension with variance below margin incurs a penalty.
    From VICReg (Bardes et al., ICLR 2022).
    """

    def __init__(self, margin=1.0, eps=1e-4):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, representations):
        if representations.dim() == 3:
            B, T, D = representations.shape
            representations = representations.reshape(B * T, D)
        var = representations.var(dim=0)
        loss = F.relu(self.margin - (var + self.eps).sqrt()).mean()
        return loss


class CovarianceRegularization(nn.Module):
    """VICReg covariance term: decorrelates representation dimensions.

    Off-diagonal elements of covariance matrix penalized toward zero.
    From VICReg (Bardes et al., ICLR 2022).
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, representations):
        if representations.dim() == 3:
            B, T, D = representations.shape
            representations = representations.reshape(B * T, D)
        N, D = representations.shape
        representations = representations - representations.mean(dim=0)
        cov = (representations.T @ representations) / (N - 1)
        diag = torch.diag(torch.diag(cov))
        off_diag = cov - diag
        loss = off_diag.pow(2).sum() / D
        return loss


class TargetCentering(nn.Module):
    """Moving average centering of target representations.

    From data2vec 2.0: subtract running mean from target embeddings
    to prevent collapse to a constant offset.
    """

    def __init__(self, dim=768, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('center', torch.zeros(1, 1, dim))

    @torch.no_grad()
    def update_center(self, target_representations):
        batch_center = target_representations.mean(dim=(0, 1), keepdim=True)
        self.center = self.momentum * self.center + (1 - self.momentum) * batch_center

    def forward(self, target_representations):
        self.update_center(target_representations)
        return target_representations - self.center


class CollapseDiagnostics(nn.Module):
    """Diagnostic metrics for monitoring representational collapse.

    Metrics from I-JEPA, NextLat, VICReg, C-JEPA:
    - Effective rank (Shannon entropy of singular values)
    - Participation ratio (effective dimensionality)
    - Condition number, numerical rank, coherence
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def compute(self, online_h, target_h):
        metrics = {}
        metrics['online_std'] = online_h.std(dim=(0, 1)).mean().item()
        metrics['target_std'] = target_h.std(dim=(0, 1)).mean().item()

        online_flat = online_h.reshape(-1, online_h.size(-1))
        target_flat = target_h.reshape(-1, target_h.size(-1))
        metrics['online_target_cosine'] = F.cosine_similarity(
            online_flat, target_flat, dim=-1
        ).mean().item()
        metrics['online_target_mse'] = F.mse_loss(online_h, target_h).item()

        metrics['effective_rank_online'] = max(self._effective_rank(online_h), 0.0)
        metrics['effective_rank_target'] = max(self._effective_rank(target_h), 0.0)
        metrics['participation_ratio_online'] = max(self._participation_ratio(online_h), 0.0)
        metrics['participation_ratio_target'] = max(self._participation_ratio(target_h), 0.0)
        metrics['condition_number_online'] = self._condition_number(online_h)
        metrics['condition_number_target'] = self._condition_number(target_h)
        metrics['numerical_rank_online'] = self._numerical_rank(online_h)
        metrics['numerical_rank_target'] = self._numerical_rank(target_h)
        metrics['coherence_online'] = self._coherence(online_h)
        metrics['coherence_target'] = self._coherence(target_h)

        if online_flat.size(0) > 1:
            metrics['online_pair_cosine'] = F.cosine_similarity(
                online_flat[:-1], online_flat[1:], dim=-1
            ).mean().item()

        return metrics

    @staticmethod
    def _effective_rank(x):
        """Shannon entropy of normalized singular values (NextLat / I-JEPA)."""
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            S_norm = S / S.sum()
            S_norm = torch.clamp(S_norm, min=1e-12)
            entropy = -torch.sum(S_norm * torch.log(S_norm))
            return entropy.exp().item()
        except Exception:
            return 0.0

    @staticmethod
    def _participation_ratio(x):
        """(sum S)^2 / sum(S^2). PR=1 means 1D collapse."""
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            return (S.sum() ** 2 / (S ** 2).sum()).item()
        except Exception:
            return 0.0

    @staticmethod
    def _condition_number(x):
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            if S[0] > 0 and S[-1] > 0:
                return (S[0] / S[-1]).item()
            return float('inf')
        except Exception:
            return float('inf')

    @staticmethod
    def _numerical_rank(x):
        flat = x.reshape(-1, x.size(-1))
        try:
            return torch.linalg.matrix_rank(flat, atol=1e-3, rtol=1e-3).item()
        except Exception:
            return 0

    @staticmethod
    def _coherence(x):
        flat = x.reshape(-1, x.size(-1))
        try:
            centered = flat - flat.mean(dim=0)
            cov = (centered.T @ centered) / max(flat.size(0) - 1, 1)
            diag = torch.diag(torch.diag(cov))
            off_diag = cov - diag
            D = x.size(-1)
            return off_diag.abs().max().item() if D > 1 else 0.0
        except Exception:
            return 0.0
