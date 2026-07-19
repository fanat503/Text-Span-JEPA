# Copyright (c) Text-Span JEPA Authors
# Collapse prevention: variance, covariance, centering, diagnostics
# Following VICReg (Bardes et al., ICLR 2022), I-JEPA (Assran et al., CVPR 2023),
# data2vec 2.0 (Baevski et al., 2023), C-JEPA (NeurIPS 2024),
# Barlow Twins (Zbontar et al., ICML 2021), DINO (Caron et al., ICCV 2021)

import math

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
        N = representations.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=representations.device, requires_grad=True)
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
        denom = max(N - 1, 1)
        cov = (representations.T @ representations) / denom
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

    Metrics from I-JEPA, NextLat, VICReg, Barlow Twins, DINO:
    - Effective rank (Shannon entropy of singular values) — NextLat
    - Participation ratio (effective dimensionality) — Roy & Vetterli
    - Condition number, numerical rank, coherence — NextLat
    - Collapsed dimension ratio — I-JEPA / lang-jepa
    - Cross-correlation redundancy — Barlow Twins
    - CKA (centered kernel alignment) — Kornblith et al.
    - Cosine similarity statistics — DINO / BYOL
    - Attention entropy — DINO
    """

    def __init__(self, collapse_threshold=1e-2):
        super().__init__()
        self.collapse_threshold = collapse_threshold

    @torch.no_grad()
    def compute(self, online_h, target_h):
        metrics = {}
        metrics['online_std'] = online_h.std(dim=(0, 1)).mean().item()
        metrics['target_std'] = target_h.std(dim=(0, 1)).mean().item()

        online_flat = online_h.reshape(-1, online_h.size(-1))
        target_flat = target_h.reshape(-1, target_h.size(-1))

        # --- Pairwise cosine (DINO / BYOL) ---
        metrics['online_target_cosine'] = F.cosine_similarity(
            online_flat, target_flat, dim=-1
        ).mean().item()
        metrics['online_target_mse'] = F.mse_loss(online_h, target_h).item()

        if online_flat.size(0) > 1:
            metrics['online_pair_cosine'] = F.cosine_similarity(
                online_flat[:-1], online_flat[1:], dim=-1
            ).mean().item()

        # --- SVD-based rank metrics (NextLat) ---
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

        # --- Rank utilization (NextLat) ---
        D = online_h.size(-1)
        N = online_flat.size(0)
        max_rank = min(N, D)
        metrics['rank_utilization_online'] = metrics['numerical_rank_online'] / max_rank if max_rank > 0 else 0.0
        metrics['rank_utilization_target'] = metrics['numerical_rank_target'] / max_rank if max_rank > 0 else 0.0

        # --- Collapsed dimension ratio (I-JEPA / lang-jepa) ---
        metrics['collapsed_dim_ratio_online'] = self._collapsed_dim_ratio(online_h)
        metrics['collapsed_dim_ratio_target'] = self._collapsed_dim_ratio(target_h)

        # --- Cross-correlation redundancy (Barlow Twins) ---
        metrics['cross_corr_redundancy'] = self._cross_corr_redundancy(online_flat, target_flat)

        # --- CKA (Kornblith et al., 2019) ---
        metrics['cka_linear'] = self._cka_linear(online_flat, target_flat)

        return metrics

    @staticmethod
    def _effective_rank(x):
        """Shannon entropy of normalized singular values (NextLat / I-JEPA).

        NextLat model_base pattern: catches SVD failures, returns 0.0.
        Also handles all-zero input (sum=0 → NaN) by returning 0.0.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            total = S.sum()
            if total == 0 or not torch.isfinite(total):
                return 0.0
            S_norm = S / total
            S_norm = torch.clamp(S_norm, min=1e-12)
            entropy = -torch.sum(S_norm * torch.log(S_norm))
            val = entropy.exp().item()
            if not math.isfinite(val):
                return 0.0
            return val
        except Exception:
            return 0.0

    @staticmethod
    def _participation_ratio(x):
        """(sum S)^2 / sum(S^2). PR=1 means 1D collapse.

        NextLat pattern: exception handling returns 0.0, NaN-guard for zero input.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            sum_sq = (S ** 2).sum()
            if sum_sq == 0 or not torch.isfinite(sum_sq):
                return 0.0
            val = (S.sum() ** 2 / sum_sq).item()
            if not math.isfinite(val):
                return 0.0
            return val
        except Exception:
            return 0.0

    @staticmethod
    def _condition_number(x):
        """Condition number S[0]/S[-1]. NextLat pattern: inf for degenerate input."""
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            if S[-1] == 0 or not torch.isfinite(S[-1]):
                return float('inf')
            if S[0] == 0 or not torch.isfinite(S[0]):
                return float('inf')
            return (S[0] / S[-1]).item()
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
        """Max absolute off-diagonal element of covariance. NextLat pattern."""
        flat = x.reshape(-1, x.size(-1))
        try:
            centered = flat - flat.mean(dim=0)
            N = max(flat.size(0) - 1, 1)
            cov = (centered.T @ centered) / N
            if cov.abs().max().item() == 0:
                return 0.0
            diag = torch.diag(torch.diag(cov))
            off_diag = cov - diag
            D = x.size(-1)
            if D <= 1:
                return 0.0
            val = off_diag.abs().max().item()
            if not math.isfinite(val):
                return 0.0
            return val
        except Exception:
            return 0.0

    @staticmethod
    def _collapsed_dim_ratio(x, threshold=1e-2):
        """Proportion of dimensions with near-zero variance (I-JEPA / lang-jepa).

        Healthy representations should have this near 0.0.
        Collapse shows up as this approaching 1.0.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            N = flat.size(0)
            if N <= 1:
                return 1.0
            var = flat.var(dim=0)
            collapsed = (var < threshold).float().mean().item()
            return collapsed
        except Exception:
            return 1.0

    @staticmethod
    def _cross_corr_redundancy(online_flat, target_flat):
        """Barlow Twins redundancy: mean absolute off-diagonal of cross-correlation.

        Barlow Twins (Zbontar et al., ICML 2021) uses the cross-correlation
        matrix between two representations. The on-diagonal should be 1
        (invariance) and off-diagonal should be 0 (redundancy reduction).
        This metric reports the mean |off-diagonal| — lower is better.
        """
        try:
            N, D = online_flat.shape
            if N <= 1:
                return 1.0
            # Normalize each dimension
            o = (online_flat - online_flat.mean(dim=0)) / (online_flat.std(dim=0) + 1e-6)
            t = (target_flat - target_flat.mean(dim=0)) / (target_flat.std(dim=0) + 1e-6)
            c = (o.T @ t) / N
            # Mean absolute off-diagonal
            diag_mask = torch.eye(D, device=c.device)
            off_diag = c * (1 - diag_mask)
            redundancy = off_diag.abs().sum() / (D * (D - 1) + 1e-8)
            val = redundancy.item()
            if not math.isfinite(val):
                return 1.0
            return val
        except Exception:
            return 1.0

    @staticmethod
    def _cka_linear(x, y):
        """Linear CKA (centered kernel alignment) between two representations.

        Kornblith et al., "Similarity of Neural Network Representations
        Revisited", ICML 2019. Measures similarity of representation
        geometry independent of orthogonal transformations.
        Returns value in [0, 1]; 1 = identical geometry.
        """
        try:
            # Center
            x = x - x.mean(dim=0, keepdim=True)
            y = y - y.mean(dim=0, keepdim=True)
            # HSIC
            hsic_xy = CollapseDiagnostics._hsic(x, y)
            hsic_xx = CollapseDiagnostics._hsic(x, x)
            hsic_yy = CollapseDiagnostics._hsic(y, y)
            denom = (hsic_xx * hsic_yy).sqrt() + 1e-10
            val = (hsic_xy / denom).item()
            if not math.isfinite(val):
                return 0.0
            return max(min(val, 1.0), 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _hsic(x, y):
        """Hilbert-Schmidt Independence Criterion (unbiased)."""
        N = x.size(0)
        if N <= 3:
            return torch.tensor(0.0, device=x.device)
        K = x @ x.T
        L = y @ y.T
        # Center the kernel matrices
        H = torch.eye(N, device=x.device) - 1.0 / N
        KH = K @ H
        LH = L @ H
        return torch.trace(KH @ LH) / ((N - 1) ** 2)
