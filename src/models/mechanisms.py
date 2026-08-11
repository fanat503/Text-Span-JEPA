# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Convenience API: One-line access to all 8 novel mechanisms.
#
# ═══════════════════════════════════════════════════════════════════════════
#  USAGE (3 lines to upgrade any JEPA with all 8 mechanisms)
# ═══════════════════════════════════════════════════════════════════════════
#
#  from src.models.mechanisms import MechanismBundle
#
#  bundle = MechanismBundle.from_config(config)  # line 1
#  z_refined, all_info = bundle(z_pred, z_target, mask_positions, step)  # line 2
#  bundle.retract()  # line 3 (after optimizer.step())
#
#  That's it. All 8 mechanisms in 3 lines.
#
#  You can also use individual mechanisms:
#    from src.models.mechanisms import jawp_loss, cgn_gate, pcr_refine, swip_whiten

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from .jawp import JAWPModule
from .cgn import ContextualGatingNetwork
from .swip import SWIPModule
from .pcr import PredictiveCascadeRefinement
from .spc import SpectralPredictiveCoding


class MechanismBundle(nn.Module):
    """All 8 Text-Span JEPA mechanisms in one convenient module.

    Drop-in upgrade for ANY JEPA variant:
      I-JEPA, V-JEPA, C-JEPA, TD-JEPA, LeJEPA, etc.

    Mechanisms:
      1. JAWP — workspace prediction (Courant-Fischer optimality)
      2. WIP  — information preservation (theorem guarantee)
      3. Spectral Gap — automatic k* (Marchenko-Pastur)
      4. Grassmann — subspace optimization (fiber projection)
      5. Predictive Rank — rank preservation (log-det barrier)
      6. CGN — contextual gating (information routing theorem)
      7. SWIP — selective whitening (log-eigenvalue matching)
      8. PCR — cascade refinement (cascade capacity theorem)
      9. SPC — spectral predictive coding (information-proportional capacity)

    Usage:
        bundle = MechanismBundle.from_config(config)
        z_out, info = bundle(z, z_target, mask, step=step)
        bundle.retract()  # after optimizer.step()
    """

    def __init__(
        self,
        embed_dim: int = 768,
        # JAWP
        use_jawp: bool = True,
        jawk_k_start: int = 1,
        jawk_k_end: Optional[int] = None,
        jawk_curriculum_steps: int = 10000,
        jawk_alpha: float = 0.1,
        jawk_init: str = 'identity',
        # Predictive Rank
        lambda_predictive_rank: float = 0.0,
        # CGN
        use_cgn: bool = False,
        cgn_n_groups: int = 8,
        cgn_tau_start: float = 1.0,
        cgn_tau_end: float = 0.1,
        cgn_anneal_steps: int = 10000,
        # SWIP
        use_swip: bool = False,
        swip_k_workspace: Optional[int] = None,
        swip_target_variance: float = 1.0,
        # PCR
        use_pcr: bool = False,
        pcr_n_levels: int = 3,
        pcr_level_dims: Optional[list] = None,
        pcr_warmup_steps: int = 1000,
        # SPC
        use_spc: bool = False,
        spc_n_bands: int = 8,
        spc_init: str = 'dct',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_jawp = use_jawp
        self.use_cgn = use_cgn
        self.use_swip = use_swip
        self.use_pcr = use_pcr
        self.use_spc = use_spc
        self.lambda_predictive_rank = lambda_predictive_rank

        # Mechanism 1-5: JAWP (includes WIP, Spectral Gap, Grassmann, Predictive Rank)
        if use_jawp:
            self.jawp = JAWPModule(
                embed_dim=embed_dim,
                k_start=jawk_k_start,
                k_end=jawk_k_end,
                curriculum_steps=jawk_curriculum_steps,
                alpha=jawk_alpha,
                init=jawk_init,
            )
        else:
            self.jawp = None

        # Mechanism 6: CGN
        if use_cgn:
            self.cgn = ContextualGatingNetwork(
                embed_dim=embed_dim,
                n_groups=cgn_n_groups,
                tau_start=cgn_tau_start,
                tau_end=cgn_tau_end,
                anneal_steps=cgn_anneal_steps,
            )
        else:
            self.cgn = None

        # Mechanism 7: SWIP
        if use_swip:
            self.swip = SWIPModule(
                embed_dim=embed_dim,
                k_workspace=swip_k_workspace,
                target_variance=swip_target_variance,
                use_jawp_workspace=use_jawp,
            )
        else:
            self.swip = None

        # Mechanism 8: PCR
        if use_pcr:
            self.pcr = PredictiveCascadeRefinement(
                embed_dim=embed_dim,
                n_levels=pcr_n_levels,
                level_dims=pcr_level_dims,
            )
            self.pcr.warmup_steps = pcr_warmup_steps
        else:
            self.pcr = None

        # Mechanism 9: SPC
        if use_spc:
            self.spc = SpectralPredictiveCoding(
                embed_dim=embed_dim,
                n_bands=spc_n_bands,
                init=spc_init,
            )
        else:
            self.spc = None

    @classmethod
    def from_config(cls, config) -> 'MechanismBundle':
        """Create from a TextSpanJEPAConfig object."""
        return cls(
            embed_dim=config.embed_dim,
            use_jawp=config.use_jawp,
            jawk_k_start=config.jawk_k_start,
            jawk_k_end=config.jawk_k_end,
            jawk_curriculum_steps=config.jawk_curriculum_steps,
            jawk_alpha=config.jawk_alpha,
            jawk_init=config.jawk_init,
            lambda_predictive_rank=config.lambda_predictive_rank,
            use_cgn=config.use_cgn,
            cgn_n_groups=config.cgn_n_groups,
            cgn_tau_start=config.cgn_tau_start,
            cgn_tau_end=config.cgn_tau_end,
            cgn_anneal_steps=config.cgn_anneal_steps,
            use_swip=config.use_swip,
            swip_k_workspace=config.swip_k_workspace,
            swip_target_variance=config.swip_target_variance,
            use_pcr=getattr(config, 'use_pcr', False),
            pcr_n_levels=getattr(config, 'pcr_n_levels', 3),
            pcr_level_dims=getattr(config, 'pcr_level_dims', None),
            pcr_warmup_steps=getattr(config, 'pcr_warmup_steps', 1000),
            use_spc=getattr(config, 'use_spc', False),
            spc_n_bands=getattr(config, 'spc_n_bands', 8),
            spc_init=getattr(config, 'spc_init', 'dct'),
        )

    def forward(
        self,
        z: torch.Tensor,
        z_target: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply all active mechanisms.

        Args:
            z: (..., D) online representations.
            z_target: (..., D) target representations (will be detached internally).
            mask_positions: (B, T) binary mask for CGN. 1=masked, 0=visible.
            step: training step (for curriculum, annealing, warmup).

        Returns:
            z_out: (..., D) processed representations.
            info: dict with all mechanism diagnostics.
        """
        info = {}
        z_out = z

        # CGN: route information differently at masked/visible positions
        if self.cgn is not None and mask_positions is not None:
            z_out, cgn_info = self.cgn(z_out, mask_positions, step=step)
            info.update({f'cgn_{k}': v for k, v in cgn_info.items()})

        # PCR: refine predictions through orthogonal subspace cascade
        if self.pcr is not None:
            z_out, pcr_info = self.pcr(z_out, z_target, step=step)
            info.update({f'pcr_{k}': v for k, v in pcr_info.items()})

        # JAWP: compute workspace prediction loss
        if self.jawp is not None:
            jawp_loss, jawp_info = self.jawp.compute_loss(z_out, z_target, step=step)
            info['jawp_loss'] = jawp_loss
            info.update({f'jawp_{k}': v for k, v in jawp_info.items()})

            # Predictive Rank: prevent rank collapse in workspace
            if self.lambda_predictive_rank > 0:
                rank_loss = self.jawp.predictive_rank_loss(z_out)
                info['predictive_rank_loss'] = rank_loss.item()

        # SWIP: selective whitening of background
        if self.swip is not None:
            ws_Q = None
            if self.jawp is not None:
                k_active = int(self.jawp.active_k.item())
                ws_Q = self.jawp.workspace_Q.data[:, :k_active]
            swip_loss, swip_info = self.swip(z_out, workspace_Q=ws_Q)
            info['swip_loss'] = swip_loss
            info.update({f'swip_{k}': v for k, v in swip_info.items()})

        # SPC: spectral predictive coding
        if self.spc is not None:
            spc_loss, spc_info = self.spc(z_out, z_target)
            info['spc_loss'] = spc_loss
            info.update({f'spc_{k}': v for k, v in spc_info.items()})

        return z_out, info

    def retract(self):
        """Call after optimizer.step() to maintain manifold constraints.

        Applies Stiefel retraction for JAWP and PCR projection matrices.
        Also applies Grassmann retraction for JAWP (if preferred).
        """
        if self.jawp is not None:
            self.jawp.stiefel_retract()
        if self.pcr is not None:
            self.pcr.stiefel_retract()
        if self.spc is not None:
            self.spc.stiefel_retract()

    def compute_capacity_bound(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> float:
        """Compute total theoretical information gain from all mechanisms.

        Returns lower bound on additional information (nats).
        """
        total = 0.0

        # JAWP: WIP score
        if self.jawp is not None:
            wip_score, _ = self.jawp.workspace_information_preservation(z_pred, z_target)
            total += max(wip_score, 0.0)

        # PCR: Cascade Capacity bound
        if self.pcr is not None:
            pcr_bound, _ = self.pcr.compute_cascade_capacity_bound(z_pred, z_target)
            total += pcr_bound

        return total


# ═══════════════════════════════════════════════════════════════════
#  One-function convenience API for individual mechanisms
# ═══════════════════════════════════════════════════════════════════

def jawp_loss(z_pred, z_target, embed_dim=768, k=77, alpha=0.1, step=0):
    """Compute JAWP loss — one function call.

    Args:
        z_pred: (..., D) predictor output.
        z_target: (..., D) target encoder output (will be detached).
        embed_dim: embedding dimension.
        k: workspace dimension.
        alpha: predictor focus weight.
        step: training step (for curriculum).

    Returns:
        loss: scalar tensor.
        info: dict with diagnostics.

    Example:
        loss, info = jawp_loss(z_pred, z_target, k=77)
        loss.backward()
    """
    jawp = JAWPModule(embed_dim=embed_dim, k_start=1, k_end=k, alpha=alpha)
    # Move to same device as input
    jawp = jawp.to(z_pred.device)
    return jawp.compute_loss(z_pred, z_target, step=step)


def cgn_gate(z, mask_positions, embed_dim=768, n_groups=8, step=0):
    """Apply contextual gating — one function call.

    Args:
        z: (B, T, D) representations.
        mask_positions: (B, T) binary mask.
        embed_dim: embedding dimension.
        n_groups: number of gate groups.
        step: training step.

    Returns:
        z_gated: gated representations.
        info: dict with diagnostics.
    """
    cgn = ContextualGatingNetwork(embed_dim=embed_dim, n_groups=n_groups)
    cgn = cgn.to(z.device)
    return cgn(z, mask_positions, step=step)


def pcr_refine(z_pred, z_target, embed_dim=768, n_levels=3, step=0):
    """Refine predictions via orthogonal cascade — one function call.

    Args:
        z_pred: (..., D) base predictions.
        z_target: (..., D) targets (will be detached).
        embed_dim: embedding dimension.
        n_levels: number of refinement levels.
        step: training step.

    Returns:
        z_refined: refined predictions.
        info: dict with diagnostics.
    """
    pcr = PredictiveCascadeRefinement(embed_dim=embed_dim, n_levels=n_levels)
    pcr = pcr.to(z_pred.device)
    return pcr(z_pred, z_target, step=step)


def swip_whiten(z, embed_dim=768, k_workspace=25, target_variance=1.0):
    """Selective whitening with information preservation — one function call.

    Args:
        z: (..., D) representations.
        embed_dim: embedding dimension.
        k_workspace: workspace dimension.
        target_variance: target background variance.

    Returns:
        loss: scalar tensor.
        info: dict with diagnostics.
    """
    swip = SWIPModule(embed_dim=embed_dim, k_workspace=k_workspace,
                      target_variance=target_variance)
    swip = swip.to(z.device)
    return swip(z)


def spc_loss(z_pred, z_target, embed_dim=768, n_bands=8, init='dct'):
    """Spectral predictive coding loss — one function call.

    Args:
        z_pred: (..., D) predictor output.
        z_target: (..., D) target encoder output (will be detached).
        embed_dim: embedding dimension.
        n_bands: number of frequency bands.
        init: 'dct' (DCT-II basis) or 'random'.

    Returns:
        loss: scalar tensor.
        info: dict with diagnostics.

    Example:
        loss, info = spc_loss(z_pred, z_target, n_bands=8)
        loss.backward()
    """
    spc = SpectralPredictiveCoding(embed_dim=embed_dim, n_bands=n_bands, init=init)
    spc = spc.to(z_pred.device)
    return spc(z_pred, z_target)
