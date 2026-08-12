# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Convenience API: One-line access to all 13 novel mechanisms.
#
# ═══════════════════════════════════════════════════════════════════════════
#  USAGE (3 lines to upgrade any JEPA with all mechanisms)
# ═══════════════════════════════════════════════════════════════════════════
#
#  from src.models.mechanisms import MechanismBundle
#
#  bundle = MechanismBundle.from_config(config)          # line 1
#  z_refined, all_info = bundle(z_pred, z_target,        # line 2
#                               mask_positions, step)
#  bundle.retract()                                       # line 3
#
#  That's it. All 11 mechanisms in 3 lines.
#
#  Mechanisms:
#    1.  JAWP  — workspace prediction (Courant-Fischer optimality)
#    2.  WIP   — information preservation (theorem guarantee)
#    3.  Spectral Gap — automatic k* (Marchenko-Pastur)
#    4.  Grassmann — subspace optimization (fiber projection)
#    5.  Predictive Rank — rank preservation (log-det barrier)
#    6.  CGN   — contextual gating (information routing theorem)
#    7.  SWIP  — selective whitening (log-eigenvalue matching)
#    8.  PCR   — cascade refinement (cascade capacity theorem)
#    9.  SPC   — spectral predictive coding (info-proportional capacity)
#    10. WSD   — workspace-target sync drift (drift bound theorem)
#    11. CMC   — cross-mask consistency (stability theorem)
#
#  You can also use individual mechanisms:
#    from src.models.mechanisms import jawp_loss, cgn_gate, pcr_refine, ...

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from .jawp import JAWPModule
from .cgn import ContextualGatingNetwork
from .swip import SWIPModule
from .pcr import PredictiveCascadeRefinement
from .spc import SpectralPredictiveCoding
from .wsd import WorkspaceSyncDrift
from .gac import GradientAllocatedCapacity
from .cmc import CrossMaskConsistency
from .sta import SpectralTransportAlignment


class MechanismBundle(nn.Module):
    """All 13 Text-Span JEPA mechanisms in one convenient module.

    Drop-in upgrade for ANY JEPA variant:
      I-JEPA, V-JEPA, C-JEPA, TD-JEPA, LeJEPA, etc.

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
        # WSD
        use_wsd: bool = False,
        wsd_k: Optional[int] = None,
        wsd_sync_interval: int = 100,
        wsd_ema_beta: float = 0.99,
        # CMC
        use_cmc: bool = False,
        cmc_second_mask_ratio: Optional[float] = None,
        cmc_min_overlap_ratio: float = 0.2,
        cmc_mode: str = 'interval',
        cmc_interval: int = 10,
        # GAC
        use_gac: bool = False,
        gac_gamma: float = 0.01,
        gac_tau_grad: float = 1e-4,
        gac_warmup_steps: int = 1000,
        # STA
        use_sta: bool = False,
        sta_eta: float = 0.01,
        sta_ema_beta: float = 0.999,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_jawp = use_jawp
        self.use_cgn = use_cgn
        self.use_swip = use_swip
        self.use_pcr = use_pcr
        self.use_spc = use_spc
        self.use_wsd = use_wsd
        self.use_cmc = use_cmc
        self.use_gac = use_gac
        self.use_sta = use_sta
        self.lambda_predictive_rank = lambda_predictive_rank

        # Mechanism 1-5: JAWP
        if use_jawp:
            self.jawp = JAWPModule(
                embed_dim=embed_dim, k_start=jawk_k_start,
                k_end=jawk_k_end, curriculum_steps=jawk_curriculum_steps,
                alpha=jawk_alpha, init=jawk_init,
            )
        else:
            self.jawp = None

        # Mechanism 6: CGN
        if use_cgn:
            self.cgn = ContextualGatingNetwork(
                embed_dim=embed_dim, n_groups=cgn_n_groups,
                tau_start=cgn_tau_start, tau_end=cgn_tau_end,
                anneal_steps=cgn_anneal_steps,
            )
        else:
            self.cgn = None

        # Mechanism 7: SWIP
        if use_swip:
            self.swip = SWIPModule(
                embed_dim=embed_dim, k_workspace=swip_k_workspace,
                target_variance=swip_target_variance,
                use_jawp_workspace=use_jawp,
            )
        else:
            self.swip = None

        # Mechanism 8: PCR
        if use_pcr:
            self.pcr = PredictiveCascadeRefinement(
                embed_dim=embed_dim, n_levels=pcr_n_levels,
                level_dims=pcr_level_dims,
            )
            self.pcr.warmup_steps = pcr_warmup_steps
        else:
            self.pcr = None

        # Mechanism 9: SPC
        if use_spc:
            self.spc = SpectralPredictiveCoding(
                embed_dim=embed_dim, n_bands=spc_n_bands, init=spc_init,
            )
        else:
            self.spc = None

        # Mechanism 10: WSD
        if use_wsd and use_jawp:
            k_ws = wsd_k or (jawk_k_end or embed_dim // 10)
            self.wsd = WorkspaceSyncDrift(
                embed_dim=embed_dim, k=k_ws,
                sync_interval=wsd_sync_interval, ema_beta=wsd_ema_beta,
            )
        else:
            self.wsd = None

        # Mechanism 11: CMC
        if use_cmc:
            self.cmc = CrossMaskConsistency(
                embed_dim=embed_dim,
                second_mask_ratio=cmc_second_mask_ratio,
                min_overlap_ratio=cmc_min_overlap_ratio,
                mode=cmc_mode, interval=cmc_interval,
            )
        else:
            self.cmc = None

        # Mechanism 12: GAC
        if use_gac:
            self.gac = GradientAllocatedCapacity(
                embed_dim=embed_dim, gamma=gac_gamma,
                tau_grad=gac_tau_grad, warmup_steps=gac_warmup_steps,
            )
        else:
            self.gac = None

        # Mechanism 13: STA
        if use_sta:
            self.sta = SpectralTransportAlignment(
                embed_dim=embed_dim, eta=sta_eta, ema_beta=sta_ema_beta,
            )
        else:
            self.sta = None

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
            use_wsd=getattr(config, 'use_wsd', False),
            wsd_k=getattr(config, 'wsd_k', None),
            wsd_sync_interval=getattr(config, 'wsd_sync_interval', 100),
            wsd_ema_beta=getattr(config, 'wsd_ema_beta', 0.99),
            use_cmc=getattr(config, 'use_cmc', False),
            cmc_second_mask_ratio=getattr(config, 'cmc_second_mask_ratio', None),
            cmc_min_overlap_ratio=getattr(config, 'cmc_min_overlap_ratio', 0.2),
            cmc_mode=getattr(config, 'cmc_mode', 'interval'),
            cmc_interval=getattr(config, 'cmc_interval', 10),
            use_gac=getattr(config, 'use_gac', False),
            gac_gamma=getattr(config, 'gac_gamma', 0.01),
            gac_tau_grad=getattr(config, 'gac_tau_grad', 1e-4),
            gac_warmup_steps=getattr(config, 'gac_warmup_steps', 1000),
            use_sta=getattr(config, 'use_sta', False),
            sta_eta=getattr(config, 'sta_eta', 0.01),
            sta_ema_beta=getattr(config, 'sta_ema_beta', 0.999),
        )

    def forward(
        self,
        z: torch.Tensor,
        z_target: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply all active mechanisms."""
        info = {}
        z_out = z

        # CGN
        if self.cgn is not None and mask_positions is not None:
            z_out, cgn_info = self.cgn(z_out, mask_positions, step=step)
            info.update({f'cgn_{k}': v for k, v in cgn_info.items()})

        # PCR
        if self.pcr is not None:
            z_out, pcr_info = self.pcr(z_out, z_target, step=step)
            info.update({f'pcr_{k}': v for k, v in pcr_info.items()})

        # JAWP
        if self.jawp is not None:
            jawp_loss, jawp_info = self.jawp.compute_loss(z_out, z_target, step=step)
            info['jawp_loss'] = jawp_loss
            info.update({f'jawp_{k}': v for k, v in jawp_info.items()})
            if self.lambda_predictive_rank > 0:
                rank_loss = self.jawp.predictive_rank_loss(z_out)
                info['predictive_rank_loss'] = rank_loss.item()

        # SWIP
        if self.swip is not None:
            ws_Q = None
            if self.jawp is not None:
                k_active = int(self.jawp.active_k.item())
                ws_Q = self.jawp.workspace_Q.data[:, :k_active]
            swip_loss, swip_info = self.swip(z_out, workspace_Q=ws_Q)
            info['swip_loss'] = swip_loss
            info.update({f'swip_{k}': v for k, v in swip_info.items()})

        # SPC
        if self.spc is not None:
            spc_loss, spc_info = self.spc(z_out, z_target)
            info['spc_loss'] = spc_loss
            info.update({f'spc_{k}': v for k, v in spc_info.items()})

        # WSD
        if self.wsd is not None and self.jawp is not None:
            k_active = int(self.jawp.active_k.item())
            Q_jawp = self.jawp.workspace_Q.data[:, :k_active]
            wsd_loss, wsd_info = self.wsd.compute_drift(Q_jawp, z_target, step=step)
            info['wsd_loss'] = wsd_loss
            info.update({f'wsd_{k}': v for k, v in wsd_info.items()})

        # STA
        if self.sta is not None:
            sta_loss, sta_info = self.sta(z_out, step=step)
            info['sta_loss'] = sta_loss
            info.update({f'sta_{k}': v for k, v in sta_info.items()})

        return z_out, info

    def compute_cmc_loss(
        self,
        z_pred_primary: torch.Tensor,
        z_pred_secondary: torch.Tensor,
        overlap_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """Compute CMC loss (requires second forward pass, call separately).

        Args:
            z_pred_primary: (B, T, D) predictions from primary mask.
            z_pred_secondary: (B, T, D) predictions from secondary mask.
            overlap_mask: (B, T) binary — 1 if masked in BOTH masks.

        Returns:
            loss: scalar tensor.
            info: dict with diagnostics.
        """
        if self.cmc is None:
            zero = torch.tensor(0.0, device=z_pred_primary.device)
            return zero, {'cmc_loss': 0.0, 'cmc_skipped': True}
        return self.cmc(z_pred_primary, z_pred_secondary, overlap_mask)

    def retract(self):
        """Call after optimizer.step() to maintain manifold constraints."""
        if self.jawp is not None:
            self.jawp.stiefel_retract()
        if self.pcr is not None:
            self.pcr.stiefel_retract()
        if self.spc is not None:
            self.spc.stiefel_retract()

    def compute_capacity_bound(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> float:
        """Compute total theoretical information gain from all mechanisms."""
        total = 0.0
        if self.jawp is not None:
            wip_score, _ = self.jawp.workspace_information_preservation(z_pred, z_target)
            total += max(wip_score, 0.0)
        if self.pcr is not None:
            pcr_bound, _ = self.pcr.compute_cascade_capacity_bound(z_pred, z_target)
            total += pcr_bound
        return total


# ═══════════════════════════════════════════════════════════════════
#  One-function convenience API for individual mechanisms
# ═══════════════════════════════════════════════════════════════════

def jawp_loss(z_pred, z_target, embed_dim=768, k=77, alpha=0.1, step=0):
    """Compute JAWP loss — one function call."""
    jawp = JAWPModule(embed_dim=embed_dim, k_start=1, k_end=k, alpha=alpha)
    jawp = jawp.to(z_pred.device)
    return jawp.compute_loss(z_pred, z_target, step=step)


def cgn_gate(z, mask_positions, embed_dim=768, n_groups=8, step=0):
    """Apply contextual gating — one function call."""
    cgn = ContextualGatingNetwork(embed_dim=embed_dim, n_groups=n_groups)
    cgn = cgn.to(z.device)
    return cgn(z, mask_positions, step=step)


def pcr_refine(z_pred, z_target, embed_dim=768, n_levels=3, step=0):
    """Refine predictions via orthogonal cascade — one function call."""
    pcr = PredictiveCascadeRefinement(embed_dim=embed_dim, n_levels=n_levels)
    pcr = pcr.to(z_pred.device)
    return pcr(z_pred, z_target, step=step)


def swip_whiten(z, embed_dim=768, k_workspace=25, target_variance=1.0):
    """Selective whitening with information preservation — one function call."""
    swip = SWIPModule(embed_dim=embed_dim, k_workspace=k_workspace,
                      target_variance=target_variance)
    swip = swip.to(z.device)
    return swip(z)


def spc_loss(z_pred, z_target, embed_dim=768, n_bands=8, init='dct'):
    """Spectral predictive coding loss — one function call."""
    spc = SpectralPredictiveCoding(embed_dim=embed_dim, n_bands=n_bands, init=init)
    spc = spc.to(z_pred.device)
    return spc(z_pred, z_target)


def wsd_drift(Q_jawp, z_target, embed_dim=768, k=77, sync_interval=100):
    """Workspace-target synchronization drift — one function call."""
    wsd = WorkspaceSyncDrift(embed_dim=embed_dim, k=k, sync_interval=sync_interval)
    wsd = wsd.to(Q_jawp.device)
    return wsd.compute_drift(Q_jawp, z_target, step=0)


def cmc_consistency(z_pred_1, z_pred_2, overlap_mask, embed_dim=768):
    """Cross-mask consistency loss — one function call."""
    cmc = CrossMaskConsistency(embed_dim=embed_dim)
    cmc = cmc.to(z_pred_1.device)
    return cmc(z_pred_1, z_pred_2, overlap_mask)


def gac_explore(z_pred, grad_norms, embed_dim=768, gamma=0.01, tau_grad=1e-4, step=0):
    """Gradient-allocated capacity loss — one function call."""
    gac = GradientAllocatedCapacity(embed_dim=embed_dim, gamma=gamma, tau_grad=tau_grad)
    gac = gac.to(z_pred.device)
    return gac(z_pred, grad_norms, step=step)


def sta_align(z, embed_dim=768, eta=0.01, ema_beta=0.999, step=0):
    """Spectral transport alignment loss — one function call."""
    sta = SpectralTransportAlignment(embed_dim=embed_dim, eta=eta, ema_beta=ema_beta)
    sta = sta.to(z.device)
    return sta(z, step=step)
