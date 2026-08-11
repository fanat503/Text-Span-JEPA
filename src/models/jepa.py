# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Main JEPA model: glues encoder, predictor, decoder, collapse prevention
# Training loop patterns from I-JEPA (Assran et al., CVPR 2023)
# Target centering + layer norm from data2vec (Baevski et al., ICML 2022)
# VICReg collapse prevention from C-JEPA (NeurIPS 2024) / VICReg (ICLR 2022)

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TextSpanJEPAEncoder
from .predictor import TextSpanJPAPredictor
from .decoder import TiedTokenDecoder
from .collapse import (
    VarianceRegularization,
    CovarianceRegularization,
    TargetCentering,
    CollapseDiagnostics,
)
from .sigreg import SIGReg, WeakSIGReg, VISReg
from .jspace import JSpaceMetrics
from .jawp import JAWPModule
from .cgn import ContextualGatingNetwork
from .swip import SWIPModule


class TextSpanJEPAConfig:
    """Configuration for Text-Span JEPA model."""

    def __init__(self, **kwargs):
        # Encoder
        self.vocab_size = kwargs.get('vocab_size', 50304)
        self.max_seq_len = kwargs.get('max_seq_len', 512)
        self.embed_dim = kwargs.get('embed_dim', 768)
        self.encoder_depth = kwargs.get('encoder_depth', 12)
        self.num_heads = kwargs.get('num_heads', 12)
        self.mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        self.qkv_bias = kwargs.get('qkv_bias', True)
        self.drop_rate = kwargs.get('drop_rate', 0.0)
        self.attn_drop_rate = kwargs.get('attn_drop_rate', 0.0)
        self.drop_path_rate = kwargs.get('drop_path_rate', 0.1)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)

        # Predictor
        self.predictor_embed_dim = kwargs.get('predictor_embed_dim', 384)
        self.predictor_depth = kwargs.get('predictor_depth', 6)
        self.future_offsets = kwargs.get('future_offsets', (1, 4, 16))
        self.num_refine_steps = kwargs.get('num_refine_steps', 3)
        self.refine_step_size = kwargs.get('refine_step_size', 0.1)

        # Decoder
        self.decoder_bias = kwargs.get('decoder_bias', False)

        # Collapse prevention
        self.variance_margin = kwargs.get('variance_margin', 1.0)
        self.centering_momentum = kwargs.get('centering_momentum', 0.9)

        # EMA target encoder
        self.ema_tau_start = kwargs.get('ema_tau_start', 0.996)
        self.ema_tau_end = kwargs.get('ema_tau_end', 0.9996)
        self.ema_schedule = kwargs.get('ema_schedule', 'cosine')

        # Loss weights
        self.lambda_span = kwargs.get('lambda_span', 1.0)
        self.lambda_future = kwargs.get('lambda_future', 0.5)
        self.lambda_decoder = kwargs.get('lambda_decoder', 0.1)
        self.lambda_variance = kwargs.get('lambda_variance', 0.1)
        self.lambda_covariance = kwargs.get('lambda_covariance', 0.04)

        # Mask curriculum
        self.mask_ratio_start = kwargs.get('mask_ratio_start', 0.15)
        self.mask_ratio_end = kwargs.get('mask_ratio_end', 0.35)

        # Future loss warmup
        self.future_warmup_steps = kwargs.get('future_warmup_steps', 0)

        # SIGReg
        self.lambda_sigreg = kwargs.get('lambda_sigreg', 0.0)
        self.sigreg_n_sketches = kwargs.get('sigreg_n_sketches', 64)
        self.sigreg_n_integration_points = kwargs.get('sigreg_n_integration_points', 17)
        self.sigreg_sigma = kwargs.get('sigreg_sigma', 1.0)

        # J-Space
        self.jspace_variance_threshold = kwargs.get('jspace_variance_threshold', 0.10)
        self.jspace_k_workspace = kwargs.get('jspace_k_workspace', 25)

        # JAWP
        self.use_jawp = kwargs.get('use_jawp', True)
        self.jawk_k_start = kwargs.get('jawk_k_start', 1)
        self.jawk_k_end = kwargs.get('jawk_k_end', None)
        self.jawk_curriculum_steps = kwargs.get('jawk_curriculum_steps', 10000)
        self.jawk_alpha = kwargs.get('jawk_alpha', 0.1)
        self.jawk_init = kwargs.get('jawk_init', 'identity')

        # Predictive Rank Regularization (from JAWP module)
        self.lambda_predictive_rank = kwargs.get('lambda_predictive_rank', 0.0)

        # CGN: Contextual Gating Network (novel mechanism #6)
        self.use_cgn = kwargs.get('use_cgn', False)
        self.cgn_n_groups = kwargs.get('cgn_n_groups', 8)
        self.cgn_tau_start = kwargs.get('cgn_tau_start', 1.0)
        self.cgn_tau_end = kwargs.get('cgn_tau_end', 0.1)
        self.cgn_anneal_steps = kwargs.get('cgn_anneal_steps', 10000)
        self.lambda_cgn_ortho = kwargs.get('lambda_cgn_ortho', 0.0)

        # SWIP: Selective Whitening with Information Preservation (novel mechanism #7)
        self.use_swip = kwargs.get('use_swip', False)
        self.swip_k_workspace = kwargs.get('swip_k_workspace', None)
        self.swip_target_variance = kwargs.get('swip_target_variance', 1.0)
        self.lambda_swip = kwargs.get('lambda_swip', 0.0)

    def validate(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}")
        if self.predictor_embed_dim % self.num_heads != 0:
            raise ValueError(f"predictor_embed_dim={self.predictor_embed_dim} must be divisible by num_heads={self.num_heads}")
        if self.encoder_depth < 1:
            raise ValueError(f"encoder_depth must be >= 1, got {self.encoder_depth}")
        if self.predictor_depth < 1:
            raise ValueError(f"predictor_depth must be >= 1, got {self.predictor_depth}")
        if self.ema_schedule not in ('cosine', 'linear'):
            raise ValueError(f"ema_schedule must be 'cosine' or 'linear', got '{self.ema_schedule}'")
        if self.lambda_span < 0:
            raise ValueError(f"lambda_span must be >= 0, got {self.lambda_span}")
        if self.lambda_future < 0:
            raise ValueError(f"lambda_future must be >= 0, got {self.lambda_future}")
        if self.variance_margin <= 0:
            raise ValueError(f"variance_margin must be > 0, got {self.variance_margin}")
        if not 0 < self.centering_momentum < 1:
            raise ValueError(f"centering_momentum must be in (0,1), got {self.centering_momentum}")
        if self.use_jawp:
            if self.jawk_k_start < 1:
                raise ValueError(f"jawk_k_start must be >= 1, got {self.jawk_k_start}")
            if self.jawk_k_end is not None and self.jawk_k_end > self.embed_dim:
                raise ValueError(f"jawk_k_end={self.jawk_k_end} cannot exceed embed_dim={self.embed_dim}")
            if self.jawk_k_end is not None and self.jawk_k_start > self.jawk_k_end:
                raise ValueError(f"jawk_k_start={self.jawk_k_start} > jawk_k_end={self.jawk_k_end}")
            if self.jawk_alpha < 0:
                raise ValueError(f"jawk_alpha must be >= 0, got {self.jawk_alpha}")
            if self.jawk_init not in ('identity', 'random', 'pca'):
                raise ValueError(f"jawk_init must be 'identity', 'random', or 'pca', got '{self.jawk_init}'")
        if self.lambda_sigreg < 0:
            raise ValueError(f"lambda_sigreg must be >= 0, got {self.lambda_sigreg}")
        if self.lambda_sigreg > 0 and self.sigreg_sigma <= 0:
            raise ValueError(f"sigreg_sigma must be > 0 when SIGReg is active, got {self.sigreg_sigma}")
        if self.future_warmup_steps < 0:
            raise ValueError(f"future_warmup_steps must be >= 0, got {self.future_warmup_steps}")
        if self.lambda_predictive_rank < 0:
            raise ValueError(f"lambda_predictive_rank must be >= 0, got {self.lambda_predictive_rank}")
        if self.use_cgn:
            if self.embed_dim % self.cgn_n_groups != 0:
                raise ValueError(f"embed_dim={self.embed_dim} must be divisible by cgn_n_groups={self.cgn_n_groups}")
            if self.cgn_n_groups < 1:
                raise ValueError(f"cgn_n_groups must be >= 1, got {self.cgn_n_groups}")
            if self.cgn_tau_start <= 0 or self.cgn_tau_end <= 0:
                raise ValueError(f"cgn temperatures must be > 0, got start={self.cgn_tau_start}, end={self.cgn_tau_end}")
            if self.lambda_cgn_ortho < 0:
                raise ValueError(f"lambda_cgn_ortho must be >= 0, got {self.lambda_cgn_ortho}")
        if self.use_swip:
            if self.swip_k_workspace is not None and self.swip_k_workspace > self.embed_dim:
                raise ValueError(f"swip_k_workspace={self.swip_k_workspace} cannot exceed embed_dim={self.embed_dim}")
            if self.swip_target_variance <= 0:
                raise ValueError(f"swip_target_variance must be > 0, got {self.swip_target_variance}")
            if self.lambda_swip < 0:
                raise ValueError(f"lambda_swip must be >= 0, got {self.lambda_swip}")
        return True


class TextSpanJEPA(nn.Module):
    """Text-Span JEPA: Latent Predictive Learning for Language Representations."""

    def __init__(self, config: TextSpanJEPAConfig):
        super().__init__()
        self.config = config

        self.encoder = TextSpanJEPAEncoder(
            vocab_size=config.vocab_size, max_seq_len=config.max_seq_len,
            embed_dim=config.embed_dim, depth=config.encoder_depth,
            num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias, drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate, drop_path_rate=config.drop_path_rate,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = TextSpanJPAPredictor(
            embed_dim=config.embed_dim, predictor_embed_dim=config.predictor_embed_dim,
            depth=config.predictor_depth, num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio, max_seq_len=config.max_seq_len,
            future_offsets=config.future_offsets, num_refine_steps=config.num_refine_steps,
            refine_step_size=config.refine_step_size,
        )

        self.decoder = TiedTokenDecoder(
            embed_dim=config.embed_dim, vocab_size=config.vocab_size,
            bias=config.decoder_bias,
        )

        self.variance_reg = VarianceRegularization(margin=config.variance_margin)
        self.covariance_reg = CovarianceRegularization()
        self.target_centering = TargetCentering(dim=config.embed_dim, momentum=config.centering_momentum)
        self.diagnostics = CollapseDiagnostics()

        self.sigreg = SIGReg(
            embed_dim=config.embed_dim,
            n_sketches=config.sigreg_n_sketches,
            n_integration_points=config.sigreg_n_integration_points,
            sigma=config.sigreg_sigma,
        )

        self.jspace_metrics = JSpaceMetrics(
            variance_threshold=config.jspace_variance_threshold,
            k_workspace=config.jspace_k_workspace,
        )

        # JAWP — novel mechanism: Jacobian-Aligned Workspace Prediction
        if config.use_jawp:
            self.jawp = JAWPModule(
                embed_dim=config.embed_dim,
                k_start=config.jawk_k_start,
                k_end=config.jawk_k_end,
                curriculum_steps=config.jawk_curriculum_steps,
                alpha=config.jawk_alpha,
                init=config.jawk_init,
            )
        else:
            self.jawp = None

        # CGN — novel mechanism #6: Contextual Gating Network
        if config.use_cgn:
            self.cgn = ContextualGatingNetwork(
                embed_dim=config.embed_dim,
                n_groups=config.cgn_n_groups,
                tau_start=config.cgn_tau_start,
                tau_end=config.cgn_tau_end,
                anneal_steps=config.cgn_anneal_steps,
            )
        else:
            self.cgn = None

        # SWIP — novel mechanism #7: Selective Whitening with Information Preservation
        if config.use_swip:
            self.swip = SWIPModule(
                embed_dim=config.embed_dim,
                k_workspace=config.swip_k_workspace,
                target_variance=config.swip_target_variance,
                use_jawp_workspace=config.use_jawp,
            )
        else:
            self.swip = None
    def update_target_encoder(self, tau):
        """EMA update: param_k <- tau * param_k + (1 - tau) * param_q.

        Micro-opt: precompute (1 - tau) outside loop.
        I-JEPA pattern: called once per training step.
        """
        one_minus_tau = 1.0 - tau
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(tau).add_(one_minus_tau * param_q.data)

    def _future_loss_weight(self, current_step):
        if self.config.future_warmup_steps <= 0:
            return self.config.lambda_future
        if current_step >= self.config.future_warmup_steps:
            return self.config.lambda_future
        progress = current_step / self.config.future_warmup_steps
        return self.config.lambda_future * progress

    def compute_loss_with_targets(self, masked_input_ids, original_input_ids,
                                   mask_positions, current_step=0, total_steps=1):
        if masked_input_ids.size(0) == 0:
            zero = torch.tensor(0.0, device=masked_input_ids.device)
            return zero, {'loss': 0.0, 'loss_span': 0.0, 'loss_future': 0.0,
                          'loss_decoder': 0.0, 'loss_variance': 0.0,
                          'loss_covariance': 0.0, 'decoder_accuracy': 0.0,
                          'future_weight': 0.0}, {}

        h_online, token_embeds_online = self.encoder(masked_input_ids)

        with torch.no_grad():
            self._prev_target_h = getattr(self, '_prev_target_h', None)
            h_target, _ = self.target_encoder(original_input_ids)
            h_target = self.target_centering(h_target)
            h_target = F.layer_norm(h_target, (h_target.size(-1),))

        # CGN: apply contextual gating before predictor
        # Routes information differently at masked vs visible positions
        cgn_info = {}
        if self.cgn is not None:
            h_online, cgn_info = self.cgn(h_online, mask_positions, step=current_step)

        span_preds, num_masked, valid_mask, future_losses, future_preds = self.predictor(
            h_online, mask_positions, token_embeds_online, h_target.detach()
        )

        # Zero-loss helper: avoids creating unnecessary computation graph nodes.
        # h_online.sum() * 0.0 still requires grad through h_online.
        # Instead, create a proper zero loss that participates in the graph.
        _zero_loss = h_online.new_tensor(0.0, requires_grad=True)
        if valid_mask.any():
            target_gathered, _, target_valid = TextSpanJPAPredictor._gather_masked(
                h_target.detach(), mask_positions
            )
            min_cols = min(valid_mask.size(1), target_valid.size(1))
            combined_valid = valid_mask[:, :min_cols] & target_valid[:, :min_cols]
            if combined_valid.any():
                if self.jawp is not None:
                    span_preds_valid = span_preds[:, :min_cols][combined_valid]
                    target_gathered_valid = target_gathered[:, :min_cols][combined_valid]
                    loss_span, jawp_info = self.jawp.compute_loss(
                        span_preds_valid, target_gathered_valid, step=current_step
                    )
                else:
                    loss_span = F.smooth_l1_loss(
                        span_preds[:, :min_cols][combined_valid],
                        target_gathered[:, :min_cols][combined_valid]
                    )
                    jawp_info = {}
            else:
                loss_span = _zero_loss
                jawp_info = {}
        else:
            loss_span = _zero_loss
            jawp_info = {}

        if len(future_losses) > 0:
            loss_future = sum(future_losses.values()) / len(future_losses)
        else:
            loss_future = torch.tensor(0.0, device=h_online.device, requires_grad=True)
        future_weight = self._future_loss_weight(current_step)

        loss_decoder = _zero_loss
        decoder_acc = torch.tensor(0.0, device=h_online.device)
        if mask_positions.any():
            target_tokens = original_input_ids[mask_positions.bool()]
            h_at_masked = h_online[mask_positions.bool()]
            if h_at_masked.size(0) > 0:
                logits = self.decoder(h_at_masked, self.encoder.token_embedding.weight)
                loss_decoder = F.cross_entropy(logits, target_tokens)
                decoder_acc = (logits.argmax(dim=-1) == target_tokens).float().mean()

        loss_variance = self.variance_reg(h_online)
        loss_covariance = self.covariance_reg(h_online)

        lambda_sigreg = self.config.lambda_sigreg
        if lambda_sigreg > 0:
            loss_sigreg = self.sigreg(h_online)
        else:
            loss_sigreg = _zero_loss

        # Predictive Rank Regularization (prevents rank collapse in workspace)
        lambda_pred_rank = self.config.lambda_predictive_rank
        if lambda_pred_rank > 0 and self.jawp is not None:
            if valid_mask.any() and span_preds.size(0) > 1:
                loss_pred_rank = self.jawp.predictive_rank_loss(span_preds)
            else:
                loss_pred_rank = _zero_loss
        else:
            loss_pred_rank = _zero_loss

        # CGN orthogonality loss: encourage visible/masked gates to differ
        lambda_cgn_ortho = self.config.lambda_cgn_ortho
        if lambda_cgn_ortho > 0 and self.cgn is not None:
            # Loss = 1 - orthogonality (minimizing pushes orthogonality toward 1)
            # Use differentiable gate logits directly:
            # cos_sim(g_visible, g_masked) — we want this close to 0 (orthogonal)
            probs_v = F.softmax(self.cgn.gate_logits_visible, dim=-1)[:, 1]
            probs_m = F.softmax(self.cgn.gate_logits_masked, dim=-1)[:, 1]
            cos_sim = F.cosine_similarity(
                probs_v.unsqueeze(0), probs_m.unsqueeze(0)
            )
            # loss = cos_sim² → minimize to make gates orthogonal
            loss_cgn_ortho = cos_sim.pow(2)
        else:
            loss_cgn_ortho = _zero_loss

        # SWIP: Selective Whitening with Information Preservation
        # Whitens background noise while preserving workspace structure
        lambda_swip = self.config.lambda_swip
        if lambda_swip > 0 and self.swip is not None:
            ws_Q = None
            if self.jawp is not None:
                k_active = int(self.jawp.active_k.item())
                ws_Q = self.jawp.workspace_Q.data[:, :k_active]
            loss_swip, swip_info = self.swip(h_online, workspace_Q=ws_Q)
        else:
            loss_swip = _zero_loss
            swip_info = {}

        total_loss = (
            self.config.lambda_span * loss_span
            + future_weight * loss_future
            + self.config.lambda_decoder * loss_decoder
            + self.config.lambda_variance * loss_variance
            + self.config.lambda_covariance * loss_covariance
            + lambda_sigreg * loss_sigreg
            + lambda_pred_rank * loss_pred_rank
            + lambda_cgn_ortho * loss_cgn_ortho
            + lambda_swip * loss_swip
        )

        loss_dict = {
            'loss': total_loss.item(),
            'loss_span': loss_span.item(),
            'loss_future': loss_future.item(),
            'loss_decoder': loss_decoder.item(),
            'loss_variance': loss_variance.item(),
            'loss_covariance': loss_covariance.item(),
            'loss_sigreg': loss_sigreg.item(),
            'loss_predictive_rank': loss_pred_rank.item(),
            'loss_cgn_ortho': loss_cgn_ortho.item(),
            'loss_swip': loss_swip.item(),
            'decoder_accuracy': decoder_acc.item(),
            'future_weight': future_weight,
        }
        for d, l in future_losses.items():
            loss_dict[f'loss_future_d{d}'] = l.item()
        if jawp_info:
            loss_dict.update({f"jawk_{k}": v for k, v in jawp_info.items()})
        if cgn_info:
            loss_dict.update({f"cgn_{k}": v for k, v in cgn_info.items()})
        if swip_info:
            loss_dict.update({f"swip_{k}": v for k, v in swip_info.items()})

        diag_dict = self.diagnostics.compute(h_online.detach(), h_target.detach(),
                                              prev_target_h=self._prev_target_h)
        diag_dict['target_center_norm'] = self.target_centering.center.norm().item()
        diag_dict['mask_fraction'] = mask_positions.float().mean().item()

        jspace_dict = self.jspace_metrics.compute(h_online.detach(), h_target.detach(), predictor_h=None)
        diag_dict.update(jspace_dict)

        if h_online.size(0) * h_online.size(1) >= 2:
            diag_dict['embedding_std_per_dim'] = h_online.std(dim=(0, 1)).mean().item()
        else:
            diag_dict['embedding_std_per_dim'] = 0.0

        # workspace_quality composite metric — single scalar health score
        diag_dict['workspace_quality'] = CollapseDiagnostics.workspace_quality(diag_dict)

        self._prev_target_h = h_target.detach().clone()
        return total_loss, loss_dict, diag_dict

    def get_num_params(self, non_embedding=True):
        enc = self.encoder.get_num_params(non_embedding)
        pred = self.predictor.get_num_params()
        dec = sum(p.numel() for p in self.decoder.parameters())
        return enc + pred + dec

    def get_num_params_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
