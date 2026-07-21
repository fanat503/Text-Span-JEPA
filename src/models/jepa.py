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

from .encoder import TextSpanJEPLEncoder
from .predictor import TextSpanJEPApredictor
from .decoder import TiedTokenDecoder
from .collapse import (
    VarianceRegularization,
    CovarianceRegularization,
    TargetCentering,
    CollapseDiagnostics,
)


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

        # EMA target encoder — scheduled tau (NOT constant, per I-JEPA)
        self.ema_tau_start = kwargs.get('ema_tau_start', 0.996)
        self.ema_tau_end = kwargs.get('ema_tau_end', 1.0)

        # Loss weights
        self.lambda_span = kwargs.get('lambda_span', 1.0)
        self.lambda_future = kwargs.get('lambda_future', 0.5)
        self.lambda_decoder = kwargs.get('lambda_decoder', 0.1)
        self.lambda_variance = kwargs.get('lambda_variance', 0.1)
        self.lambda_covariance = kwargs.get('lambda_covariance', 0.04)

        # Mask curriculum
        self.mask_ratio_start = kwargs.get('mask_ratio_start', 0.15)
        self.mask_ratio_end = kwargs.get('mask_ratio_end', 0.35)

        # Fix #2: Future loss warmup (prevents instability from early target encoder)
        self.future_warmup_steps = kwargs.get('future_warmup_steps', 0)

    def validate(self):
        """Validate config values — catch dimension errors before model creation.

        Checks that embed_dim and predictor_embed_dim are divisible by num_heads.
        """
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by "
                f"num_heads={self.num_heads} (got {self.embed_dim}/{self.num_heads}"
                f"={self.embed_dim/self.num_heads:.1f})")
        if self.predictor_embed_dim % self.num_heads != 0:
            raise ValueError(
                f"predictor_embed_dim={self.predictor_embed_dim} must be divisible by "
                f"num_heads={self.num_heads} (got {self.predictor_embed_dim}/{self.num_heads}"
                f"={self.predictor_embed_dim/self.num_heads:.1f})")
        if self.encoder_depth < 1:
            raise ValueError(f"encoder_depth must be >= 1, got {self.encoder_depth}")
        if self.predictor_depth < 1:
            raise ValueError(f"predictor_depth must be >= 1, got {self.predictor_depth}")
        return True


class TextSpanJEPA(nn.Module):
    """Text-Span JEPA: Latent Predictive Learning for Language Representations."""

    def __init__(self, config: TextSpanJEPAConfig):
        super().__init__()
        self.config = config

        self.encoder = TextSpanJEPLEncoder(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
        )

        # Target encoder: EMA copy, no gradients (I-JEPA pattern)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = TextSpanJEPApredictor(
            embed_dim=config.embed_dim,
            predictor_embed_dim=config.predictor_embed_dim,
            depth=config.predictor_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            max_seq_len=config.max_seq_len,
            future_offsets=config.future_offsets,
            num_refine_steps=config.num_refine_steps,
            refine_step_size=config.refine_step_size,
        )

        self.decoder = TiedTokenDecoder(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            bias=config.decoder_bias,
        )

        # Collapse prevention (VICReg + data2vec centering)
        self.variance_reg = VarianceRegularization(margin=config.variance_margin)
        self.covariance_reg = CovarianceRegularization()
        self.target_centering = TargetCentering(
            dim=config.embed_dim, momentum=config.centering_momentum
        )
        self.diagnostics = CollapseDiagnostics()

    @torch.no_grad()
    def update_target_encoder(self, tau):
        """EMA update of target encoder — I-JEPA pattern:
        param_k = tau * param_k + (1 - tau) * param_q
        """
        for param_q, param_k in zip(self.encoder.parameters(),
                                     self.target_encoder.parameters()):
            param_k.data.mul_(tau).add_((1. - tau) * param_q.detach().data)

    @torch.no_grad()
    def snapshot_target_encoder(self):
        """Save a snapshot of current target encoder output for stability metric.

        Called before EMA update so we can measure how much the target moved.
        Returns the target hidden states for the last batch processed.
        """
        if hasattr(self, '_prev_target_h') and self._prev_target_h is not None:
            return self._prev_target_h
        return None

    def _future_loss_weight(self, current_step):
        """Fix #2: Warmup future loss from 0 to lambda_future.

        Without warmup, the target encoder is unstable early.
        data2vec annealing pattern: linear warmup from 0.
        """
        if self.config.future_warmup_steps <= 0:
            return self.config.lambda_future
        if current_step >= self.config.future_warmup_steps:
            return self.config.lambda_future
        progress = current_step / self.config.future_warmup_steps
        return self.config.lambda_future * progress

    def compute_loss_with_targets(self, masked_input_ids, original_input_ids,
                                   mask_positions, current_step=0, total_steps=1):
        """Compute all loss components.

        I-JEPA pattern: separate masked/original inputs, target encoder
        on original with no_grad, layer_norm on targets.
        data2vec pattern: centering on targets before loss.
        Fix #3: torch.gather with valid_mask for correct loss computation.
        """
        # === Online encoder: masked input ===
        h_online, token_embeds_online = self.encoder(masked_input_ids)

        # === Target encoder: original input (no gradient) ===
        # I-JEPA: layer_norm(h, (h.size(-1),))  on target
        # data2vec: centering before layer_norm
        with torch.no_grad():
            # Snapshot before computing new target (for stability metric)
            self._prev_target_h = getattr(self, '_prev_target_h', None)
            h_target, _ = self.target_encoder(original_input_ids)
            h_target = self.target_centering(h_target)
            h_target = F.layer_norm(h_target, (h_target.size(-1),))

        # === Predictor: span + future predictions ===
        span_preds, num_masked, valid_mask, future_losses, future_preds = self.predictor(
            h_online, mask_positions, token_embeds_online, h_target.detach()
        )

        # === Span loss — only on valid (non-padded) masked positions ===
        # Fix #3: reuse valid_mask from predictor's _gather_masked for alignment
        B, D = h_online.size(0), h_online.size(-1)
        if valid_mask.any():
            # Gather target at same masked positions as predictions
            target_gathered, _, target_valid = TextSpanJEPApredictor._gather_masked(
                h_target.detach(), mask_positions
            )
            # Both valid_masks should be identical (same mask_positions),
            # but use min_cols as safety guard for shape alignment
            min_cols = min(valid_mask.size(1), target_valid.size(1))
            # Combine valid masks: position must be valid in BOTH
            combined_valid = valid_mask[:, :min_cols] & target_valid[:, :min_cols]
            if combined_valid.any():
                loss_span = F.smooth_l1_loss(
                    span_preds[:, :min_cols][combined_valid],
                    target_gathered[:, :min_cols][combined_valid]
                )
            else:
                loss_span = torch.tensor(0.0, device=h_online.device)
        else:
            loss_span = torch.tensor(0.0, device=h_online.device)

        # === Future loss with warmup (Fix #2) ===
        loss_future = sum(future_losses.values()) / max(len(future_losses), 1)
        if isinstance(loss_future, (int, float)):
            loss_future = torch.tensor(float(loss_future), device=h_online.device)
        future_weight = self._future_loss_weight(current_step)

        # === Decoder loss (tied, auxiliary grounding) ===
        loss_decoder = torch.tensor(0.0, device=h_online.device)
        decoder_acc = torch.tensor(0.0, device=h_online.device)
        if mask_positions.any():
            # Gather target tokens at masked positions — vectorized via boolean indexing
            target_tokens = original_input_ids[mask_positions.bool()]
            # Gather online hidden states at masked positions
            h_at_masked = h_online[mask_positions.bool()]
            if h_at_masked.size(0) > 0:
                logits = self.decoder(h_at_masked, self.encoder.token_embedding.weight)
                loss_decoder = F.cross_entropy(logits, target_tokens)
                decoder_acc = (logits.argmax(dim=-1) == target_tokens).float().mean()

        # === Collapse prevention losses (VICReg) ===
        loss_variance = self.variance_reg(h_online)
        loss_covariance = self.covariance_reg(h_online)

        # === Total loss ===
        total_loss = (
            self.config.lambda_span * loss_span
            + future_weight * loss_future
            + self.config.lambda_decoder * loss_decoder
            + self.config.lambda_variance * loss_variance
            + self.config.lambda_covariance * loss_covariance
        )

        loss_dict = {
            'loss': total_loss.item(),
            'loss_span': loss_span.item(),
            'loss_future': loss_future.item(),
            'loss_decoder': loss_decoder.item(),
            'loss_variance': loss_variance.item(),
            'loss_covariance': loss_covariance.item(),
            'decoder_accuracy': decoder_acc.item(),
            'future_weight': future_weight,
        }
        for d, l in future_losses.items():
            loss_dict[f'loss_future_d{d}'] = l.item()

        diag_dict = self.diagnostics.compute(h_online.detach(), h_target.detach(),
                                              prev_target_h=self._prev_target_h)
        diag_dict['target_center_norm'] = self.target_centering.center.norm().item()
        diag_dict['mask_fraction'] = mask_positions.float().mean().item()

        # Update prev_target_h snapshot for next step's stability metric
        self._prev_target_h = h_target.detach().clone()

        return total_loss, loss_dict, diag_dict

    def get_num_params(self, non_embedding=True):
        enc = self.encoder.get_num_params(non_embedding)
        pred = self.predictor.get_num_params()
        dec = sum(p.numel() for p in self.decoder.parameters())
        return enc + pred + dec

    def get_num_params_trainable(self):
        """Count only trainable parameters (excludes EMA target encoder)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
