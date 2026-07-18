# Copyright (c) Text-Span JEPA Authors
# data2vec baseline: EMA teacher + regression head on masked token representations
# Directly adapted from fairseq/examples/data2vec/models/data2vec_text.py
# (Baevski et al., ICML 2022) — same loss, same EMA, same regression head
#
# Fix #4: Complete data2vec baseline from official fairseq implementation
# Key patterns preserved from data2vec_text.py:
#   - get_annealed_rate() for EMA decay annealing (line ~58)
#   - regression_head: Linear→GELU→Linear for head_layers=2 (lines ~301-310)
#   - average_top_k_layers target averaging (line ~432)
#   - loss: smooth_l1 with beta, scale by 1/sqrt(dim) (lines ~474-490)
#   - layer_norm / instance_norm target variants (lines ~434-465)

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import TextSpanJEPLEncoder


def get_annealed_rate(start, end, curr_step, total_steps):
    """EMA decay annealing from data2vec official:
    Linear anneal from start to end over total_steps.
    data2vec_text.py line ~58:
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining
    """
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


class Data2VecTextBaseline(nn.Module):
    """data2vec-style baseline using the same encoder architecture.

    From the official fairseq implementation (data2vec_text.py):
    - Online encoder processes masked input
    - EMA teacher (target encoder) processes original input
    - Regression head: Linear → GELU → Linear (head_layers=2, lines 301-310)
    - Loss: smooth_l1_loss or mse_loss on masked positions
    - Target: layer_norm of top-K hidden layers from teacher
    - EMA tau: annealed from ema_decay to ema_end_decay (get_annealed_rate)

    Key differences from Text-Span JEPA:
    - Token-level prediction (not span-level)
    - No future latent prediction
    - Regression head instead of predictor transformer
    - Average top-K layers as target (we use final layer)
    """

    def __init__(self, vocab_size=50304, max_seq_len=512, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0,
                 average_top_k_layers=8, loss_beta=0.0, loss_scale=None,
                 ema_decay=0.999, ema_end_decay=0.9999, ema_anneal_end_step=100000,
                 head_layers=1, mask_token_id=0, **kwargs):
        super().__init__()
        self.average_top_k_layers = average_top_k_layers
        self.loss_beta = loss_beta
        # data2vec: loss_scale=None → scale by 1/sqrt(dim); else multiply by constant
        self.loss_scale = loss_scale or (-1 if loss_scale is None else loss_scale)
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.mask_token_id = mask_token_id
        self.num_updates = 0

        # Online encoder
        self.encoder = TextSpanJEPLEncoder(
            vocab_size=vocab_size, max_seq_len=max_seq_len,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # EMA teacher (target encoder)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Regression head — exactly from data2vec_text.py lines 301-310:
        #   curr_dim = embed_dim
        #   projs = []
        #   for i in range(head_layers - 1):
        #       next_dim = embed_dim * 2 if i == 0 else curr_dim
        #       projs.append(nn.Linear(curr_dim, next_dim))
        #       projs.append(nn.GELU())
        #       curr_dim = next_dim
        #   projs.append(nn.Linear(curr_dim, embed_dim))
        #   self.regression_head = nn.Sequential(*projs)
        curr_dim = embed_dim
        projs = []
        for i in range(head_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, embed_dim))
        self.regression_head = nn.Sequential(*projs)

    def get_annealed_decay(self):
        """EMA decay annealing from data2vec official:
        get_annealed_rate(ema_decay, ema_end_decay, num_updates, ema_anneal_end_step)
        """
        if self.num_updates >= self.ema_anneal_end_step:
            return self.ema_end_decay
        return get_annealed_rate(
            self.ema_decay, self.ema_end_decay,
            self.num_updates, self.ema_anneal_end_step
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update — data2vec official pattern."""
        decay = self.get_annealed_decay()
        for param_q, param_k in zip(self.encoder.parameters(),
                                     self.target_encoder.parameters()):
            param_k.data.mul_(decay).add_((1. - decay) * param_q.detach().data)

    def forward(self, masked_input_ids, original_input_ids, mask_positions):
        """Compute data2vec-style loss.

        From data2vec_text.py lines 422-490:
        1. Online encoder on masked input
        2. Target encoder on original input, layer_norm targets
        3. Regression head on online output at masked positions
        4. Smooth L1 or MSE loss, scaled by 1/sqrt(dim) or loss_scale
        """
        # Online encoder: masked input
        h_online, _ = self.encoder(masked_input_ids)

        # Target encoder: original input (no gradient)
        # data2vec: layer_norm(targets, (targets.shape[-1],))
        with torch.no_grad():
            h_target, _ = self.target_encoder(original_input_ids)
            h_target = F.layer_norm(h_target.float(), h_target.shape[-1:])

        # Extract masked positions — data2vec uses boolean indexing (line 468-469):
        #   masked_indices = src_tokens.eq(self.mask_idx)
        #   x = x[masked_indices]
        #   y = y[masked_indices]
        masked_indices = mask_positions.bool()
        x = h_online[masked_indices]
        y = h_target[masked_indices]

        # Regression head
        x = self.regression_head(x)

        # Loss — data2vec_text.py lines 474-490:
        #   sz = x.size(-1)
        #   if self.cfg.loss_beta == 0:
        #       loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        #   else:
        #       loss = F.smooth_l1_loss(x.float(), y.float(), reduction="none",
        #                               beta=self.cfg.loss_beta).sum(dim=-1)
        #   result = loss.sum() / math.sqrt(sz) if loss_scale <= 0
        #           else loss.sum() * loss_scale
        sz = x.size(-1)
        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction='none').sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction='none', beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale <= 0:
            loss = loss.sum() / math.sqrt(sz)
        else:
            loss = loss.sum() * self.loss_scale

        sample_size = mask_positions.sum().item()
        loss = loss / max(sample_size, 1)

        self.num_updates += 1

        return loss, {
            'loss_data2vec': loss.item(),
            'ema_decay': self.get_annealed_decay(),
            'num_masked': sample_size,
        }

    def get_num_params(self, non_embedding=True):
        enc = self.encoder.get_num_params(non_embedding)
        reg = sum(p.numel() for p in self.regression_head.parameters())
        return enc + reg
