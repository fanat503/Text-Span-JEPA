"""Minimal masked language modeling baseline sharing JEPA encoder blocks.

Use this as a reviewer-facing baseline: same tokenizer/data/mask policy, but
predicts tokens directly instead of predicting EMA latent targets.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TextJEPAConfig, TokenEncoder

class MaskedLMBaseline(nn.Module):
    def __init__(self,cfg:TextJEPAConfig):
        super().__init__(); self.cfg=cfg; self.encoder=TokenEncoder(cfg); self.lm_head=nn.Linear(cfg.n_embd,cfg.padded_vocab_size,bias=False); self.lm_head.weight=self.encoder.wte.weight
    def forward(self,ids:torch.Tensor,mask:torch.Tensor):
        masked=torch.where(mask, torch.full_like(ids,int(self.cfg.mask_token_id)), ids)
        h=self.encoder(masked); logits=self.lm_head(h).float(); logits[...,self.cfg.vocab_size:]=torch.finfo(logits.dtype).min
        loss=F.cross_entropy(logits[mask], ids[mask])
        acc=(logits[mask].argmax(-1)==ids[mask]).float().mean()
        return {'loss':loss,'accuracy':acc}
