# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# MLM baseline: BERT-style masked language model, same encoder, fair comparison

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import TextSpanJEPLEncoder


class MLMBaseline(nn.Module):
    """BERT-style MLM baseline using the same encoder architecture.

    Fair comparison: identical model capacity, identical compute,
    only the training objective differs.
    """

    def __init__(self, vocab_size=50304, max_seq_len=512, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.1,
                 **kwargs):
        super().__init__()
        self.encoder = TextSpanJEPLEncoder(
            vocab_size=vocab_size, max_seq_len=max_seq_len,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate,
        )
        self.mlm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Decoder attribute for train.py compatibility (same as mlm_head)
        self.decoder = self.mlm_head

    def forward(self, input_ids):
        h, _ = self.encoder(input_ids)
        logits = self.mlm_head(h)
        return logits

    def compute_loss(self, masked_input_ids, original_input_ids, mask_positions):
        logits = self.forward(masked_input_ids)
        masked_logits = logits[mask_positions.bool()]
        masked_targets = original_input_ids[mask_positions.bool()]
        loss = F.cross_entropy(masked_logits, masked_targets)
        accuracy = (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
        return loss, {'loss_mlm': loss.item(), 'mlm_accuracy': accuracy.item()}

    def get_num_params(self, non_embedding=True):
        enc = self.encoder.get_num_params(non_embedding)
        head = sum(p.numel() for p in self.mlm_head.parameters())
        return enc + head
