# Copyright (c) Text-Span JEPA Authors
# Evaluation: linear probes, future-token probes, geometry metrics
# Following NextLat (Microsoft, 2025) / I-JEPA evaluation protocols

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LinearProbe:
    """Linear probe: train a linear classifier on frozen representations."""

    def __init__(self, embed_dim=768, num_classes=2, lr=1e-3, max_epochs=100):
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.lr = lr
        self.max_epochs = max_epochs

    def evaluate(self, model, dataset, device='cuda'):
        model.eval()
        self.classifier = self.classifier.to(device)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(self.max_epochs):
            for batch in loader:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                with torch.no_grad():
                    h, _ = model.encoder(input_ids)
                    h_pooled = h.mean(dim=1)
                logits = self.classifier(h_pooled)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return {'accuracy': 0.0}


class FutureTokenProbe:
    """Future-token probe from NextLat: predictive information in representations."""

    def __init__(self, embed_dim=768, vocab_size=50304, offsets=(1, 4, 16)):
        self.vocab_size = vocab_size
        self.offsets = offsets
        self.probes = nn.ModuleDict({
            f'offset_{d}': nn.Linear(embed_dim, vocab_size)
            for d in offsets
        })

    def evaluate(self, model, dataset, device='cuda', max_steps=5000):
        model.eval()
        self.probes = self.probes.to(device)
        results = {}

        for d in self.offsets:
            probe = self.probes[f'offset_{d}']
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            total_correct = 0
            total_samples = 0

            for step, batch in enumerate(loader):
                if step >= max_steps:
                    break
                input_ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
                B, T = input_ids.shape
                if T <= d:
                    continue
                with torch.no_grad():
                    h, _ = model.encoder(input_ids)
                h_src = h[:, :T-d, :].reshape(-1, h.size(-1))
                target = input_ids[:, d:].reshape(-1)
                logits = probe(h_src)
                loss = F.cross_entropy(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_correct += (logits.argmax(dim=-1) == target).sum().item()
                total_samples += target.size(0)

            results[f'future_probe_d{d}'] = total_correct / max(total_samples, 1)
        return results


class GeometryMetrics:
    """Representation geometry metrics from I-JEPA / NextLat / C-JEPA."""

    @staticmethod
    @torch.no_grad()
    def compute(representations):
        if representations.dim() == 3:
            B, T, D = representations.shape
            representations = representations.reshape(B * T, D)
        N, D = representations.shape
        metrics = {}

        try:
            S = torch.linalg.svdvals(representations)
            S_norm = S / S.sum()
            S_norm = torch.clamp(S_norm, min=1e-12)
            metrics['effective_rank'] = -torch.sum(S_norm * torch.log(S_norm)).exp().item()
            metrics['participation_ratio'] = (S.sum() ** 2 / (S ** 2).sum()).item()
            metrics['condition_number'] = (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
            metrics['numerical_rank'] = torch.linalg.matrix_rank(
                representations, atol=1e-3, rtol=1e-3
            ).item()
            metrics['rank_utilization'] = metrics['numerical_rank'] / min(N, D)

            centered = representations - representations.mean(dim=0)
            cov = (centered.T @ centered) / (N - 1)
            diag = torch.diag(torch.diag(cov))
            off_diag = cov - diag
            metrics['coherence'] = off_diag.abs().max().item() if D > 1 else 0.0
        except Exception as e:
            metrics['error'] = str(e)

        return metrics
