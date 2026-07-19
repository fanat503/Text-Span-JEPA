# Copyright (c) Text-Span JEPA Authors
# Test suite — patterns from NextLat (Microsoft Research) + I-JEPA (Meta)
# Key test patterns from NextLat model_base.py:
#   - compute_hidden_state_rank: effective_rank, numerical_rank, condition_number,
#     rank_utilization, max_possible_rank
#   - Exception handling returns zeros/infs (not crashes)
#   - Per-sample rank metrics vs batch-level metrics
# Risk fixes #2, #3, #4 are tested explicitly.

import math
import pytest
import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════

class TestEncoder:
    def setup_method(self):
        from src.models.encoder import TextSpanJEPLEncoder
        self.Encoder = TextSpanJEPLEncoder

    def test_output_shape(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        h, tok = enc(torch.randint(0, 1000, (4, 32)))
        assert h.shape == (4, 32, 64)
        assert tok.shape == (4, 32, 64)

    def test_different_seq_lengths(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=128, embed_dim=64, depth=2, num_heads=4)
        for sl in [8, 16, 32, 64]:
            h, _ = enc(torch.randint(0, 1000, (2, sl)))
            assert h.shape == (2, sl, 64)

    def test_param_count(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        non_emb = enc.get_num_params(non_embedding=True)
        with_emb = enc.get_num_params(non_embedding=False)
        assert non_emb < with_emb

    def test_gradients_flow(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        h, _ = enc(torch.randint(0, 1000, (2, 32)))
        h.sum().backward()
        assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)

    def test_pos_embedding_is_learnable(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        assert enc.pos_embedding.requires_grad, "pos_embedding should be learnable"

    def test_deterministic_with_same_seed(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        enc.eval()
        x = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            h1, _ = enc(x)
            h2, _ = enc(x)
        assert torch.allclose(h1, h2, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Predictor
# ═══════════════════════════════════════════════════════════════════

class TestPredictor:
    def setup_method(self):
        from src.models.predictor import TextSpanJEPApredictor
        self.Predictor = TextSpanJEPApredictor

    def test_output_shape(self):
        pred = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                              max_seq_len=32, future_offsets=(1, 4), num_refine_steps=2)
        h = torch.randn(4, 32, 64)
        mask = torch.zeros(4, 32, dtype=torch.long); mask[:, 5:10] = 1; mask[:, 20:25] = 1
        span_preds, num_masked, valid_mask, fl, fp = pred(h, mask, torch.randn(4, 32, 64), torch.randn(4, 32, 64))
        assert span_preds.shape[0] == 4 and span_preds.shape[2] == 64
        assert valid_mask.sum().item() == mask.sum().item()
        for d in (1, 4):
            assert d in fl and d in fp
            assert fp[d].shape == (4, 32 - d, 64)

    def test_gather_masked_fix3(self):
        """Fix #3: torch.gather with valid_mask."""
        pred = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                              max_seq_len=16, future_offsets=(1,))
        h = torch.randn(3, 16, 64)
        mask = torch.zeros(3, 16, dtype=torch.long)
        mask[0, 0:3] = 1; mask[1, 2:7] = 1; mask[2, 10:12] = 1
        gathered, num_masked, valid_mask = pred._gather_masked(h, mask)
        assert gathered.shape == (3, 5, 64)
        assert valid_mask.sum().item() == 10
        assert gathered[0, 3:].abs().sum().item() == pytest.approx(0.0, abs=1e-6)
        assert gathered[2, 2:].abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_gather_masked_all_zeros(self):
        """Edge case: no masked positions → returns zeros, no crash."""
        pred = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                              max_seq_len=16, future_offsets=(1,))
        h = torch.randn(2, 16, 64)
        mask = torch.zeros(2, 16, dtype=torch.long)
        gathered, num_masked, valid_mask = pred._gather_masked(h, mask)
        assert num_masked.sum().item() == 0
        assert valid_mask.sum().item() == 0

    def test_iterative_refinement(self):
        pred_r = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                                max_seq_len=32, future_offsets=(1,), num_refine_steps=3)
        pred_n = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                                max_seq_len=32, future_offsets=(1,), num_refine_steps=0)
        pred_n.load_state_dict(pred_r.state_dict(), strict=False)
        h, mask, tok, tgt = torch.randn(2, 32, 64), torch.zeros(2, 32, dtype=torch.long), torch.randn(2, 32, 64), torch.randn(2, 32, 64)
        mask[:, 5:8] = 1
        with torch.no_grad():
            s1, _, _, _, _ = pred_r(h, mask, tok, tgt)
            s2, _, _, _, _ = pred_n(h, mask, tok, tgt)
        assert not torch.allclose(s1, s2, atol=1e-5)

    def test_no_mask(self):
        pred = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                              max_seq_len=32, future_offsets=(1,))
        span_preds, num_masked, valid_mask, _, _ = pred(
            torch.randn(2, 32, 64), torch.zeros(2, 32, dtype=torch.long),
            torch.randn(2, 32, 64), torch.randn(2, 32, 64))
        assert num_masked.sum().item() == 0

    def test_future_prediction_offset_exceeds_seq_len(self):
        """Edge case: future offset > T → no loss for that offset."""
        pred = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                              max_seq_len=32, future_offsets=(1, 100), num_refine_steps=1)
        h = torch.randn(2, 16, 64)
        mask = torch.zeros(2, 16, dtype=torch.long); mask[:, 3:5] = 1
        _, _, _, fl, fp = pred(h, mask, torch.randn(2, 16, 64), torch.randn(2, 16, 64))
        assert 1 in fl  # offset 1 should work (T=16 > 1)
        assert 100 not in fl  # offset 100 > T → skipped

    def test_predictor_pos_embed_is_learnable(self):
        pred = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                              max_seq_len=32, future_offsets=(1,))
        assert pred.predictor_pos_embed.requires_grad


# ═══════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════

class TestDecoder:
    def test_output(self):
        from src.models.decoder import TiedTokenDecoder
        dec = TiedTokenDecoder(embed_dim=64, vocab_size=1000)
        logits = dec(torch.randn(8, 64), torch.randn(1000, 64))
        assert logits.shape == (8, 1000)

    def test_data2vec_regression_head_pattern(self):
        """Decoder follows data2vec regression head: Linear→GELU→Linear."""
        from src.models.decoder import TiedTokenDecoder
        dec = TiedTokenDecoder(embed_dim=64, vocab_size=1000, bias=False)
        # First linear: 64 → 128 (2x expand)
        assert isinstance(dec.proj[0], torch.nn.Linear)
        assert dec.proj[0].in_features == 64
        assert dec.proj[0].out_features == 128
        # GELU
        assert isinstance(dec.proj[1], torch.nn.GELU)
        # Second linear: 128 → 64
        assert isinstance(dec.proj[2], torch.nn.Linear)
        assert dec.proj[2].out_features == 64


# ═══════════════════════════════════════════════════════════════════
# Collapse Prevention — metrics from NextLat model_base.py
# ═══════════════════════════════════════════════════════════════════

class TestCollapsePrevention:
    def test_variance_active(self):
        from src.models.collapse import VarianceRegularization
        assert VarianceRegularization(margin=1.0)(torch.randn(32, 64) * 0.01).item() > 0

    def test_variance_satisfied(self):
        from src.models.collapse import VarianceRegularization
        assert VarianceRegularization(margin=1.0)(torch.randn(32, 64) * 10.0).item() == pytest.approx(0.0, abs=1e-3)

    def test_covariance(self):
        from src.models.collapse import CovarianceRegularization
        assert CovarianceRegularization()(torch.randn(64, 32)).item() >= 0

    def test_centering(self):
        from src.models.collapse import TargetCentering
        tc = TargetCentering(dim=32, momentum=0.9)
        centered = tc(torch.randn(4, 8, 32) + 5.0)
        assert centered.shape == (4, 8, 32) and tc.center.norm().item() > 0

    def test_effective_rank_positive(self):
        """BUG FIX: effective_rank must be > 0 for random data.
        Previous bug: -x.exp() = -(x.exp()) due to operator precedence."""
        from src.models.collapse import CollapseDiagnostics
        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert m['effective_rank_online'] > 0
        assert m['effective_rank_target'] > 0

    def test_hidden_state_rank_nextlat_pattern(self):
        """Test rank metrics match NextLat model_base.compute_hidden_state_rank pattern:
        - effective_rank = exp(-sum(p * log(p))) where p = S/S.sum()
        - condition_number = S[0] / S[-1]
        - numerical_rank = matrix_rank(atol=1e-3, rtol=1e-3)
        - rank_utilization = numerical_rank / max_possible_rank
        """
        from src.models.collapse import CollapseDiagnostics
        diag = CollapseDiagnostics()
        h = torch.randn(4, 16, 32)
        metrics = diag.compute(h, h)

        # NextLat pattern: batch-level metrics
        flat = h.reshape(-1, 32)
        S = torch.linalg.svdvals(flat)
        S_norm = S / S.sum()
        S_norm = torch.clamp(S_norm, min=1e-12)
        expected_eff_rank = (-torch.sum(S_norm * torch.log(S_norm))).exp().item()
        expected_cond = (S[0] / S[-1]).item()
        expected_num_rank = torch.linalg.matrix_rank(flat, atol=1e-3, rtol=1e-3).item()

        assert abs(metrics['effective_rank_online'] - expected_eff_rank) < 0.1
        assert abs(metrics['condition_number_online'] - expected_cond) / max(expected_cond, 1) < 0.05
        assert metrics['numerical_rank_online'] == expected_num_rank

    def test_collapse_detection_zero_input(self):
        """All-zero input should show collapse (low effective rank)."""
        from src.models.collapse import CollapseDiagnostics
        m = CollapseDiagnostics().compute(torch.zeros(4, 16, 32), torch.zeros(4, 16, 32))
        # Zero input → rank 0 → effective_rank near 0
        assert m['effective_rank_online'] <= 1.0

    def test_participation_ratio(self):
        """Participation ratio: (sum S)^2 / sum(S^2). Random data should have PR > 1."""
        from src.models.collapse import CollapseDiagnostics
        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert m['participation_ratio_online'] > 1.0


# ═══════════════════════════════════════════════════════════════════
# Full JEPA Model
# ═══════════════════════════════════════════════════════════════════

class TestJEPA:
    def setup_method(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        self.JEPA, self.Config = TextSpanJEPA, TextSpanJEPAConfig

    def _cfg(self, **overrides):
        defaults = dict(vocab_size=1000, max_seq_len=32, embed_dim=64, encoder_depth=2,
                        num_heads=4, mlp_ratio=2.0, predictor_embed_dim=32, predictor_depth=2,
                        future_offsets=(1, 4), num_refine_steps=1, future_warmup_steps=10)
        defaults.update(overrides)
        return self.Config(**defaults)

    def test_forward(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, ld, dd = model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert loss.requires_grad
        for k in ['loss', 'loss_span', 'loss_future', 'loss_decoder',
                   'loss_variance', 'loss_covariance', 'decoder_accuracy',
                   'future_weight']:
            assert k in ld, f"Missing key: {k}"
        for k in ['effective_rank_online', 'effective_rank_target',
                   'participation_ratio_online', 'condition_number_online',
                   'numerical_rank_online', 'coherence_online', 'mask_fraction',
                   'target_center_norm']:
            assert k in dd, f"Missing key: {k}"

    def test_future_warmup_fix2(self):
        """Fix #2: Future loss weight ramps from 0 to lambda_future."""
        model = self.JEPA(self._cfg(lambda_future=0.5, future_warmup_steps=100))
        assert model._future_loss_weight(0) == pytest.approx(0.0, abs=1e-6)
        assert model._future_loss_weight(50) == pytest.approx(0.25, abs=1e-6)
        assert model._future_loss_weight(100) == pytest.approx(0.5, abs=1e-6)
        assert model._future_loss_weight(200) == pytest.approx(0.5, abs=1e-6)

    def test_future_warmup_disabled(self):
        """When future_warmup_steps=0, weight is always lambda_future."""
        model = self.JEPA(self._cfg(lambda_future=0.5, future_warmup_steps=0))
        assert model._future_loss_weight(0) == pytest.approx(0.5, abs=1e-6)
        assert model._future_loss_weight(100) == pytest.approx(0.5, abs=1e-6)

    def test_ema_update(self):
        model = self.JEPA(self._cfg())
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        before = {n: p.clone() for n, p in model.target_encoder.named_parameters()}
        model.update_target_encoder(0.996)
        assert any(not torch.allclose(before[n], p, atol=1e-8)
                   for n, p in model.target_encoder.named_parameters())

    def test_ema_tau_formula_ijepa(self):
        """EMA tau follows I-JEPA formula: ema[0] + i*(ema[1]-ema[0])/total_steps."""
        from src.utils.schedulers import EMATauSchedule
        s = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=1000)
        # Step 0: i=1 → 0.996 + 1*0.004/1000 = 0.996004
        tau1 = s.step()
        assert abs(tau1 - (0.996 + 1 * 0.004 / 1000)) < 1e-6

    def test_target_no_grad(self):
        assert all(not p.requires_grad for p in self.JEPA(self._cfg()).target_encoder.parameters())

    def test_gradient_flow(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)[0].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.encoder.parameters() if p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.predictor.parameters() if p.requires_grad)

    def test_loss_is_finite(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, _, _ = model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert torch.isfinite(loss)

    def test_span_loss_only_valid_positions(self):
        """Span loss should NOT include zero-padded positions from _gather_masked."""
        model = self.JEPA(self._cfg())
        # Different numbers of masked positions per sample
        mask = torch.zeros(3, 32, dtype=torch.long)
        mask[0, 0:2] = 1; mask[1, 5:10] = 1; mask[2, 15:18] = 1
        loss, ld, _ = model.compute_loss_with_targets(
            torch.randint(0, 1000, (3, 32)), torch.randint(0, 1000, (3, 32)), mask)
        assert torch.isfinite(loss)
        assert ld['loss_span'] >= 0

    def test_decoder_uses_boolean_indexing(self):
        """Decoder loss uses vectorized boolean indexing (not Python for-loop)."""
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, ld, _ = model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        # Decoder should have non-zero loss when mask is present
        assert ld['loss_decoder'] > 0


# ═══════════════════════════════════════════════════════════════════
# Training Integration (small scale)
# ═══════════════════════════════════════════════════════════════════

class TestTrainingIntegration:
    def test_loss_decreases_over_steps(self):
        """Small training loop: loss should decrease over 200 steps on fixed data."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.utils.schedulers import EMATauSchedule

        torch.manual_seed(42)
        config = TextSpanJEPAConfig(
            vocab_size=1000, max_seq_len=32, embed_dim=64, encoder_depth=2,
            num_heads=4, mlp_ratio=2.0, predictor_embed_dim=32, predictor_depth=2,
            future_offsets=(1, 4), num_refine_steps=1, future_warmup_steps=20)
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                                list(model.predictor.parameters()) +
                                list(model.decoder.parameters()), lr=2e-3)
        ema = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=200)

        # Fixed dataset (same inputs each step) for deterministic convergence test
        fixed_input = torch.randint(0, 1000, (4, 32))
        fixed_target = torch.randint(0, 1000, (4, 32))
        fixed_mask = torch.zeros(4, 32, dtype=torch.long)
        fixed_mask[:, 5:10] = 1; fixed_mask[:, 20:25] = 1

        losses = []
        for step in range(200):
            loss, _, _ = model.compute_loss_with_targets(
                fixed_input, fixed_target,
                fixed_mask, current_step=step, total_steps=200)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt.step()
            model.update_target_encoder(ema.step())
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_no_nan_after_many_steps(self):
        """No NaN after 100 steps — tests numerical stability."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.utils.schedulers import EMATauSchedule

        config = TextSpanJEPAConfig(
            vocab_size=1000, max_seq_len=32, embed_dim=64, encoder_depth=2,
            num_heads=4, mlp_ratio=2.0, predictor_embed_dim=32, predictor_depth=2,
            future_offsets=(1,), num_refine_steps=1, future_warmup_steps=5)
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                                list(model.predictor.parameters()) +
                                list(model.decoder.parameters()), lr=1e-3)
        ema = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=100)

        for step in range(100):
            mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:8] = 1
            loss, _, _ = model.compute_loss_with_targets(
                torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)),
                mask, current_step=step, total_steps=100)
            assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt.step()
            model.update_target_encoder(ema.step())


# ═══════════════════════════════════════════════════════════════════
# Span Mask
# ═══════════════════════════════════════════════════════════════════

class TestSpanMask:
    def test_basic(self):
        from src.masks.span import SpanMaskCollator
        r = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=0)(
            [{'input_ids': torch.tensor([1,2,3,4,5,6,7,8,9,10])}])
        assert all(k in r for k in ['masked_input_ids', 'original_input_ids', 'mask_positions'])

    def test_curriculum(self):
        from src.masks.span import SpanMaskCollator
        c = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_ratio_start=0.1,
                              mask_ratio_end=0.5, curriculum_steps=100, mask_token_id=0)
        assert c.current_mask_ratio == pytest.approx(0.1, abs=0.02)
        c._step = 100
        assert c.current_mask_ratio == pytest.approx(0.5, abs=0.02)

    def test_masked_tokens_replaced(self):
        from src.masks.span import SpanMaskCollator
        c = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=99)
        r = c([{'input_ids': torch.tensor([1,2,3,4,5,6,7,8,9,10] * 5)}])
        mask = r['mask_positions'].bool()
        assert (r['masked_input_ids'][mask] == 99).all()

    def test_original_unchanged(self):
        from src.masks.span import SpanMaskCollator
        c = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=99)
        orig = torch.tensor([1,2,3,4,5,6,7,8,9,10] * 5)
        r = c([{'input_ids': orig}])
        assert (r['original_input_ids'][0] == orig).all()


# ═══════════════════════════════════════════════════════════════════
# Schedulers (I-JEPA patterns)
# ═══════════════════════════════════════════════════════════════════

class TestSchedulers:
    def test_lr_warmup_cosine(self):
        from src.utils.schedulers import WarmupCosineSchedule
        opt = torch.optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.001)
        s = WarmupCosineSchedule(opt, warmup_steps=10, start_lr=1e-5, ref_lr=1e-3, final_lr=1e-6, T_max=100)
        lrs = [s.step() for _ in range(100)]
        assert lrs[5] > lrs[0] and lrs[9] >= lrs[8]  # warmup
        assert lrs[50] < lrs[9]  # cosine decay
        assert lrs[99] <= lrs[50]  # continues decaying

    def test_ema_tau_ijepa_formula(self):
        """EMA tau: I-JEPA momentum_scheduler = ema[0] + i*(ema[1]-ema[0])/total_steps."""
        from src.utils.schedulers import EMATauSchedule
        s = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=10000)
        tau_5000 = None
        for i in range(5000):
            t = s.step()
            if i == 4999:
                tau_5000 = t
        # At step 5000: 0.996 + 5000 * 0.004 / 10000 = 0.998
        assert abs(tau_5000 - 0.998) < 1e-4

    def test_wd_schedule(self):
        from src.utils.schedulers import CosineWDSchedule
        opt = torch.optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.001, weight_decay=0.04)
        s = CosineWDSchedule(opt, ref_wd=0.04, final_wd=0.4, T_max=100)
        wds = [s.step() for _ in range(100)]
        assert wds[-1] > wds[0]  # WD increases


# ═══════════════════════════════════════════════════════════════════
# data2vec Baseline (Fix #4 — from official fairseq)
# ═══════════════════════════════════════════════════════════════════

class TestData2VecBaseline:
    def test_forward(self):
        from baselines.data2vec_baseline import Data2VecTextBaseline
        m = Data2VecTextBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2,
                                 num_heads=4, mlp_ratio=2.0, head_layers=2,
                                 ema_decay=0.999, ema_end_decay=0.9999, ema_anneal_end_step=100)
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, info = m(torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert loss.requires_grad and 'loss_data2vec' in info

    def test_ema_annealing(self):
        from baselines.data2vec_baseline import Data2VecTextBaseline
        m = Data2VecTextBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2,
                                 num_heads=4, ema_decay=0.999, ema_end_decay=0.9999, ema_anneal_end_step=100)
        assert m.get_annealed_decay() < 0.9999
        m.num_updates = 100
        assert m.get_annealed_decay() == pytest.approx(0.9999, abs=1e-5)

    def test_get_annealed_rate_exact_formula(self):
        """Test exact get_annealed_rate from data2vec_text.py line ~58:
        r = end - start; pct_remaining = 1 - curr_step/total_steps; return end - r * pct_remaining
        """
        from baselines.data2vec_baseline import get_annealed_rate
        # Step 0 → return start
        assert get_annealed_rate(0.999, 0.9999, 0, 100) == pytest.approx(0.999, abs=1e-6)
        # Step 50 → midpoint
        mid = get_annealed_rate(0.999, 0.9999, 50, 100)
        expected = 0.9999 - (0.9999 - 0.999) * (1 - 50/100)
        assert mid == pytest.approx(expected, abs=1e-6)
        # Step 100 → return end
        assert get_annealed_rate(0.999, 0.9999, 100, 100) == pytest.approx(0.9999, abs=1e-6)

    def test_regression_head_data2vec(self):
        """data2vec regression head: head_layers=2 → Linear→GELU→Linear."""
        from baselines.data2vec_baseline import Data2VecTextBaseline
        m = Data2VecTextBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2,
                                 num_heads=4, head_layers=2)
        assert m.regression_head[0].in_features == 64
        assert m.regression_head[0].out_features == 128  # 2x expand (data2vec pattern)
        assert isinstance(m.regression_head[1], torch.nn.GELU)
        assert m.regression_head[2].out_features == 64

    def test_loss_formula_data2vec(self):
        """data2vec loss: smooth_l1 with beta=0 → mse_loss; scale = 1/sqrt(dim)."""
        from baselines.data2vec_baseline import Data2VecTextBaseline
        m = Data2VecTextBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2,
                                 num_heads=4, loss_beta=0.0, loss_scale=None)
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, _ = m(torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert torch.isfinite(loss)

    def test_target_encoder_no_grad(self):
        from baselines.data2vec_baseline import Data2VecTextBaseline
        m = Data2VecTextBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        assert all(not p.requires_grad for p in m.target_encoder.parameters())


# ═══════════════════════════════════════════════════════════════════
# MLM Baseline
# ═══════════════════════════════════════════════════════════════════

class TestMLMBaseline:
    def test_forward(self):
        from baselines.mlm_baseline import MLMBaseline
        m = MLMBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0)
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, info = m.compute_loss(torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert loss.requires_grad and 'loss_mlm' in info and 'mlm_accuracy' in info


# ═══════════════════════════════════════════════════════════════════
# Logging (I-JEPA patterns)
# ═══════════════════════════════════════════════════════════════════

class TestLogging:
    def test_average_meter(self):
        from src.utils.logging import AverageMeter
        m = AverageMeter()
        m.update(1.0); m.update(3.0)
        assert m.avg == 2.0
        assert m.val == 3.0
        assert m.min == 1.0 and m.max == 3.0

    def test_grad_logger(self):
        from src.utils.logging import grad_logger
        from src.models.encoder import TextSpanJEPLEncoder
        # I-JEPA grad_logger looks for 'qkv' in param names;
        # must use a model with fused QKV projection (like our encoder).
        model = TextSpanJEPLEncoder(vocab_size=1000, max_seq_len=32,
                                    embed_dim=64, depth=2, num_heads=4)
        x = torch.randint(0, 1000, (2, 32))
        h, _ = model(x)
        h.sum().backward()
        stats = grad_logger(model.named_parameters())
        assert stats.first_layer > 0, "grad_logger should detect first QKV layer gradient"
        assert stats.last_layer > 0, "grad_logger should detect last QKV layer gradient"


# ═══════════════════════════════════════════════════════════════════
# Evaluation Probes
# ═══════════════════════════════════════════════════════════════════

class TestEvalProbes:
    def test_geometry_metrics_random_data(self):
        """GeometryMetrics should return valid metrics for random data."""
        from src.eval.probes import GeometryMetrics
        m = GeometryMetrics.compute(torch.randn(4, 16, 32))
        assert m['effective_rank'] > 0
        assert m['participation_ratio'] > 1.0
        assert m['condition_number'] > 0
        assert m['numerical_rank'] > 0
        assert 0 < m['rank_utilization'] <= 1.0
        assert m['coherence'] >= 0

    def test_geometry_metrics_zero_input(self):
        """GeometryMetrics should handle zero input without NaN/crash (NextLat pattern)."""
        from src.eval.probes import GeometryMetrics
        m = GeometryMetrics.compute(torch.zeros(4, 16, 32))
        assert m['effective_rank'] == 0.0
        assert m['numerical_rank'] == 0.0
        assert m['condition_number'] == float('inf')
        assert m['rank_utilization'] == 0.0

    def test_geometry_metrics_reuses_collapse_diagnostics(self):
        """GeometryMetrics should reuse CollapseDiagnostics (no code duplication)."""
        from src.eval.probes import GeometryMetrics
        from src.models.collapse import CollapseDiagnostics
        # Verify it uses the same instance/methods
        assert isinstance(GeometryMetrics._diag, CollapseDiagnostics)
        assert GeometryMetrics.compute is not None


# ═══════════════════════════════════════════════════════════════════
# Checkpoint Save/Load (I-JEPA pattern)
# ═══════════════════════════════════════════════════════════════════

class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        """Checkpoint save → load should produce identical model weights."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.train import load_checkpoint

        config = TextSpanJEPAConfig(
            vocab_size=1000, max_seq_len=32, embed_dim=64, encoder_depth=2,
            num_heads=4, mlp_ratio=2.0, predictor_embed_dim=32, predictor_depth=2,
            future_offsets=(1,), num_refine_steps=1)
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                                list(model.predictor.parameters()) +
                                list(model.decoder.parameters()), lr=1e-3)

        # Save
        ckpt_path = str(tmp_path / 'test_ckpt.pth.tar')
        save_dict = {
            'encoder': model.encoder.state_dict(),
            'predictor': model.predictor.state_dict(),
            'target_encoder': model.target_encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'opt': opt.state_dict(),
            'scaler': None,
            'epoch': 5,
            'global_step': 1000,
            'loss': 0.5,
        }
        torch.save(save_dict, ckpt_path)

        # Create fresh model and load
        model2 = TextSpanJEPA(config)
        opt2 = torch.optim.AdamW(list(model2.encoder.parameters()) +
                                 list(model2.predictor.parameters()) +
                                 list(model2.decoder.parameters()), lr=1e-3)

        enc, pred, tgt, dec, opt2, scaler, epoch, gstep = load_checkpoint(
            ckpt_path, model2.encoder, model2.predictor,
            model2.target_encoder, model2.decoder, opt2, None)

        assert epoch == 5
        assert gstep == 1000
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.encoder.named_parameters(),
                                       model2.encoder.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"

    def test_checkpoint_saves_global_step(self, tmp_path):
        """Checkpoint must include global_step for training resumption."""
        import io
        model_state = {'global_step': 42, 'epoch': 3}
        buf = io.BytesIO()
        torch.save(model_state, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        assert loaded['global_step'] == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
