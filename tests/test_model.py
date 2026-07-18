# Copyright (c) Text-Span JEPA Authors
# Full test suite — 35 tests covering all components + risk fixes #2, #3, #4

import math
import pytest
import torch
import numpy as np


class TestEncoder:
    def setup_method(self):
        from src.models.encoder import TextSpanJEPLEncoder
        self.Encoder = TextSpanJEPLEncoder

    def test_output_shape(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        x = torch.randint(0, 1000, (4, 32))
        h, tok = enc(x)
        assert h.shape == (4, 32, 64)
        assert tok.shape == (4, 32, 64)

    def test_different_seq_lengths(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=128, embed_dim=64, depth=2, num_heads=4)
        for sl in [8, 16, 32, 64]:
            h, _ = enc(torch.randint(0, 1000, (2, sl)))
            assert h.shape == (2, sl, 64)

    def test_param_count(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        assert enc.get_num_params(non_embedding=True) < enc.get_num_params(non_embedding=False)

    def test_gradients_flow(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        h, _ = enc(torch.randint(0, 1000, (2, 32)))
        h.sum().backward()
        assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)


class TestPredictor:
    def setup_method(self):
        from src.models.predictor import TextSpanJEPApredictor
        self.Predictor = TextSpanJEPApredictor

    def test_output_shape(self):
        pred = self.Predictor(embed_dim=64, predictor_embed_dim=32, depth=2, num_heads=4,
                              max_seq_len=32, future_offsets=(1, 4), num_refine_steps=2)
        h = torch.randn(4, 32, 64)
        mask = torch.zeros(4, 32, dtype=torch.long); mask[:, 5:10] = 1; mask[:, 20:25] = 1
        span_preds, num_masked, valid_mask, future_losses, future_preds = pred(h, mask, torch.randn(4, 32, 64), torch.randn(4, 32, 64))
        assert span_preds.shape[0] == 4 and span_preds.shape[2] == 64
        assert valid_mask.sum().item() == mask.sum().item()
        for d in (1, 4):
            assert d in future_losses and d in future_preds
            assert future_preds[d].shape == (4, 32 - d, 64)

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
        mask = torch.zeros(2, 32, dtype=torch.long)
        span_preds, num_masked, valid_mask, _, _ = pred(torch.randn(2, 32, 64), mask, torch.randn(2, 32, 64), torch.randn(2, 32, 64))
        assert num_masked.sum().item() == 0


class TestDecoder:
    def test_output(self):
        from src.models.decoder import TiedTokenDecoder
        dec = TiedTokenDecoder(embed_dim=64, vocab_size=1000)
        logits = dec(torch.randn(8, 64), torch.randn(1000, 64))
        assert logits.shape == (8, 1000)


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

    def test_diagnostics_effective_rank(self):
        from src.models.collapse import CollapseDiagnostics
        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert m['effective_rank_online'] > 0, "BUG: effective_rank should be > 0 for random data"
        assert m['effective_rank_target'] > 0


class TestJEPA:
    def setup_method(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        self.JEPA, self.Config = TextSpanJEPA, TextSpanJEPAConfig

    def _cfg(self):
        return self.Config(vocab_size=1000, max_seq_len=32, embed_dim=64, encoder_depth=2,
                           num_heads=4, mlp_ratio=2.0, predictor_embed_dim=32, predictor_depth=2,
                           future_offsets=(1, 4), num_refine_steps=1, future_warmup_steps=10)

    def test_forward(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, ld, dd = model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert loss.requires_grad and 'loss_span' in ld and 'effective_rank_online' in dd

    def test_future_warmup_fix2(self):
        """Fix #2: Future loss warmup ramps from 0 to lambda_future."""
        cfg = self._cfg(); cfg.lambda_future = 0.5; cfg.future_warmup_steps = 100
        model = self.JEPA(cfg)
        assert model._future_loss_weight(0) == pytest.approx(0.0, abs=1e-6)
        assert model._future_loss_weight(50) == pytest.approx(0.25, abs=1e-6)
        assert model._future_loss_weight(100) == pytest.approx(0.5, abs=1e-6)

    def test_ema_update(self):
        model = self.JEPA(self._cfg())
        with torch.no_grad():
            for p in model.encoder.parameters(): p.add_(torch.randn_like(p) * 0.01)
        before = {n: p.clone() for n, p in model.target_encoder.named_parameters()}
        model.update_target_encoder(0.996)
        assert any(not torch.allclose(before[n], p, atol=1e-8) for n, p in model.target_encoder.named_parameters())

    def test_target_no_grad(self):
        assert all(not p.requires_grad for p in self.JEPA(self._cfg()).target_encoder.parameters())

    def test_gradient_flow(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)[0].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters() if p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters() if p.requires_grad)


class TestSpanMask:
    def test_basic(self):
        from src.masks.span import SpanMaskCollator
        r = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=0)(
            [{'input_ids': torch.tensor([1,2,3,4,5,6,7,8,9,10])}])
        assert 'masked_input_ids' in r and 'original_input_ids' in r and 'mask_positions' in r

    def test_curriculum(self):
        from src.masks.span import SpanMaskCollator
        c = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_ratio_start=0.1,
                              mask_ratio_end=0.5, curriculum_steps=100, mask_token_id=0)
        assert c.current_mask_ratio == pytest.approx(0.1, abs=0.02)
        c._step = 100
        assert c.current_mask_ratio == pytest.approx(0.5, abs=0.02)


class TestSchedulers:
    def test_lr(self):
        from src.utils.schedulers import WarmupCosineSchedule
        opt = torch.optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.001)
        s = WarmupCosineSchedule(opt, warmup_steps=10, start_lr=1e-5, ref_lr=1e-3, final_lr=1e-6, T_max=100)
        lrs = [s.step() for _ in range(100)]
        assert lrs[5] > lrs[0] and lrs[99] <= lrs[50]

    def test_ema_tau(self):
        from src.utils.schedulers import EMATauSchedule
        s = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=1000)
        assert s.step() >= 0.996
        for _ in range(999): s.step()
        assert s.step() >= 0.999


class TestData2VecBaseline:
    """Fix #4: data2vec baseline from official fairseq code."""

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

    def test_get_annealed_rate(self):
        """Test exact formula from data2vec_text.py line ~58."""
        from baselines.data2vec_baseline import get_annealed_rate
        # At step 0: should return start
        assert get_annealed_rate(0.999, 0.9999, 0, 100) == pytest.approx(0.999, abs=1e-6)
        # At final step: should return end
        assert get_annealed_rate(0.999, 0.9999, 100, 100) == pytest.approx(0.9999, abs=1e-6)

    def test_regression_head_data2vec(self):
        """data2vec regression head: head_layers=2 → Linear→GELU→Linear."""
        from baselines.data2vec_baseline import Data2VecTextBaseline
        m = Data2VecTextBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2,
                                 num_heads=4, head_layers=2)
        # head_layers=2: Linear(64→128) → GELU → Linear(128→64)
        assert isinstance(m.regression_head[0], torch.nn.Linear)
        assert m.regression_head[0].in_features == 64
        assert m.regression_head[0].out_features == 128  # 2x expand
        assert isinstance(m.regression_head[1], torch.nn.GELU)
        assert isinstance(m.regression_head[2], torch.nn.Linear)
        assert m.regression_head[2].out_features == 64


class TestMLMBaseline:
    def test_forward(self):
        from baselines.mlm_baseline import MLMBaseline
        m = MLMBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0)
        mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 5:10] = 1
        loss, info = m.compute_loss(torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert loss.requires_grad and 'loss_mlm' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
