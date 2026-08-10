# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Tests for JAWP: Jacobian-Aligned Workspace Prediction (NOVEL MECHANISM)

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TestJAWPCore:
    """Core tests for the JAWP mechanism."""

    def test_import(self):
        from src.models.jawp import JAWPModule
        assert JAWPModule is not None

    def test_creation(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=128, k_start=1, k_end=13)
        assert jawp.embed_dim == 128
        assert jawp.k_start == 1
        assert jawp.k_end == 13
        assert jawp.workspace_Q.shape == (128, 13)

    def test_no_beta_no_gamma(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=7)
        assert not hasattr(jawp, 'beta')
        assert not hasattr(jawp, 'gamma')

    def test_learned_Q_is_parameter(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7)
        assert isinstance(jawp.workspace_Q, nn.Parameter)
        assert jawp.workspace_Q.requires_grad

    def test_identity_init(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=7, init='identity')
        Q = jawp.workspace_Q.data
        assert torch.allclose(Q[:7, :7], torch.eye(7), atol=1e-6)

    def test_random_init_is_orthogonal(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=7, init='random')
        Q = jawp.workspace_Q.data
        gram = Q.T @ Q
        assert torch.allclose(gram, torch.eye(7), atol=1e-5)

    def test_cosine_curriculum(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=128, k_start=1, k_end=13,
                          curriculum_steps=1000)
        assert jawp.current_k(0) == 1
        k_mid = jawp.current_k(500)
        assert 4 <= k_mid <= 10
        assert jawp.current_k(1000) == 13
        assert jawp.current_k(2000) == 13

    def test_stiefel_retract_keeps_orthonormal(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7, init='identity')
        with torch.no_grad():
            jawp.workspace_Q.add_(torch.randn(64, 7) * 0.5)
        gram_before = jawp.workspace_Q.data.T @ jawp.workspace_Q.data
        assert not torch.allclose(gram_before, torch.eye(7), atol=0.1)
        jawp.stiefel_retract()
        gram_after = jawp.workspace_Q.data.T @ jawp.workspace_Q.data
        assert torch.allclose(gram_after, torch.eye(7), atol=1e-5)


class TestJAWPLoss:
    """Tests for JAWP loss computation."""

    def _make_jawp(self, **kwargs):
        from src.models.jawp import JAWPModule
        defaults = dict(embed_dim=64, k_start=4, k_end=7,
                        curriculum_steps=0, alpha=0.1)
        defaults.update(kwargs)
        return JAWPModule(**defaults)

    def test_loss_is_nonnegative(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(8, 32, 64)
        z_target = torch.randn(8, 32, 64)
        loss, info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert loss.item() >= 0

    def test_loss_uses_mse_not_smooth_l1(self):
        jawp = self._make_jawp(alpha=0.0)
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        loss, info = jawp.compute_loss(z_pred, z_target, step=1000)
        Q = jawp.workspace_Q.data[:, :7]
        expected = F.mse_loss(z_pred @ Q, z_target @ Q)
        assert abs(loss.item() - expected.item()) < 1e-4

    def test_loss_components_present(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(8, 32, 64)
        z_target = torch.randn(8, 32, 64)
        _, info = jawp.compute_loss(z_pred, z_target, step=1000)
        for key in ['loss_workspace', 'loss_predictor_focus', 'k',
                    'workspace_utilization', 'target_ws_fraction',
                    'workspace_cosine', 'ortho_score',
                    'predictive_relevance', 'pca_alignment']:
            assert key in info, f"Missing key: {key}"

    def test_gradient_flows_through_Q(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
        loss.backward()
        assert jawp.workspace_Q.grad is not None
        assert jawp.workspace_Q.grad.abs().sum() > 0

    def test_gradient_flows_through_z_pred(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64, requires_grad=True)
        z_target = torch.randn(32, 64)
        loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
        loss.backward()
        assert z_pred.grad is not None
        assert z_pred.grad.abs().sum() > 0

    def test_z_target_does_not_get_gradients(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64, requires_grad=True)
        loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
        loss.backward()
        assert z_target.grad is None or z_target.grad.abs().sum() == 0

    def test_predictor_focus_penalty_works(self):
        jawp = self._make_jawp(alpha=1.0)
        Q = jawp.workspace_Q.data[:, :7]
        z_target = torch.randn(32, 64)
        z_pred_ws = (z_target @ Q) @ Q.T
        _, info_ws = jawp.compute_loss(z_pred_ws, z_target, step=1000)
        z_pred_bg = z_target - (z_target @ Q) @ Q.T
        _, info_bg = jawp.compute_loss(z_pred_bg + z_target, z_target, step=1000)
        assert info_ws['loss_predictor_focus'] < info_bg['loss_predictor_focus']

    def test_diagnostics_bounded(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        _, info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert 0 <= info['workspace_utilization'] <= 1.01
        assert 0 <= info['target_ws_fraction'] <= 1.01
        assert -1 <= info['workspace_cosine'] <= 1.01
        assert 0 <= info['ortho_score'] <= 1.01

    def test_no_nan_or_inf(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(4, 8, 64) * 1e-7
        z_target = torch.randn(4, 8, 64) * 1e-7
        loss, info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert math.isfinite(loss.item())
        for k, v in info.items():
            assert math.isfinite(v), f"{k} = {v}"

    def test_stiefel_retract_after_training_step(self):
        jawp = self._make_jawp()
        optimizer = torch.optim.Adam(jawp.parameters(), lr=0.01)
        for _ in range(10):
            z_pred = torch.randn(16, 64)
            z_target = torch.randn(16, 64)
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            jawp.stiefel_retract()
        Q = jawp.workspace_Q.data[:, :7]
        gram = Q.T @ Q
        assert torch.allclose(gram, torch.eye(7), atol=1e-4)


class TestJAWPCourantFischer:
    """Tests for the Courant-Fischer theorem and convergence guarantees."""

    def _make_synthetic_data(self, D=32, k=4, N=128, noise_pred=0.01, noise_unpred=5.0):
        torch.manual_seed(42)
        z_pred = torch.randn(N, D)
        z_target = z_pred.clone()
        for i in range(k):
            z_target[:, i] = z_pred[:, i] + torch.randn(N) * noise_pred
        for i in range(k, D):
            z_target[:, i] = torch.randn(N) * noise_unpred
        return z_pred, z_target

    def test_gradient_form_matches_courant_fischer(self):
        from src.models.jawp import JAWPModule
        D, k, N = 32, 4, 64
        jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k,
                          curriculum_steps=0, init='random', alpha=0.0)
        z_pred = torch.randn(N, D)
        z_target = torch.randn(N, D)
        Q = jawp.workspace_Q[:, :k]
        pred_ws = z_pred @ Q
        target_ws = z_target.detach() @ Q
        loss = F.mse_loss(pred_ws, target_ws)
        loss.backward()
        grad_Q = jawp.workspace_Q.grad.data[:, :k]
        Q_data = jawp.workspace_Q.data[:, :k]
        R = z_pred - z_target
        analytical_grad = (2.0 / (N * k)) * R.T @ R @ Q_data
        relative_err = (grad_Q - analytical_grad).norm().item() / (grad_Q.norm().item() + 1e-10)
        assert relative_err < 0.01

    def test_convergence_to_optimal_subspace(self):
        from src.models.jawp import JAWPModule
        D, k, N = 32, 4, 128
        z_pred, z_target = self._make_synthetic_data(D, k, N)
        R = z_pred - z_target
        Sigma_res = (R.T @ R) / (N - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_res)
        Q_optimal = eigenvectors[:, :k]
        optimal_risk = torch.trace(Q_optimal.T @ Sigma_res @ Q_optimal).item()
        jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k,
                          curriculum_steps=0, init='random', alpha=0.0)
        optimizer = torch.optim.SGD([jawp.workspace_Q], lr=0.02)
        for _ in range(1000):
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            jawp.stiefel_retract()
            optimizer.step()
            jawp.stiefel_retract()
        Q_learned = jawp.workspace_Q.data[:, :k]
        learned_risk = torch.trace(Q_learned.T @ Sigma_res @ Q_learned).item()
        risk_ratio = learned_risk / (optimal_risk + 1e-10)
        assert risk_ratio < 1.5

    def test_corollary_jawp_leq_pca(self):
        from src.models.jawp import JAWPModule
        D, k, N = 32, 4, 128
        z_pred, z_target = self._make_synthetic_data(D, k, N)
        R = z_pred - z_target
        Sigma_res = (R.T @ R) / (N - 1)
        cov_target = (z_target.T @ z_target) / (N - 1)
        _, V_pca = torch.linalg.eigh(cov_target)
        Q_pca = V_pca[:, -k:]
        pca_risk = torch.trace(Q_pca.T @ Sigma_res @ Q_pca).item()
        jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k,
                          curriculum_steps=0, init='random', alpha=0.0)
        optimizer = torch.optim.SGD([jawp.workspace_Q], lr=0.02)
        for _ in range(1000):
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            jawp.stiefel_retract()
            optimizer.step()
            jawp.stiefel_retract()
        Q_learned = jawp.workspace_Q.data[:, :k]
        jawp_risk = torch.trace(Q_learned.T @ Sigma_res @ Q_learned).item()
        assert jawp_risk <= pca_risk + 1e-3

    def test_subspace_similarity_with_optimal(self):
        from src.models.jawp import JAWPModule
        D, k, N = 32, 4, 128
        z_pred, z_target = self._make_synthetic_data(D, k, N)
        R = z_pred - z_target
        Sigma_res = (R.T @ R) / (N - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_res)
        Q_optimal = eigenvectors[:, :k]
        jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k,
                          curriculum_steps=0, init='random', alpha=0.0)
        optimizer = torch.optim.SGD([jawp.workspace_Q], lr=0.02)
        for _ in range(1000):
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            jawp.stiefel_retract()
            optimizer.step()
            jawp.stiefel_retract()
        Q_learned = jawp.workspace_Q.data[:, :k]
        cross = Q_learned.T @ Q_optimal
        similarity = (cross ** 2).sum() / k
        assert similarity > 0.9


class TestJAWPNovelty:
    """Tests that verify JAWP's novelty vs. PCA."""

    def test_task_adaptivity_not_just_PCA(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=32, k_start=2, k_end=2,
                          curriculum_steps=0, init='random', alpha=0.1)
        optimizer = torch.optim.Adam(jawp.parameters(), lr=0.02)
        for _ in range(300):
            z_pred = torch.randn(32, 32)
            z_target = z_pred.clone()
            z_target[:, 0] = torch.randn(32) * 10.0
            z_target[:, 1] = torch.randn(32) * 10.0
            z_target[:, 2] = z_pred[:, 2] + torch.randn(32) * 0.1
            z_target[:, 3] = z_pred[:, 3] + torch.randn(32) * 0.1
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            jawp.stiefel_retract()
        Q = jawp.workspace_Q.data[:, :2]
        pred_align = Q[2:4, :].norm().item()
        unpred_align = Q[0:2, :].norm().item()
        ratio = pred_align / (unpred_align + 1e-10)
        assert ratio > 1.5


class TestJAWPAPI:
    """Tests for the public API that other papers will use."""

    def test_get_workspace_basis(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7,
                          curriculum_steps=0)
        Q = jawp.get_workspace_basis()
        assert Q.shape == (64, 4)

    def test_project_to_workspace(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7,
                          curriculum_steps=0)
        z = torch.randn(8, 32, 64)
        z_ws = jawp.project_to_workspace(z)
        assert z_ws.shape == (8, 32, 4)

    def test_project_to_background(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7,
                          curriculum_steps=0)
        z = torch.randn(8, 32, 64)
        z_bg = jawp.project_to_background(z)
        assert z_bg.shape == (8, 32, 64)

    def test_workspace_plus_background_equals_original(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7,
                          curriculum_steps=0, init='identity')
        z = torch.randn(8, 32, 64)
        Q = jawp.get_workspace_basis()
        z_ws = jawp.project_to_workspace(z)
        z_bg = jawp.project_to_background(z)
        z_recon = z_bg + (z_ws @ Q.T)
        assert torch.allclose(z, z_recon, atol=1e-5)

    def test_drop_in_api(self):
        from src.models.jawp import JAWPModule
        jawp = JAWPModule(embed_dim=768, k_start=1, k_end=77)
        z_pred = torch.randn(4, 32, 768)
        z_target = torch.randn(4, 32, 768)
        loss, info = jawp.compute_loss(z_pred, z_target, step=5000)
        assert math.isfinite(loss.item())

    def test_all_shapes_correct(self):
        from src.models.jawp import JAWPModule
        D, k_start, k_end = 64, 2, 7
        jawp = JAWPModule(embed_dim=D, k_start=k_start, k_end=k_end,
                          curriculum_steps=0, init='identity')
        assert jawp.workspace_Q.shape == (D, k_end)
        B, T = 4, 16
        z_pred = torch.randn(B, T, D)
        z_target = torch.randn(B, T, D)
        loss, info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert loss.shape == torch.Size([])
        assert info['k'] == k_end
        Q = jawp.get_workspace_basis()
        assert Q.shape == (D, k_end)
        z_ws = jawp.project_to_workspace(z_pred)
        assert z_ws.shape == (B, T, k_end)
        z_bg = jawp.project_to_background(z_pred)
        assert z_bg.shape == (B, T, D)
        z_recon = z_bg + (z_ws @ Q.T)
        assert z_recon.shape == (B, T, D)
        assert torch.allclose(z_pred, z_recon, atol=1e-5)


class TestJAWPWithJEPA:
    """Integration tests: JEPA model with JAWP."""

    def test_jepa_with_jawp(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        config = TextSpanJEPAConfig(
            embed_dim=64, encoder_depth=2, num_heads=4,
            predictor_embed_dim=32, predictor_depth=1,
            max_seq_len=32, vocab_size=100,
            use_jawp=True, jawk_k_start=1, jawk_k_end=7,
            future_offsets=(1,), num_refine_steps=0,
        )
        config.validate()
        model = TextSpanJEPA(config)
        assert model.jawp is not None

    def test_jepa_without_jawp(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        config = TextSpanJEPAConfig(
            embed_dim=64, encoder_depth=2, num_heads=4,
            predictor_embed_dim=32, predictor_depth=1,
            max_seq_len=32, vocab_size=100,
            use_jawp=False,
            future_offsets=(1,), num_refine_steps=0,
        )
        config.validate()
        model = TextSpanJEPA(config)
        assert model.jawp is None

    def test_loss_with_jawp_is_valid(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        config = TextSpanJEPAConfig(
            embed_dim=64, encoder_depth=2, num_heads=4,
            predictor_embed_dim=32, predictor_depth=1,
            max_seq_len=32, vocab_size=100,
            use_jawp=True, jawk_k_start=1, jawk_k_end=7,
            future_offsets=(1,), num_refine_steps=0,
        )
        config.validate()
        model = TextSpanJEPA(config)
        masked = torch.randint(0, 100, (2, 32))
        original = torch.randint(0, 100, (2, 32))
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 8:16] = 1
        total_loss, loss_dict, diag_dict = model.compute_loss_with_targets(
            masked, original, mask, current_step=100, total_steps=1000)
        assert math.isfinite(total_loss.item())
        assert total_loss.item() > 0
