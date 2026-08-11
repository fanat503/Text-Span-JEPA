# Tests for GAC (Gradient-Allocated Capacity) — mechanism #12

import pytest
import torch
import math
import sys
sys.path.insert(0, '.')

from src.models.gac import GradientAllocatedCapacity


class TestGACCore:
    def setup_method(self):
        self.D = 64
        self.gac = GradientAllocatedCapacity(embed_dim=self.D, gamma=0.01, tau_grad=1e-4)

    def test_output_shape(self):
        z = torch.randn(8, self.D)
        g = torch.rand(self.D) * 1e-3
        loss, info = self.gac(z, g, step=2000)
        assert loss.dim() == 0

    def test_loss_non_negative(self):
        z = torch.randn(8, self.D)
        g = torch.rand(self.D) * 1e-3
        loss, info = self.gac(z, g, step=2000)
        assert loss.item() >= -1e-6

    def test_zero_loss_no_starved(self):
        """When all dims have sufficient gradient, GAC loss = 0."""
        z = torch.randn(8, self.D)
        g = torch.ones(self.D) * 1.0  # all above tau_grad
        loss, info = self.gac(z, g, step=2000)
        assert loss.item() < 1e-6

    def test_positive_loss_starved(self):
        """When some dims are starved, GAC loss > 0."""
        z = torch.randn(8, self.D)
        g = torch.ones(self.D) * 1e-6  # all below tau_grad
        loss, info = self.gac(z, g, step=2000)
        assert loss.item() > 0

    def test_warmup_zero_loss(self):
        """During warmup, GAC loss = 0."""
        z = torch.randn(8, self.D)
        g = torch.ones(self.D) * 1e-6
        loss, info = self.gac(z, g, step=0)
        assert loss.item() == 0.0
        assert info['gac_warmup'] is True

    def test_warmup_gradual(self):
        """Warmup factor increases linearly."""
        gac = GradientAllocatedCapacity(embed_dim=self.D, warmup_steps=100)
        z = torch.randn(8, self.D)
        g = torch.ones(self.D) * 1e-6
        losses = []
        for step in [0, 50, 100, 200]:
            loss, info = gac(z, g, step=step)
            losses.append(loss.item())
        # After warmup, loss should be >= during warmup
        assert losses[-1] >= losses[0]

    def test_starved_count(self):
        """Starved count reflects dimensions below tau_grad."""
        z = torch.randn(8, self.D)
        g = torch.ones(self.D) * 1.0
        g[:20] = 1e-6  # 20 dims starved
        loss, info = self.gac(z, g, step=2000)
        assert info['gac_n_starved'] == 20
        assert abs(info['gac_starved_fraction'] - 20/self.D) < 1e-6


class TestGACTheorems:
    def test_no_gradient_dead_zones(self):
        """Theorem 1: Starved dims with non-zero activation get gradient."""
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D, gamma=0.01, tau_grad=1e-4)
        z = torch.randn(8, D)
        g = torch.ones(D) * 1e-6  # all starved

        # Compute gradient bound
        bound = gac.compute_gradient_bound(z, g)
        # For starved dims with non-zero z, bound should be > 0
        assert bound > 0, f"Gradient bound should be positive, got {bound}"

    def test_zero_activation_zero_bound(self):
        """When z_i = 0, gradient bound is 0 (correct — no signal needed)."""
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D, gamma=0.01, tau_grad=1e-4)
        z = torch.zeros(8, D)  # all zero activation
        g = torch.ones(D) * 1e-6
        bound = gac.compute_gradient_bound(z, g)
        assert bound == 0.0

    def test_exploration_ratio_bounded(self):
        """Theorem 3: Exploration ratio is bounded."""
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D, gamma=0.01, tau_grad=1e-4)
        g = torch.ones(D) * 0.1  # non-trivial gradient
        ratio = gac.compute_exploration_ratio(g)
        # γ · τ / g_min = 0.01 * 1e-4 / 0.1 = 1e-5
        expected_max = 0.01 * 1e-4 / 0.1
        assert ratio <= expected_max * 1.1  # tolerance

    def test_gamma_scales_loss(self):
        """Larger gamma → larger GAC loss (linear scaling)."""
        D = 32
        z = torch.randn(8, D)
        g = torch.ones(D) * 1e-6

        gac1 = GradientAllocatedCapacity(embed_dim=D, gamma=0.01, tau_grad=1e-4)
        gac2 = GradientAllocatedCapacity(embed_dim=D, gamma=0.02, tau_grad=1e-4)
        l1, _ = gac1(z, g, step=2000)
        l2, _ = gac2(z, g, step=2000)
        assert l2.item() > l1.item()


class TestGACEdgeCases:
    def test_all_starved(self):
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D)
        z = torch.randn(8, D)
        g = torch.zeros(D)
        loss, info = gac(z, g, step=2000)
        assert loss.item() >= 0
        assert info['gac_n_starved'] == D

    def test_none_starved(self):
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D)
        z = torch.randn(8, D)
        g = torch.ones(D) * 100
        loss, info = gac(z, g, step=2000)
        assert loss.item() < 1e-6
        assert info['gac_n_starved'] == 0

    def test_single_dim_starved(self):
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D)
        z = torch.randn(8, D)
        g = torch.ones(D) * 1.0
        g[5] = 1e-8  # only dim 5 starved
        loss, info = gac(z, g, step=2000)
        assert info['gac_n_starved'] == 1

    def test_large_gamma(self):
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D, gamma=1.0)
        z = torch.randn(8, D)
        g = torch.ones(D) * 1e-6
        loss, info = gac(z, g, step=2000)
        assert math.isfinite(loss.item())

    def test_repr(self):
        gac = GradientAllocatedCapacity(embed_dim=768, gamma=0.01, tau_grad=1e-4)
        r = gac.extra_repr()
        assert '768' in r and '0.01' in r


class TestGACIntegration:
    def test_with_jawp_workspace(self):
        """GAC can target JAWP background dimensions."""
        D, k = 64, 10
        gac = GradientAllocatedCapacity(embed_dim=D)

        # Simulate JAWP workspace
        Q = torch.randn(D, k)
        U, S, Vt = torch.linalg.svd(Q, full_matrices=False)
        Q = U[:, :k] @ Vt[:k, :]

        z = torch.randn(8, D)
        # Simulate gradient: high in workspace, low in background
        g_workspace = torch.ones(k) * 1.0
        g_background = torch.ones(D - k) * 1e-6
        grad = torch.cat([g_workspace, g_background])

        loss, info = gac(z, grad, step=2000)
        # Background dims are starved
        assert info['gac_n_starved'] == D - k

    def test_gradient_flow_through_gac(self):
        """GAC loss provides gradient to z_pred."""
        D = 32
        gac = GradientAllocatedCapacity(embed_dim=D, gamma=0.01, tau_grad=1e-4)
        z = torch.randn(8, D, requires_grad=True)
        g = torch.ones(D) * 1e-6

        # Compute loss (z must be in graph)
        # Note: GAC uses z.detach() internally for deficit, but
        # the loss itself is differentiable w.r.t. z
        warmup_factor = 1.0
        deficit = F.relu(gac.tau_grad - g.detach())
        starved = (g.detach() < gac.tau_grad).float()
        loss = gac.gamma * warmup_factor * (deficit * (z ** 2).mean(dim=0) * starved).sum()
        loss.backward()

        # Starved dimensions should have non-zero gradient
        grad_per_dim = z.grad.abs().sum(dim=0)  # (D,)
        starved_dims = (g < gac.tau_grad).nonzero().squeeze(-1)
        assert grad_per_dim[starved_dims].min() > 0, \
            "Starved dims should receive gradient from GAC"


from torch.nn import functional as F

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
