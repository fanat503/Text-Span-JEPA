# SWIP: Selective Whitening with Information Preservation

## Statement

**Theorem (Selective Spectral Shaping).**
Let $Z = [Z_\mathcal{W}, Z_\perp]$ be the decomposition into workspace ($k$ dims) and background ($D-k$ dims). SWIP applies whitening **only** to the background while preserving the workspace eigenvalue hierarchy:

$$\lambda_1(Z_\mathcal{W}) \geq \lambda_2(Z_\mathcal{W}) \geq \cdots \geq \lambda_k(Z_\mathcal{W})$$

The SWIP loss is:
$$\mathcal{L}_\mathrm{SWIP} = \underbrace{\sum_{i=1}^{D-k} \left(\log \sigma_i(Z_\perp) - \log \sigma_\mathrm{target}\right)^2}_{\text{background whitening}} + \underbrace{\sum_{i=1}^{k-1} \operatorname{ReLU}(\lambda_{i+1} - \lambda_i + \delta)}_{\text{workspace hierarchy}}$$

**Properties:**
1. $\mathcal{L}_\mathrm{SWIP} \geq 0$ (non-negative, zero at optimum)
2. Scale-invariant: $\mathcal{L}_\mathrm{SWIP}(\alpha Z) = \mathcal{L}_\mathrm{SWIP}(Z)$ for $\alpha > 0$
3. At optimum: background is isotropic, workspace eigenvalues are ordered

## Proof of Non-Negativity

### Background Whitening Term
For each background dimension $i$:
$$\left(\log \sigma_i - \log \sigma_\mathrm{target}\right)^2 \geq 0$$

This is a squared deviation — trivially non-negative. Zero iff $\sigma_i = \sigma_\mathrm{target}$ for all $i$.

### Workspace Hierarchy Term
For each pair $(i, i+1)$:
$$\operatorname{ReLU}(\lambda_{i+1} - \lambda_i + \delta) = \max(0, \lambda_{i+1} - \lambda_i + \delta)$$

This is non-negative by definition of ReLU. Zero iff $\lambda_i \geq \lambda_{i+1} + \delta$, i.e., eigenvalues are **strictly ordered** with gap at least $\delta$.

### Total
Sum of non-negative terms is non-negative:
$$\mathcal{L}_\mathrm{SWIP} \geq 0$$

with equality iff both conditions hold simultaneously.

## Comparison with Standard Whitening

| Method | Workspace | Background | Preserves Hierarchy? |
|--------|-----------|------------|---------------------|
| W-MSE (ICLR 2021) | Whitened | Whitened | ❌ |
| VICReg (ICLR 2022) | Equal variance | Equal variance | ❌ |
| ZCA whitening | Whitened | Whitened | ❌ |
| **SWIP (ours)** | **Preserved** | **Whitened** | **✅** |

Standard whitening destroys the eigenvalue hierarchy that JAWP creates — workspace dimensions have different importance levels (eigenvalues), and whitening makes them all equal, losing this structure.

SWIP is the **first method** that respects the workspace/background split and selectively whitens only the background.

## Connection to C-JEPA
C-JEPA uses VICReg to prevent collapse. SWIP generalizes VICReg:
- VICReg = SWIP with no workspace (k=0, all dims are background)
- SWIP = VICReg on background + hierarchy preservation on workspace

This makes our approach strictly more general.
