# SPC: Spectral Predictive Coding

## Statement

**Theorem (Information-Proportional Capacity Allocation).**
Let $z = \sum_b U_b c_b$ be the spectral decomposition into $B$ frequency bands, where $U_b \in \mathbb{R}^{D \times d_b}$ are learned band bases (on Stiefel manifold). The SPC-weighted prediction loss is:

$$\mathcal{L}_\mathrm{SPC} = \sum_{b=1}^{B} w_b \cdot \|c_b^{\mathrm{pred}} - c_b^{\mathrm{target}}\|^2$$

where $w_b \in \Delta^{B-1}$ (simplex-constrained band weights).

**Property (Parseval's Equality).**
When the band bases $U_b$ are orthonormal (maintained by Stiefel retraction):

$$\|z^{\mathrm{pred}} - z^{\mathrm{target}}\|^2 = \sum_{b=1}^{B} \|c_b^{\mathrm{pred}} - c_b^{\mathrm{target}}\|^2$$

The total prediction error equals the sum of per-band errors — no information is lost by the spectral decomposition.

## Proof of Parseval's Equality

### Setup
Let $U = [U_1, \ldots, U_B] \in \mathrm{St}(D, D)$ be the concatenated band basis. Since $U$ is orthonormal ($U^\top U = I_D$):

$$z = U c = \sum_b U_b c_b, \quad c = U^\top z$$

### Parseval's Equality
$$\|z^{\mathrm{pred}} - z^{\mathrm{target}}\|^2 = \|U(c^{\mathrm{pred}} - c^{\mathrm{target}})\|^2 = (c^{\mathrm{pred}} - c^{\mathrm{target}})^\top U^\top U (c^{\mathrm{pred}} - c^{\mathrm{target}})$$

Since $U^\top U = I$:
$$= \|c^{\mathrm{pred}} - c^{\mathrm{target}}\|^2 = \sum_{b=1}^{B} \|c_b^{\mathrm{pred}} - c_b^{\mathrm{target}}\|^2$$

### Weighted Loss Property
The SPC-weighted loss satisfies:
$$\min_b(w_b) \cdot \|z^{\mathrm{pred}} - z^{\mathrm{target}}\|^2 \leq \mathcal{L}_\mathrm{SPC} \leq \max_b(w_b) \cdot \|z^{\mathrm{pred}} - z^{\mathrm{target}}\|^2$$

When all $w_b = 1/B$: $\mathcal{L}_\mathrm{SPC} = \frac{1}{B} \|z^{\mathrm{pred}} - z^{\mathrm{target}}\|^2$ (uniform weighting, recovers standard JEPA).

## Simplex Constraint
The band weights $w_b$ are constrained to the simplex $\Delta^{B-1}$ via Gumbel-Softmax:
$$w_b = \frac{\exp((\alpha_b + g_b) / \tau)}{\sum_{b'} \exp((\alpha_{b'} + g_{b'}) / \tau)}$$

where $\alpha_b$ are learnable logits, $g_b$ is Gumbel noise, and $\tau$ is temperature.

### Interpretation
- High $w_b$: allocate more prediction capacity to band $b$
- Low $w_b$: reduce capacity for band $b$
- The model learns which frequency bands are most informative for prediction

## Why This Is Novel
- **Focal Loss** (ICCV 2017): down-weights easy **samples** — SPC down-weights easy **frequency bands**
- **Multi-scale prediction** (FPN, U-Net): different scales at different **layers** — SPC operates within a **single layer**
- **Spectral regularization** (Lipschitz): constrains weight spectra — SPC adaptively weights **prediction loss** per band
- No prior work learns simplex-constrained band weights for JEPA prediction loss

## DCT Initialization
Band bases are initialized as DCT (Discrete Cosine Transform) basis vectors, providing:
- Low bands: smooth, low-frequency structure
- High bands: fine-grained, high-frequency detail
- Stiefel retraction maintains orthonormality during training
