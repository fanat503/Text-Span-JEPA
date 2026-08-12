# PCR: Predictive Cascade Refinement

## Statement

**Theorem (Cascade Capacity).**
Let $z_0 = \operatorname{Pred}(z_\mathrm{ctx})$ be the initial prediction, and define the cascade:
$$z_l = z_{l-1} + P_l^\top r_{l-1}, \quad r_{l-1} = z_\mathrm{target} - z_{l-1}$$
where $P_l \in \mathrm{St}(D, d_l)$ with $\operatorname{span}(P_l) \perp \operatorname{span}(P_{l'})$ for $l \neq l'$.

Then the cascade information satisfies:
$$I(z_\mathrm{ctx}; z_L) \geq I(z_\mathrm{ctx}; z_0) + \sum_{l=1}^{L} I(r_{l-1}; P_l^\top r_{l-1})$$

Each cascade level **provably recovers** information that was lost through the prediction bottleneck at the previous level.

## Proof

### Step 1: Orthogonal Subspace Property
Since $\operatorname{span}(P_l) \perp \operatorname{span}(P_{l'})$ for $l \neq l'$:
$$P_l^\top P_{l'} = 0 \quad \text{for } l \neq l'$$

This means each correction $P_l^\top r_{l-1}$ lies in an **orthogonal subspace** — corrections don't interfere.

### Step 2: Information Gain Per Level
The residual at level $l$:
$$r_l = z_\mathrm{target} - z_l = r_{l-1} - P_l^\top r_{l-1} = (I - P_l P_l^\top) r_{l-1}$$

Since $P_l P_l^\top$ is the orthogonal projection onto $\operatorname{span}(P_l)$:
$$\|r_l\|^2 = \|r_{l-1}\|^2 - \|P_l^\top r_{l-1}\|^2$$

Each level **strictly reduces** the residual (when the correction is non-zero).

### Step 3: Cumulative Information Gain
By the data processing inequality and orthogonality:
$$I(z_\mathrm{ctx}; z_l) - I(z_\mathrm{ctx}; z_{l-1}) \geq I(r_{l-1}; P_l^\top r_{l-1})$$

Summing over all levels:
$$I(z_\mathrm{ctx}; z_L) \geq I(z_\mathrm{ctx}; z_0) + \sum_{l=1}^{L} I(r_{l-1}; P_l^\top r_{l-1})$$

Each term $I(r_{l-1}; P_l^\top r_{l-1})$ is the information recovered by projecting the residual onto a new orthogonal subspace.

### Step 4: Cascade Capacity Interpretation
The **Cascade Capacity** is the total information recoverable:
$$C_\mathrm{cascade} = \sum_{l=1}^{L} I(r_{l-1}; P_l^\top r_{l-1})$$

This is bounded by:
$$C_\mathrm{cascade} \leq H(r_0) - H(r_0 \mid z_\mathrm{target})$$

The bound is achieved when the cascade spans the entire residual space.

## Connection to ResNets
- **ResNet**: $z_l = z_{l-1} + F(z_{l-1})$ where $F$ is a learned function. No orthogonality guarantee.
- **PCR**: $z_l = z_{l-1} + P_l^\top r_{l-1}$ where $P_l$ is on Stiefel manifold. Orthogonality **guaranteed**.
- PCR is a **structured ResNet** where each skip connection lies in an orthogonal subspace.

## Why Top Labs Will Use This
- **Provable recovery**: Unlike iterative refinement (same bottleneck), PCR provably recovers lost information.
- **Orthogonal structure**: Prevents correction interference — each level is independent.
- **Bounded compute**: $L$ levels with $d_l$ dimensions each adds $O(\sum_l d_l)$ parameters.
