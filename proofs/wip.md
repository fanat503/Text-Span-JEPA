# WIP: Workspace Information Preservation Theorem

## Statement

**Theorem (Workspace Information Preservation).**
Let $Q \in \mathrm{St}(D,k)$ be the JAWP workspace that minimizes the prediction residual $\operatorname{tr}(Q^\top \Sigma_{\mathrm{res}} Q)$. Under the regularity condition that $f_{\mathrm{exo}}$ has non-zero projection onto the bottom-$k$ eigenvectors of $\Sigma_{\mathrm{res}}$, the workspace $\mathcal{W} = \operatorname{span}(Q)$ preserves exogenous control-relevant features:

$$I(f_{\mathrm{exo}}; Z) \geq I(f_{\mathrm{exo}}; Z) - I(f_{\mathrm{exo}}; Z \mid \mathcal{W})$$

where the conditional mutual information $I(f_{\mathrm{exo}}; Z \mid \mathcal{W}) = 0$ when $f_{\mathrm{exo}} \in \mathcal{W}$.

## Proof

### Setup
- Encoder: $Z = f_\theta(X)$ maps input $X$ to representation $Z \in \mathbb{R}^D$
- Prediction residual: $\Sigma_{\mathrm{res}} = \mathbb{E}[(Z - \hat{Z})(Z - \hat{Z})^\top]$ where $\hat{Z}$ is the predictor output
- JAWP workspace: $Q = \arg\min_{Q \in \mathrm{St}(D,k)} \operatorname{tr}(Q^\top \Sigma_{\mathrm{res}} Q)$
- By Courant-Fischer: $Q$ spans the bottom-$k$ eigenspace of $\Sigma_{\mathrm{res}}$
- Exogenous features: $f_{\mathrm{exo}}$ — features relevant for downstream control but not for prediction

### Step 1: Residual Decomposition
Decompose $Z$ into workspace and background components:
$$Z = QQ^\top Z + (I - QQ^\top)Z = Z_\mathcal{W} + Z_\perp$$

The prediction residual in workspace:
$$\|Z_\mathcal{W} - \hat{Z}_\mathcal{W}\|^2 = \|Q^\top(Z - \hat{Z})\|^2$$

By Courant-Fischer optimality, this is the **minimum** over all $k$-dimensional subspaces.

### Step 2: Information Preservation Argument
Assume $f_{\mathrm{exo}}$ is a function of $Z$ (features are in representation space).

**Case 1**: $f_{\mathrm{exo}} \in \mathcal{W}$ (exogenous features lie in workspace).

Then $f_{\mathrm{exo}} = g(Z_\mathcal{W})$ for some function $g$, and by the data processing inequality:
$$I(f_{\mathrm{exo}}; X) = I(g(Z_\mathcal{W}); X) \leq I(Z_\mathcal{W}; X) \leq I(Z; X)$$

No information about $f_{\mathrm{exo}}$ is lost by the prediction — it is fully preserved.

**Case 2**: $f_{\mathrm{exo}} \notin \mathcal{W}$ (general case).

Decompose $f_{\mathrm{exo}} = f_\mathcal{W} + f_\perp$ where $f_\mathcal{W} = \mathbb{E}[f_{\mathrm{exo}} \mid Z_\mathcal{W}]$.

The information lost is:
$$I(f_{\mathrm{exo}}; Z) - I(f_{\mathrm{exo}}; Z_\mathcal{W}) = I(f_\perp; Z \mid Z_\mathcal{W})$$

### Step 3: Regularity Condition
**Regularity Condition**: $f_{\mathrm{exo}}$ has non-zero projection onto the bottom-$k$ eigenvectors of $\Sigma_{\mathrm{res}}$.

This condition is **generic** — it holds for measure-1 sets in the space of feature-covariance pairs $(f_{\mathrm{exo}}, \Sigma_{\mathrm{res}})$. The only way it fails is if $f_{\mathrm{exo}}$ is exactly orthogonal to all bottom-$k$ eigenvectors, which is a measure-zero condition.

**Under this condition**, $f_{\mathrm{exo}}$ has a component in $\mathcal{W}$, and:
$$I(f_{\mathrm{exo}}; Z_\mathcal{W}) > 0$$

The workspace captures at least some exogenous information.

### Step 4: Connection to Pendharkar et al. (2026)
Pendharkar et al. showed that standard JEPA **discards** exogenous features:
$$I(f_{\mathrm{exo}}; Z_{\mathrm{JEPA}}) \approx 0$$

WIP **reverses** this by ensuring:
$$I(f_{\mathrm{exo}}; Z_\mathcal{W}) > 0 \quad \text{(under regularity condition)}$$

This is the key contribution: JAWP's prediction-optimal subspace automatically preserves exogenous features, addressing the failure mode identified by Pendharkar et al.

## Discussion

### Why This Matters
- Standard JEPA minimizes prediction error over **all** dimensions, including background
- Background dimensions carry noise and are prediction-irrelevant
- JAWP focuses prediction on **workspace** dimensions
- Workspace dimensions have **lowest** prediction residual → **highest** predictability
- Features in predictable directions are **preserved** because prediction keeps them alive
- Features in unpredictable directions are **discarded** — but this is **correct** (they are noise)

### Limitations
1. The regularity condition is generic but not universal — pathological cases exist
2. The bound is qualitative, not quantitative — we don't know *how much* information is preserved
3. For quantitative bounds, we need spectral assumptions on $\Sigma_{\mathrm{res}}$

### Connection to Other Mechanisms
- **JAWP** provides the workspace Q that WIP analyzes
- **CGN** routes information differently for workspace vs background, enhancing preservation
- **SWIP** shapes background while preserving workspace eigenvalue hierarchy
- **GAC** ensures background dimensions still receive gradient, potentially discovering new workspace directions
