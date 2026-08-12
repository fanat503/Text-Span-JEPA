# Pre-Experimental Hypotheses — Text-Span JEPA

This file documents our hypotheses BEFORE running experiments. This follows
the top-lab practice of pre-registering predictions to avoid HARKing
(Hypothesizing After Results are Known).

## H1: JAWP Workspace Improves Prediction Quality

**Hypothesis**: JAWP workspace prediction loss will be significantly lower
than full-space MSE for the same predictor architecture.

**Rationale**: By Courant-Fischer, JAWP predicts only in the most predictable
k dimensions. The remaining D-k dimensions (noise) contribute zero useful
gradient but large loss in full-space MSE.

**Prediction**: JAWP loss / full-space loss < 0.3 at convergence.

**Ablation**: `no_jawp.yaml` vs full config.

## H2: CGN Routing Improves Downstream Accuracy

**Hypothesis**: CGN's different gating for visible vs masked positions will
improve linear probe accuracy by ≥2% over uniform processing.

**Rationale**: The Information Routing theorem guarantees that routing
information differently at masked positions (prediction-focused) vs visible
positions (context-preserving) preserves more task-relevant information.

**Prediction**: Linear probe accuracy with CGN > without CGN by ≥2%.

**Ablation**: `cgn_on.yaml` vs `no_cgn` (implicit in default).

## H3: STA Reduces Workspace Oscillation

**Hypothesis**: STA will reduce the standard deviation of JAWP workspace
Q's principal angles between consecutive logging steps by ≥50%.

**Rationale**: The Davis-Kahan bound d_Gr ≤ W_1/δ shows that bounding
W_1 (via STA) directly bounds workspace drift.

**Prediction**: std(principal_angles) with STA < 0.5 * std(without STA).

**Ablation**: `sta_on.yaml` vs `no_sta.yaml`.

## H4: CMC Consistency Improves Representation Stability

**Hypothesis**: CMC will reduce the variance of predictions under different
masking patterns by ≥30%.

**Rationale**: The Stability Theorem guarantees |f(z_pred_1) - f(z_pred_2)| ≤
||w|| √(L_CMC). Smaller L_CMC → more stable downstream predictions.

**Prediction**: Var_mask(z_pred) with CMC < 0.7 * Var_mask(without CMC).

**Ablation**: `cmc_on.yaml` vs `no_cmc.yaml`.

## H5: GAC Prevents Gradient Starvation

**Hypothesis**: GAC will maintain non-zero gradient flow in ≥95% of
dimensions throughout training.

**Rationale**: The No Dead Zones theorem guarantees that every active
dimension receives gradient signal: ||∂L_GAC/∂z_i|| > 0 when z_i ≠ 0
and ||g_i|| < τ_grad.

**Prediction**: Starved fraction with GAC < 0.05 throughout training.

**Ablation**: `gac_on.yaml` vs `no_gac.yaml`.

## H6: SWIP Preserves Workspace While Whitening Background

**Hypothesis**: SWIP will make background dimensions approximately isotropic
(variance ratio < 2.0) while preserving workspace eigenvalue hierarchy
(ratio > 0.8).

**Rationale**: SWIP's log-eigenvalue matching targets background variance
= σ²_target while leaving workspace eigenvalues unchanged.

**Prediction**: background_condition_number with SWIP < 2.0,
workspace_cosine_similarity > 0.8.

**Ablation**: `swip_on.yaml` vs `no_swip.yaml`.

## H7: PCR Recovers Information Lost Through Bottleneck

**Hypothesis**: PCR cascade refinement will reduce prediction error by ≥15%
compared to single-pass prediction.

**Rationale**: The Cascade Capacity theorem shows that each orthogonal
subspace adds complementary information: I(z_ctx; z_L) ≥ I(z_ctx; z_0) +
Σ I(r_{l-1}; P_l r_{l-1}).

**Prediction**: loss_span with PCR < 0.85 * loss_span without PCR.

**Ablation**: `pcr_on.yaml` vs `no_pcr.yaml`.

## H8: WSD Prevents Workspace-Target Desynchronization

**Hypothesis**: WSD will keep the Grassmann distance between JAWP workspace
and target encoder workspace below a threshold that decreases with λ_wsd.

**Rationale**: The Drift Bound theorem: Δ_WSD(t) ≤ Δ(0)·exp(-λ·t) +
ν_max/λ. With λ_wsd > 0, drift is exponentially bounded.

**Prediction**: max_drift with WSD < 0.5 * max_drift without WSD.

**Ablation**: `wsd_on.yaml` vs `no_wsd.yaml`.

## H9: Mechanism Composition Is Approximately Additive

**Hypothesis**: The total improvement from all mechanisms will be
approximately the sum of individual improvements (no strong negative
interactions).

**Rationale**: Mechanisms operate on orthogonal aspects of the representation:
JAWP (workspace), CGN (routing), SWIP (background shaping), PCR (refinement),
SPC (spectral capacity), WSD (sync), CMC (consistency), GAC (gradient),
STA (stability).

**Prediction**: improvement(all) ≈ Σ improvement(each), with |residual| < 20%.

## H10: Scaling Behavior Follows Power Law

**Hypothesis**: Downstream performance will follow a power law in model size:
performance ≈ a * (params)^b with b ∈ [0.3, 0.5].

**Rationale**: Standard neural scaling laws (Kaplan et al., 2020) with
JEPA-specific modifications from workspace dimension.

**Prediction**: Linear probe accuracy scales as params^0.35 ± 0.1.

**Ablation**: `config/scaling/` configs (30M, 100M, 140M, 300M).

## H11: RDC Prevents Loss of Exogenous Features

**Hypothesis**: RDC will reduce the orthogonal drift ratio (||Δz_⊥|| / ||Δz||)
by ≥50% compared to standard JEPA, preserving control-relevant features.

**Rationale**: The Drift Compensation Bound guarantees:
||z_{⊥,T} - z_{⊥,0}|| ≤ ε(1-η_rdc)^T · T/√k
With η_rdc > 0, orthogonal drift is exponentially bounded.

**Prediction**: drift_ratio with RDC < 0.5 * drift_ratio without RDC.
Orthogonal features will have higher variance (not collapsed).

**Ablation**: `rdc_on.yaml` vs `no_rdc.yaml`.
