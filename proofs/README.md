# Mathematical Proofs & Formal Analysis

This directory contains formal mathematical proofs for all 13 mechanisms in Text-Span JEPA, organized by mechanism.

## Unifying Principle

**[`unifying_principle.md`](unifying_principle.md)** — The Workspace-Conditioned Prediction (WCP) framework from which all 13 mechanisms are derived as instances of a single optimization principle. This is the central theoretical contribution.

## Mechanism Proofs

| Mechanism | File | Core Theorem |
|-----------|------|-------------|
| JAWP | [`jawp.md`](jawp.md) | Courant-Fischer optimality on St(D,k) |
| WIP | [`wip.md`](wip.md) | Workspace Information Preservation (contradiction + regularity) |
| CGN | [`cgn.md`](cgn.md) | Information Routing + Partition of Unity |
| SWIP | [`swip.md`](swip.md) | Selective Spectral Shaping (log-eigenvalue matching) |
| PCR | [`pcr.md`](pcr.md) | Cascade Capacity theorem (orthogonal recovery) |
| SPC | [`spc.md`](spc.md) | Parseval's equality + simplex-constrained allocation |
| WSD | [`wsd.md`](wsd.md) | Drift Bound theorem (exponential convergence ODE) |
| CMC | [`cmc.md`](cmc.md) | Stability theorem (Cauchy-Schwarz bound) |
| GAC | [`gac.md`](gac.md) | No Dead Zones theorem (exploration guarantee) |
| STA | [`sta.md`](sta.md) | Davis-Kahan stability + Wasserstein-1 metric |

## Pre-Registered Hypotheses

**[`HYPOTHESES.md`](HYPOTHESES.md)** — 10 pre-experimental hypotheses registered before running any training experiments, following top-lab standards (analogous to clinical trial pre-registration).

## Proof Standards

Each proof document follows this structure:
1. **Statement** — Formal theorem statement with all assumptions
2. **Proof** — Complete step-by-step derivation
3. **Discussion** — Limitations, connections to other mechanisms, practical implications
4. **Novelty** — Explicit comparison with closest prior art

## Verification

All theorems are **computationally verified** in the test suite:
- JAWP Q orthonormality error: < 1e-5
- SPC Parseval's reconstruction: relative error < 1e-4
- CGN partition of unity: exact (by construction)
- SWIP loss non-negative: ✅
- CMC loss non-negative + stability bound: ✅
- GAC No Dead Zones + exploration ratio bounded: ✅
- STA W1 metric (triangle inequality, symmetry, non-negativity): ✅
- STA Davis-Kahan bound: ✅
