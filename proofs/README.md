# Proofs Directory — Formal Mathematical Grounding for Text-Span JEPA

This directory contains formal proofs, hypotheses, and verification results
for all 13 novel mechanisms in Text-Span JEPA. Each mechanism has:

1. **Problem statement** — what issue it addresses
2. **Formal theorem** — the mathematical guarantee
3. **Proof** — step-by-step derivation
4. **Assumptions/conditions** — when the theorem holds
5. **Verification** — how tests confirm the theorem computationally

Organization follows top-lab standards (Meta FAIR, Google DeepMind, Microsoft Research).

## Mechanism Index

| # | Mechanism | File | Core Theorem | Key Reference |
|---|-----------|------|--------------|---------------|
| 1 | JAWP | [jajwp.md](jajwp.md) | Courant-Fischer optimality | Golub & Van Loan, Thm 8.1.2 |
| 2 | WIP | [wip.md](wip.md) | Information preservation (by contradiction) | Pendharkar et al., 2026 |
| 3 | Spectral Gap | [spectral_gap.md](spectral_gap.md) | Marchenko-Pastur detection | Marchenko & Pastur, 1967 |
| 4 | Grassmann | [grassmann.md](grassmann.md) | Fiber projection convergence | Absil et al., 2008, Thm 7.4.2 |
| 5 | Predictive Rank | [predictive_rank.md](predictive_rank.md) | Log-determinant barrier | Vershynin, 2018 |
| 6 | CGN | [cgn.md](cgn.md) | Partition of unity + sufficient statistic | Bardes et al., ICLR 2022 |
| 7 | SWIP | [swip.md](swip.md) | Log-eigenvalue matching | Zbontar et al., ICML 2021 |
| 8 | PCR | [pcr.md](pcr.md) | Cascade capacity theorem | This work |
| 9 | SPC | [spc.md](spc.md) | Parseval's theorem / Info-proportional | Parseval, 1806 |
| 10 | WSD | [wsd.md](wsd.md) | Drift bound (ODE) | This work |
| 11 | CMC | [cmc.md](cmc.md) | Stability theorem (Cauchy-Schwarz) | Cauchy-Schwarz inequality |
| 12 | GAC | [gac.md](gac.md) | No Dead Zones theorem | This work |
| 13 | STA | [sta.md](sta.md) | Davis-Kahan + Wasserstein | Davis & Kahan, 1970 |

## Verification Protocol

Each proof is verified by:
1. **Unit tests** — shape, non-negativity, edge cases
2. **Theorem tests** — direct computational verification
3. **Integration tests** — mechanism works in full model
4. **Convergence tests** — mechanism converges under gradient descent

All verification results are logged in `tests/` directory.

## How to Add a New Mechanism

1. Create `proofs/new_mechanism.md` with problem, theorem, proof
2. Implement in `src/models/new_mechanism.py`
3. Add theorem verification in `tests/test_new_mechanism.py`
4. Integrate in `src/models/jepa.py` and `src/models/mechanisms.py`
5. Add checkpoint save/restore in `src/train.py`
6. Add config fields in `defaults.yaml`
7. Update this README and `NOVELTY_AUDIT.md`
