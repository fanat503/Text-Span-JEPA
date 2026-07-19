# Text-Span JEPA

**Latent Predictive Learning for Language Representations**

[![Tests](https://github.com/fanat503/Text-Span-JEPA/actions/workflows/ci.yaml/badge.svg)](https://github.com/fanat503/Text-Span-JEPA/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Self-supervised text representation learning architecture extending the **JEPA (Joint-Embedding Predictive Architecture)** framework [LeCun, 2022] to bidirectional text. Predicts latent representations of masked spans and future positions — no token reconstruction required.

---

## Key Innovations

| # | Innovation | Motivation | Source |
|---|-----------|------------|--------|
| 1 | **Span-level latent prediction** | Predict contiguous blocks in latent space, not tokens | I-JEPA [Assran et al., CVPR 2023] adapted for text |
| 2 | **Future latent prediction** | Multi-offset queries predict h[t+d] from h[t] | NextLat [Teoh et al., 2025] |
| 3 | **Iterative refinement** | Multiple cheap predictor passes (no encoder re-run) | Diffusion-inspired latent refinement |
| 4 | **Tied decoder grounding** | Weight-tied token decoder prevents representational collapse | Anti-collapse auxiliary task |
| 5 | **Scheduled EMA τ** | Linear ramp 0.996→1.0 (NOT constant) | I-JEPA momentum schedule |
| 6 | **VICReg collapse prevention** | Variance margin + covariance decorrelation + target centering | VICReg [Bardes et al., ICLR 2022], C-JEPA [NeurIPS 2024] |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Masked Input ──→ [Online Encoder] ──→ h_online        │
│                          │                              │
│                     [Predictor]                         │
│                    ╱            ╲                        │
│         span predictions    future predictions           │
│                    ╲            ╱                        │
│                     ╲          ╱                         │
│  Original Input ──→ [Target Encoder (EMA)] ──→ h_target │
│                          │                              │
│              [Tied Decoder] (auxiliary grounding)        │
│              [VICReg Collapse Prevention]                │
│              [Target Centering (data2vec)]               │
└─────────────────────────────────────────────────────────┘
```

### Parameter Budget (~120M, Base Config)

| Component | Parameters | % of Trainable |
|-----------|-----------|----------------|
| Encoder (online) | 124,082,688 | 84.2% |
| Predictor | 11,437,057 | 8.3% |
| Decoder (tied) | 2,360,832 | 1.7% |
| Target encoder (frozen EMA) | 124,082,688 | — |
| **Total trainable** | **137,880,577** | **100%** |
| Non-embedding | 98,853,889 | 71.8% |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run 51 tests
pytest tests/ -v

# Tiny model, 2 epochs (debug)
bash scripts/debug.sh

# Kaggle T4/P100, 120M params
bash scripts/train_kaggle.sh
```

## Training

### Configuration

| Config | Params | embed_dim | depth | GPU | `configs/` |
|--------|--------|-----------|-------|-----|-------------|
| Debug | ~0.5M | 128 | 2 | CPU | `debug.yaml` |
| Base | ~120M | 768 | 12 | Kaggle T4 | `base.yaml` |
| Kaggle | ~120M | 768 | 12 | Kaggle T4 | `kaggle.yaml` |

### Key Hyperparameters (Base)

```yaml
model:
  embed_dim: 768
  encoder_depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  predictor_embed_dim: 384
  predictor_depth: 6
  future_offsets: [1, 4, 16]
  num_refine_steps: 3
  ema_tau_start: 0.996
  ema_tau_end: 1.0        # Scheduled, NOT constant

optimization:
  lr: 0.001
  weight_decay: 0.04
  warmup: 10              # epochs
  ema: [0.996, 1.0]      # Linear ramp over total steps
```

### EMA Schedule (Critical)

Following I-JEPA exactly:

```
τ(i) = τ_start + i × (τ_end − τ_start) / total_steps
```

- Early training: lower τ → target encoder updates faster → adapts to changing online encoder
- Late training: higher τ → target encoder stabilizes → predictions have consistent targets

## Loss Components

```
L_total = λ_span · L_span + λ_future · L_future + λ_dec · L_decoder
        + λ_var · L_variance + λ_cov · L_covariance
```

| Loss | Weight | Purpose | Pattern From |
|------|--------|---------|-------------|
| `L_span` | 1.0 | Smooth L1 between predicted and target latent at masked positions | I-JEPA |
| `L_future` | 0.5 | Multi-offset future latent prediction (warmup from 0) | NextLat |
| `L_decoder` | 0.1 | Cross-entropy on tied decoder (anti-collapse grounding) | — |
| `L_variance` | 0.1 | Per-dimension variance ≥ margin | VICReg |
| `L_covariance` | 0.04 | Off-diagonal covariance penalty | VICReg |

## Risk Fixes

| # | Risk | Fix | File |
|---|------|-----|------|
| 2 | Future loss instability (early target encoder is noisy) | `future_warmup_steps` ramps λ_future from 0→λ | `jepa.py` |
| 3 | Boolean indexing on masks → irregular shapes | `_gather_masked()` with `torch.gather` + `valid_mask` | `predictor.py` |
| 4 | data2vec baseline fidelity | From official `fairseq/examples/data2vec/models/data2vec_text.py` | `baselines/` |

## Collapse Diagnostics

Following NextLat `compute_hidden_state_rank` pattern:

| Metric | Formula | Normal Range | Zero Input |
|--------|---------|-------------|-----------|
| Effective rank | exp(−Σ p·log(p)), p=S/S.sum() | 10–30 | 0.0 |
| Participation ratio | (ΣS)² / Σ(S²) | 5–20 | 0.0 |
| Condition number | S[0] / S[-1] | 10–1000 | ∞ |
| Numerical rank | matrix_rank(atol=1e-3) | 20–64 | 0 |
| Rank utilization | num_rank / min(N,D) | 0.3–0.9 | 0.0 |
| Coherence | max|off-diag(cov)| | 0.01–0.5 | 0.0 |

Exception handling follows NextLat: try/except returns 0.0 (for rank metrics) or inf (for condition number), never crashes.

## Code Patterns from Reference Papers

| Source | File | Pattern | Our File |
|--------|------|---------|----------|
| I-JEPA | `src/train.py` | momentum_scheduler, loss_fn=smooth_l1_loss, layer_norm(h,(D,)) | `src/train.py` |
| I-JEPA | `src/helper.py` | init_opt param_groups, init_weights trunc_normal_, depth-wise rescaling | `src/models/encoder.py` |
| I-JEPA | `src/utils/logging.py` | AverageMeter, CSVLogger, grad_logger | `src/utils/logging.py` |
| I-JEPA | `src/utils/schedulers.py` | WarmupCosineSchedule, CosineWDSchedule | `src/utils/schedulers.py` |
| data2vec | `data2vec_text.py:58` | `get_annealed_rate(start, end, curr, total)` | `baselines/data2vec_baseline.py` |
| data2vec | `data2vec_text.py:301` | regression_head: Linear→GELU→Linear (head_layers=2) | `baselines/data2vec_baseline.py` |
| data2vec | `data2vec_text.py:474` | loss: smooth_l1/mse with beta, scale=1/√dim | `baselines/data2vec_baseline.py` |
| NextLat | `models/model_base.py` | compute_hidden_state_rank: effective_rank via Shannon entropy, exception→0 | `src/models/collapse.py` |
| VICReg | — | variance margin + covariance off-diagonal penalty | `src/models/collapse.py` |

## Project Structure

```
├── src/models/        # encoder, predictor, decoder, collapse, jepa
├── src/masks/         # span masking with curriculum
├── src/datasets/      # WikiText-103 / BookCorpus for Kaggle
├── src/utils/         # schedulers (I-JEPA), logging (I-JEPA)
├── src/eval/          # linear probe, future-token probe, geometry
├── baselines/         # data2vec (official fairseq), MLM
├── configs/           # base.yaml, debug.yaml, kaggle.yaml
├── tests/             # 51 tests (51/51 passing)
├── scripts/           # train_kaggle.sh, debug.sh
├── .github/workflows/ # CI: pytest + black
├── .gitattributes     # linguist overrides (NextLat pattern)
├── pyproject.toml     # build config + ruff + black
├── requirements.txt   # pip dependencies
└── LICENSE            # MIT
```

## Checkpoints

Training saves two checkpoint files per epoch:
- `checkpoint-latest.pth.tar` — overwritten each epoch (for resumption)
- `checkpoint-ep{N}.pth.tar` — kept per epoch

Checkpoint contents (I-JEPA pattern):
```python
{
    'encoder':         encoder.state_dict(),
    'predictor':       predictor.state_dict(),
    'target_encoder':  target_encoder.state_dict(),
    'decoder':         decoder.state_dict(),
    'opt':             optimizer.state_dict(),
    'scaler':          scaler.state_dict(),
    'epoch':           epoch,
    'global_step':     global_step,
    'loss':            loss_meter.avg,
}
```

Resume training: set `meta.load_checkpoint: true` in config YAML.

## Citation

```bibtex
@article{textspanjepa2026,
  title={Text-Span JEPA: Latent Predictive Learning for Language Representations},
  author={Text-Span JEPA Authors},
  year={2026}
}
```

## References

- Assran, M. et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA), CVPR 2023
- Baevski, A. et al., "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language" (data2vec 2.0), ICML 2022
- Bardes, A. et al., "VicReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (VICReg), ICLR 2022
- Teoh, J. et al., "Next-Latent Prediction Transformers Learn Compact World Models" (NextLat), 2025
- LeCun, Y., "A Path Towards Autonomous Machine Intelligence", 2022

## License

MIT
