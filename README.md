# Text-Span JEPA

Latent predictive learning for language representations.

The core idea is simple: instead of reconstructing tokens, predict latent representations at masked spans and future positions. This follows the JEPA framework (LeCun 2022) — the predictor operates in representation space, not pixel/token space.

## Setup

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

Requires Python 3.9+ and PyTorch 2.0+. We train on WikiText-103 out of the box; swap the dataset loader for anything else.

## Training

```bash
python -m src.train --fname configs/small-100m.yaml
```

Configs live in `configs/`. No CLI overrides — everything is in the YAML (same as I-JEPA).

To resume: set `meta.load_checkpoint: true` in the config. It picks up `checkpoint-latest.pth.tar` from the log directory.

### Available configs

| Config | Params | GPU memory | Epochs | Notes |
|--------|--------|-----------|--------|-------|
| `debug.yaml` | ~0.5M | CPU | 2 | Sanity check |
| `small-100m.yaml` | ~90M | 16 GB | 30 | V100 / RTX 3090 |
| `base-200m.yaml` | ~140M | 24 GB | 50 | A5000 / L4 |
| `large-350m.yaml` | ~280M | 40 GB | 50 | A100 |
| `kaggle.yaml` | ~140M | 16 GB (T4) | 50 | Kaggle-specific paths |

All configs target 2B training tokens by default. Adjust `data.batch_size` and `optimization.epochs` to match your hardware.

## What's going on

The architecture has three moving parts:

**Encoder** — bidirectional Transformer (same as I-JEPA's ViT, but for 1D text). Shared between online and target. The target encoder is an EMA copy with scheduled τ ramping from 0.996→1.0 — this is important, constant τ doesn't work well (I-JEPA's momentum schedule).

**Predictor** — narrow Transformer that takes encoder output, inserts mask tokens at span positions, and predicts the target latent. Two prediction modes run simultaneously:
- *Span prediction*: mask contiguous blocks, predict their latents. The predictor does iterative refinement (multiple cheap passes, no encoder re-run).
- *Future prediction*: predict h[t+d] from h[t] with learned offset queries. Lightweight — no iterative refinement here, just a single forward pass.

**Decoder** — tiny weight-tied projection back to token space. This is auxiliary: if representations collapse to a uniform vector, the decoder can't predict different tokens, so it acts as an anti-collapse signal.

**Collapse prevention** — VICReg terms (variance margin + covariance decorrelation) plus data2vec-style target centering. These are the standard safeguards; without them, JEPA models can silently collapse (loss goes down but representations become useless).

## Loss

```
L = λ_span · smooth_l1(z_pred, z_target) 
  + λ_future · smooth_l1(z_future, z_target_future)
  + λ_dec · cross_entropy(logits, tokens)
  + λ_var · max(0, margin - sqrt(var)) 
  + λ_cov · off_diag(cov)²
```

Future loss has a warmup: λ_future ramps from 0 over the first N steps. Without this, the target encoder is too unstable early on and the future prediction loss injects noise.

## Code structure

```
src/models/       encoder, predictor, decoder, collapse diagnostics, main model
src/masks/        span masking with curriculum
src/datasets/     WikiText-103 / BookCorpus loader (Kaggle-compatible)
src/utils/        schedulers and logging (from I-JEPA)
src/eval/         linear probe, future-token probe, geometry metrics
baselines/        data2vec (from official fairseq), MLM
configs/          per-size YAML configs
tests/            64 tests
```

The differences between Text-Span JEPA, data2vec, and MLM are best understood by reading their respective `compute_loss()` functions (this follows NextLat's convention).

## Diagnostics

One thing the papers don't emphasize enough: you cannot debug a JEPA by watching loss decrease. You need auxiliary diagnostics. We compute these every step:

| Metric | Source | What to look for |
|--------|--------|-----------------|
| effective_rank | NextLat | Should stay >5, collapse → 1 |
| participation_ratio | Roy & Vetterli | >1, collapse → 1 |
| collapsed_dim_ratio | lang-jepa | Near 0 is healthy, →1 is collapse |
| rank_utilization | NextLat | 0.3–0.9 healthy |
| condition_number | NextLat | 10–1000 healthy, ∞ is degenerate |
| coherence | NextLat | Low is healthy |
| cross_corr_redundancy | Barlow Twins | Near 0 is healthy |
| cka_linear | Kornblith et al. | Online-target similarity |
| online_std | I-JEPA | Near 0 → collapse |

The code follows NextLat's exception pattern: if SVD or any metric computation fails, return 0.0 (or inf for condition_number). Never crash the training loop.

## Code provenance

Patterns from reference implementations, with names changed:

- I-JEPA: momentum scheduler, param groups, smooth_l1 loss, layer_norm on targets, trunc_normal_ init, depth-wise rescaling, AverageMeter, CSVLogger, grad_logger, WarmupCosineSchedule, CosineWDSchedule
- data2vec: `get_annealed_rate`, regression head (Linear→GELU→Linear), loss scaling by 1/√dim, target centering
- NextLat: `compute_hidden_state_rank` with effective_rank via Shannon entropy, exception→0.0 pattern, rank_utilization
- VICReg: variance margin, off-diagonal covariance penalty
- Barlow Twins: cross-correlation redundancy
- Kornblith et al.: linear CKA

## Citation

```bibtex
@article{textspanjepa2026,
  title={Text-Span JEPA: Latent Predictive Learning for Language Representations},
  author={Text-Span JEPA Authors},
  year={2026}
}
```

## License

MIT
