# Text-Span JEPA

Latent Predictive Learning for Language Representations

[![Tests](https://github.com/fanat503/Text-Span-JEPA/actions/workflows/ci.yaml/badge.svg)](https://github.com/fanat503/Text-Span-JEPA/actions/workflows/ci.yaml)

---

## Installation

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Training

```bash
# Single GPU
python -m src.train --fname configs/base.yaml

# Debug (tiny model)
bash scripts/debug.sh

# Kaggle T4
bash scripts/train_kaggle.sh
```

Config files are in `configs/`. All hyperparameters are specified there — no command-line overrides needed (following I-JEPA).

To resume from a checkpoint, set `meta.load_checkpoint: true` in the config YAML.

## Method

Text-Span JEPA extends the Joint-Embedding Predictive Architecture (JEPA) framework to bidirectional text. The model predicts latent representations — not tokens — at masked spans and future positions.

Key components:

- **Span-level latent prediction**: contiguous block masking in latent space (adapted from I-JEPA multiblock masking for 1D text)
- **Future latent prediction**: multi-offset queries predict h[t+d] from h[t] (following NextLat)
- **Iterative refinement**: multiple predictor passes in latent space — the encoder is not re-run
- **Scheduled EMA τ**: linear ramp from 0.996 → 1.0, same formula as I-JEPA: `τ(i) = τ_start + i·(τ_end − τ_start) / total_steps`
- **VICReg collapse prevention**: variance margin + covariance decorrelation + target centering (data2vec)

## Code Structure

```
├── src/models/       encoder, predictor, decoder, collapse, jepa
├── src/masks/        span masking with curriculum
├── src/datasets/     WikiText-103 / BookCorpus for Kaggle
├── src/utils/        schedulers, logging (from I-JEPA)
├── src/eval/         linear probe, future-token probe, geometry metrics
├── baselines/        data2vec (from official fairseq), MLM
├── configs/          base.yaml, debug.yaml, kaggle.yaml
├── tests/            56 tests
└── scripts/          train_kaggle.sh, debug.sh
```

The main differences between algorithms can be understood via their `compute_loss()` functions (following NextLat).

## Algorithms

| File | Algorithm |
|------|-----------|
| `src/models/jepa.py` | Text-Span JEPA (span + future latent prediction) |
| `baselines/data2vec_baseline.py` | data2vec (token-level regression, from fairseq) |
| `baselines/mlm_baseline.py` | MLM (BERT-style token reconstruction) |

## Loss

```
L = λ_span · L_span + λ_future · L_future + λ_dec · L_decoder
  + λ_var · L_variance + λ_cov · L_covariance
```

- `L_span`: smooth L1 between predicted and target latent at masked positions (I-JEPA)
- `L_future`: multi-offset future latent prediction with warmup from 0 (NextLat)
- `L_decoder`: cross-entropy on tied decoder — anti-collapse grounding
- `L_variance`, `L_covariance`: VICReg collapse prevention

## Code Provenance

Patterns reproduced with variable names changed, logic preserved:

| Source | Pattern | File |
|--------|---------|------|
| I-JEPA `src/train.py` | momentum_scheduler, smooth_l1_loss, layer_norm on targets, train_step closure | `src/train.py` |
| I-JEPA `src/helper.py` | init_opt param_groups, trunc_normal_ init, depth-wise rescaling | `src/models/encoder.py` |
| I-JEPA `src/utils/logging.py` | AverageMeter, CSVLogger, grad_logger | `src/utils/logging.py` |
| I-JEPA `src/utils/schedulers.py` | WarmupCosineSchedule, CosineWDSchedule | `src/utils/schedulers.py` |
| data2vec `data2vec_text.py:58` | `get_annealed_rate()` | `baselines/data2vec_baseline.py` |
| data2vec `data2vec_text.py:301` | regression_head: Linear→GELU→Linear | `baselines/data2vec_baseline.py` |
| data2vec `data2vec_text.py:474` | loss: smooth_l1/mse, scale=1/√dim | `baselines/data2vec_baseline.py` |
| NextLat `model_base.py` | compute_hidden_state_rank: effective_rank, exception→0 | `src/models/collapse.py` |

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
