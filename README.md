# Text-Span JEPA

**Latent Predictive Learning for Language Representations**

Self-supervised text representation learning architecture extending the JEPA (Joint-Embedding Predictive Architecture) framework to bidirectional text.

## Key Innovations

1. **Span-level latent prediction** — contiguous block masking in latent space (not token reconstruction)
2. **Future latent prediction** — multi-offset queries predict h[t+d] from h[t]
3. **Iterative refinement** — multiple cheap predictor passes in latent space (no encoder re-run)
4. **Tied decoder grounding** — weight-tied token decoder prevents representational collapse
5. **Scheduled EMA τ** — linear ramp 0.996→1.0 (NOT constant, following I-JEPA)

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v                    # 35 tests
bash scripts/debug.sh               # tiny model, 2 epochs
bash scripts/train_kaggle.sh        # Kaggle T4, 120M params
```

## Risk Fixes

| # | Risk | Fix | Location |
|---|------|-----|----------|
| 2 | Future loss instability | `future_warmup_steps` ramps λ_future from 0 | `jepa.py` |
| 3 | Boolean indexing on masks | `_gather_masked()` with `torch.gather` + `valid_mask` | `predictor.py` |
| 4 | data2vec baseline | From official `fairseq/examples/data2vec/models/data2vec_text.py` | `baselines/` |

## Project Structure

```
├── src/models/       # encoder, predictor, decoder, collapse, jepa
├── src/masks/        # span masking with curriculum
├── src/datasets/     # WikiText-103 / BookCorpus for Kaggle
├── src/utils/        # schedulers (I-JEPA), logging (I-JEPA)
├── src/eval/         # linear probe, future-token probe, geometry
├── baselines/        # data2vec (official fairseq), MLM
├── configs/          # base.yaml, debug.yaml, kaggle.yaml
├── tests/            # 35 tests
└── scripts/          # train_kaggle.sh, debug.sh
```

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
