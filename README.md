# Text-Span JEPA

Standalone research repo for JEPA-style text representation learning.

## Core idea

The online encoder sees masked spans. The EMA target encoder sees the original text. A predictor maps online latents to target latents at masked positions. A tied decoder grounds predicted latents to token ids. Variance/covariance losses guard against collapse.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/make_dummy_data.py
python scripts/audit.py
python scripts/validate_data.py --config configs/small.json
python train.py --config configs/small.json
python eval.py --checkpoint runs/jepa_small/best_val_jepa_small.pt --out runs/jepa_small/eval.json
```

## Generate ablation configs

```bash
python scripts/make_configs.py --base configs/small.json --out-dir configs/ablations
```

## Representation diagnostics

```bash
python scripts/analyze_representations.py --checkpoint runs/jepa_small/best_val_jepa_small.pt --out reps.json
python scripts/retrieval_eval.py --checkpoint runs/jepa_small/best_val_jepa_small.pt --out retrieval.json
python scripts/linear_probe.py --checkpoint runs/jepa_small/best_val_jepa_small.pt --out probe.json
```

## Important caveat

This is not an autoregressive LLM. Do not compare JEPA loss to perplexity. Compare representations, retrieval, probes, collapse metrics, and quality-vs-compute.

## NextLat-style diagnostics

Future-token linear probes:

```bash
python scripts/train_future_probe.py --checkpoint runs/jepa_small/best_val_jepa_small.pt --out runs/jepa_small/future_probe.json
```

Effective latent rank:

```bash
python scripts/effective_rank.py --checkpoint runs/jepa_small/best_val_jepa_small.pt --out runs/jepa_small/effective_rank.json
```

Materialize config:

```bash
python scripts/materialize_config.py --config configs/small.json --out runs/materialized_small.json
```
