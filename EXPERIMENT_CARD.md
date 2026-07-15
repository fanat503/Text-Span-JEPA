# Experiment Card

## Hypothesis

Text-Span JEPA learns useful text representations more sample-efficiently than token reconstruction baselines.

## Required comparisons

- Text-Span JEPA
- MLM baseline
- GPT hidden-state baseline, if available
- ablations: no decoder, no covariance, different mask ratios

## Primary metrics

- representation probe accuracy
- retrieval score
- decoder accuracy
- collapse metrics
- quality-vs-compute curves

## Run table

| Run | Config | Data | Status | Notes |
|---|---|---|---|---|
| JEPA | configs/small.json | | pending | |
