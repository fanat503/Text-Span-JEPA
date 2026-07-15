# Model Card: Text-Span JEPA

## Intended use

Research on self-supervised text representation learning via latent masked-span prediction.

## Not intended use

This is not an autoregressive LLM and should not be evaluated as a text generator without additional decoding/fine-tuning.

## Core objective

Online encoder predicts EMA target encoder latent states at masked span positions. A tied decoder provides token-level grounding as an auxiliary signal.

## Key safety/quality metrics

- prediction loss
- decoder loss / accuracy
- online and target representation std
- variance / covariance regularization
- representation anisotropy / participation ratio
- downstream probes / retrieval

## Known limitations

- No perplexity.
- No generation.
- Requires baselines for claims.
- JEPA loss alone is not a downstream quality metric.
