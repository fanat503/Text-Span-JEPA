# Reviewer attacks

1. Not an LLM: true; evaluate representation quality, not perplexity.
2. Latent loss meaningless: tied token decoder grounds predictions.
3. Collapse: EMA target + predictor + variance/covariance losses + std metrics.
4. Just MLM: decoder is auxiliary; primary loss is latent EMA target prediction.
5. No baselines: add MLM/GPT encoder baselines before submission.
6. Decoder dominates: report decoder_weight ablation.
7. Compute not lower: report quality-vs-compute, not just final metric.
8. Mask policy arbitrary: report span/mask sensitivity.
9. Dataset leakage: fixed dataset card and fingerprints required.
10. No downstream: add probes/retrieval/entity-binding.
11. EMA leakage: target is stop-gradient and never input to online.
12. No generation: position as representation learning.
13. Collapse hidden by decoder: report variance/covariance/PR.
14. Metrics cherry-picked: pre-register metrics.
15. Novelty: emphasize EMA latent span prediction + tied grounding + collapse guards.
