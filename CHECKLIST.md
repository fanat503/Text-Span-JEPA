# NeurIPS readiness checklist

## Must-have before making claims

- [ ] Dataset card filled.
- [ ] Train/val token files validated.
- [ ] JEPA training curve logged.
- [ ] MLM baseline trained on same data/mask budget.
- [ ] Representation diagnostics computed.
- [ ] Decoder accuracy reported.
- [ ] Collapse metrics reported: online_std, target_std, variance/covariance.
- [ ] Ablation: decoder_weight=0.
- [ ] Ablation: covariance_weight=0.
- [ ] Compute report: wall-clock and tokens/sec.

## Strong paper additions

- [ ] Linear probe on frozen representations.
- [ ] Retrieval benchmark.
- [ ] Mask-ratio sensitivity.
- [ ] Span-length sensitivity.
- [ ] Quality-vs-compute curves vs MLM/GPT encoder.
