# text-span-jepa

predict latent representations at masked spans + future positions.

not token reconstruction. not contrastive. just latent prediction in representation space — following LeCun's JEPA proposal.

## why

reconstruction (MLM, MAE) forces the model to waste capacity on low-level details. contrastive (SimCSE, CLIP) needs negative sampling and augmentations. JEPA: skip both. predict in latent space, let the encoder learn what matters.

the twist for text: **span-level masking** (not random tokens) + **future latent prediction** (predict where the sequence is going). spans force broader context. future prediction gives the model a reason to encode directionality.

## how it works

3 components:

**encoder** — standard bidirectional transformer. shared architecture between online and target. target encoder is EMA copy with scheduled τ ramping from 0.996→1.0 (constant τ doesn't work, I-JEPA showed this).

**predictor** — narrow transformer that predicts target latents at masked positions. two modes:
- span: mask contiguous blocks, predict their latents. iterative refinement (multiple cheap passes, no encoder re-run). the predictor does N lightweight forward passes, each time getting a slightly better estimate. like "thinking" in latent space.
- future: predict h[t+d] from h[t] with learned offset queries. single pass, no refinement — it's a simpler task.

**decoder** — weight-tied projection to token space. auxiliary. if latents collapse to uniform, the decoder can't predict different tokens, so it's an implicit anti-collapse signal.

collapse prevention: VICReg (variance margin + covariance decorrelation) + data2vec target centering. these are necessary — JEPA models can silently collapse (loss goes down, representations become useless).

## setup

```
pip install -r requirements.txt
pip install -e .
```

python 3.9+, pytorch 2.0+. trains on wikitext-103 by default.

## train

```
python -m src.train --fname configs/small-100m.yaml
```

all config in YAML. no CLI overrides — same as I-JEPA. resume with `meta.load_checkpoint: true`.

configs: `debug.yaml` (sanity), `small-100m.yaml` (~90M, 16GB), `base-200m.yaml` (~140M, 24GB), `large-350m.yaml` (~280M, 40GB), `kaggle.yaml` (tuned for T4).

## loss

```
L = λ_span · smooth_l1(z_pred, z_target)
  + λ_future · smooth_l1(z_future, z_target_future)
  + λ_dec · CE(logits, tokens)
  + λ_var · max(0, margin - √var)
  + λ_cov · off_diag(cov)²
```

future loss has warmup from 0 (early target encoder is unstable, raw future loss injects noise).

## diagnostics

**you cannot debug a JEPA by watching loss go down.** loss can decrease while representations collapse. you need auxiliary metrics:

- `effective_rank` (NextLat) — shannon entropy of SVD spectrum. should stay >5, collapse → 1
- `participation_ratio` (Roy & Vetterli) — effective dimensionality. >1 is alive, 1 = rank-1 collapse
- `sv_entropy` (I-JEPA) — normalized entropy of singular values. 1 = uniform spectrum, 0 = single component
- `svd_sharpness` (C-JEPA/BYOL) — S[0]²/ΣS[i]². 1 = rank-1 collapse
- `alpha_norm` (LeCun 2022) — power-law exponent of singular value spectrum. higher = concentrated info
- `intrinsic_dim` (Ansuini et al.) — two-nearest-neighbor ID estimate. lower = more structured
- `collapsed_dim_ratio` (I-JEPA) — fraction of near-zero-variance dimensions. 0 is healthy
- `mean_pairwise_cosine` (DINOv2) — intra-batch similarity. high = collapse
- `representation_stability` (I-JEPA) — cosine between consecutive target updates. >0.99 is good
- `cross_corr_redundancy` (Barlow Twins) — off-diagonal of cross-correlation. near 0 is healthy
- `cka_linear` / `cka_rbf` (Kornblith et al.) — online-target similarity. tracks alignment
- `condition_number`, `numerical_rank`, `coherence`, `rank_utilization` (NextLat) — standard SVD diagnostics

all follow NextLat's exception pattern: SVD failure → return 0.0 (or inf for condition_number). never crash the training loop.

## structure

```
src/models/       encoder, predictor, decoder, collapse diagnostics, main model
src/masks/        span masking with curriculum
src/datasets/     wikitext-103 / bookcorpus loader (kaggle-compatible)
src/utils/        schedulers, logging (from I-JEPA)
src/eval/         linear probe, future-token probe, geometry metrics
baselines/        data2vec (from official fairseq), MLM
configs/          per-size YAML configs
tests/            pytest suite
```

## provenance

code patterns from reference implementations (variable names changed):

- I-JEPA: momentum scheduler, param groups, smooth_l1 loss, layer_norm on targets, trunc_normal_ init, depth-wise rescaling, AverageMeter, CSVLogger, grad_logger
- data2vec: get_annealed_rate, regression head (Linear→GELU→Linear), loss scaling by 1/√dim, target centering
- NextLat: compute_hidden_state_rank with effective_rank via shannon entropy, exception→0.0, rank_utilization
- VICReg: variance margin, off-diagonal covariance penalty
- Barlow Twins: cross-correlation redundancy
- Kornblith et al.: linear CKA via HSIC, RBF CKA

## cite

```bibtex
@article{textspanjepa2026,
  title={Text-Span JEPA: Latent Predictive Learning for Language Representations},
  author={Text-Span JEPA Authors},
  year={2026}
}
```

## license

apache 2.0
