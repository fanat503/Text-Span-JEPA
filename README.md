text-span-jepa
==============

latent prediction at masked spans + future positions.

not token reconstruction. not contrastive. predict in latent space — that's the whole point of JEPA (LeCun 2022). the encoder learns what matters because it never has to waste capacity on low-level token details.

the twist for text: span-level masking (not random tokens) forces the model to use broader context. future latent prediction (where h[t+d] is going) gives it a reason to encode directionality. these two things together are what make this different from I-JEPA (which does images) and data2vec (which does token-level).

---

setup
-----

```
pip install -r requirements.txt
pip install -e .
```

python 3.9+, pytorch 2.0+. trains on wikitext-103 out of the box.

training
--------

```
python -m src.train --fname configs/small-100m.yaml
```

everything in the YAML, nothing on the CLI. same convention as I-JEPA.

resume: set `meta.load_checkpoint: true` in the config. picks up from `checkpoint-latest.pth.tar`.

configs: `debug.yaml` (sanity), `small-100m.yaml` (~90M, 16GB), `base-200m.yaml` (~140M, 24GB), `large-350m.yaml` (~280M, 40GB), `kaggle.yaml` (tuned for T4).

architecture
------------

three components. nothing else.

**encoder** — bidirectional transformer. same architecture for online and target. target encoder is EMA copy with scheduled τ = 0.996 → 1.0 (constant τ doesn't work, I-JEPA showed this and we confirmed it).

**predictor** — narrow transformer. takes encoder output, inserts mask tokens at span positions, predicts target latent. two modes:
- span: mask contiguous blocks, predict their latents. iterative refinement (N cheap passes without re-running the encoder). each pass gets a slightly better estimate — like "thinking" in latent space.
- future: predict h[t+d] from h[t] with learned offset queries. single pass, no refinement. it's a simpler task.

**decoder** — weight-tied projection to token space. auxiliary. if latents collapse to a uniform vector, the decoder can't predict different tokens, so it acts as an implicit anti-collapse signal. doesn't dominate training.

collapse prevention: VICReg (variance margin + covariance decorrelation) + data2vec target centering. these are not optional — JEPA models silently collapse (loss goes down, representations become useless, you wouldn't know unless you check).

loss
----

```
L = λ_span · smooth_l1(z_pred, z_target)
  + λ_future · smooth_l1(z_future, z_target_future)
  + λ_dec · CE(logits, tokens)
  + λ_var · max(0, margin − √var)
  + λ_cov · off_diag(cov)²
```

future loss has warmup from 0 — early target encoder is unstable, raw future loss injects noise. without warmup, training diverges within the first ~2k steps.

diagnostics
-----------

you cannot debug a JEPA by watching loss go down. loss decreases while representations collapse. you need auxiliary metrics. we log 30+ every step:

**nextlat / nextlat-rank (Microsoft Research, 2025)**
effective_rank — shannon entropy of SVD spectrum. should stay >5, collapse → 1
participation_ratio — (ΣS)²/ΣS², effective dimensionality. 1 = rank-1 collapse
condition_number — S[0]/S[-1]. healthy 10–1000, ∞ = degenerate
numerical_rank — torch.linalg.matrix_rank. should be close to min(N,D)
rank_utilization — numerical_rank / min(N,D). 0.3–0.9 healthy
coherence — max |off-diagonal| of covariance. low is healthy

**i-jepa (Assran et al., CVPR 2023)**
collapsed_dim_ratio — fraction of near-zero-variance dimensions. near 0 is healthy
sv_entropy — normalized entropy of singular values. 1 = spread spectrum, 0 = single component
representation_stability — cosine between consecutive target updates. >0.99 is good

**c-jepa / byol (Grill et al., NeurIPS 2020)**
svd_sharpness — S[0]²/ΣS². 1 = rank-1 collapse. random data ~1/D

**lecun (2022) — jepa position paper**
alpha_norm — power-law exponent of SVD spectrum. higher = concentrated information

**ansuini et al. (NeurIPS 2019)**
intrinsic_dim — two-nearest-neighbor intrinsic dimensionality estimate. lower = more structured

**dinov2 (Oquab et al., 2024)**
mean_pairwise_cosine — intra-batch cosine similarity. high = collapse
cov_trace — trace of feature covariance / D. near-zero = collapse

**wang & isola (ICLR 2022)**
uniformity — alignment + uniformity on hypersphere. measures distribution quality

**barlow twins (Zbontar et al., ICML 2021)**
cross_corr_redundancy — mean |off-diagonal| of cross-correlation matrix. near 0 is healthy

**kornblith et al. (ICML 2019)**
cka_linear — linear CKA via HSIC. online-target similarity
cka_rbf — nonlinear CKA via RBF kernel. more sensitive to differences

all follow the nextlat exception pattern: SVD failure → 0.0 (or inf for condition_number). never crash the training loop.

code structure
--------------

```
src/models/       encoder, predictor, decoder, collapse diagnostics, main model
src/masks/        span masking with curriculum
src/datasets/     wikitext-103 / bookcorpus (kaggle-compatible)
src/utils/        schedulers, logging (from I-JEPA)
src/eval/         linear probe, future-token probe, geometry metrics
baselines/        data2vec (from official fairseq), MLM
configs/          per-size YAML configs
defaults.yaml     default config (all values, NextLat pattern)
scripts/          training scripts per benchmark
tests/            84 tests
train_probe.py    probe evaluation (NextLat pattern)
```

the differences between text-span jepa, data2vec, and MLM are best understood by reading their respective `compute_loss()` functions. this follows the nextlat convention.

provenance
----------

code patterns from reference implementations (variable names changed):

I-JEPA: momentum scheduler, param groups, smooth_l1 loss, layer_norm on targets, trunc_normal_ init, depth-wise rescaling, AverageMeter, CSVLogger, grad_logger

data2vec: get_annealed_rate, regression head (Linear→GELU→Linear), loss scaling by 1/√dim, target centering

NextLat: compute_hidden_state_rank with effective_rank via shannon entropy, exception→0.0, rank_utilization, defaults.yaml

VICReg: variance margin, off-diagonal covariance penalty

Barlow Twins: cross-correlation redundancy

Kornblith et al.: linear CKA via HSIC, RBF CKA

cite
----

```bibtex
@article{textspanjepa2026,
  title={Text-Span JEPA: Latent Predictive Learning for Language Representations},
  author={Text-Span JEPA Authors},
  year={2026}
}
```

license
-------

apache 2.0
