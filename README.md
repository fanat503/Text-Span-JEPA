text-span-jepa
==============

latent prediction at masked spans + future positions.

not token reconstruction. predict in latent space — that's the whole point of JEPA (LeCun 2022). the encoder learns what matters because it never has to waste capacity on low-level token details or noise.

the twist for text: span-level masking forces the model to use broader context. future latent prediction gives it a reason to encode directionality. these two things together are what make this different from I-JEPA and data2vec.

---

setup
-----

```
pip install -r requirements.txt
pip install -e .
```

python 3.9+, pytorch 2.0+. trains on wikitext.

training
--------

```
python -m src.train --fname configs/small-100m.yaml
```

resume: set `meta.load_checkpoint: true` in the config. picks up from `checkpoint-latest.pth.tar`.

configs: `debug.yaml` (something like sanity), `small-100m.yaml` (90M, 16GB (mini model)), `base-200m.yaml` (140M, 24GB), `large-350m.yaml` (280M, 40GB), `kaggle.yaml` (for T4).

architecture
------------

three components.

encoder — bidirectional transformer. same architecture for online and target. target encoder is EMA copy with scheduled tau = from 0.996 to 1.0 (constant tau doesn't work, I-JEPA showed this).

predictor — narrow transformer. takes encoder output, inserts mask tokens at span positions, predicts target latent. two modes:
- span: predict latents of masked blocks. each pass gets a slightly better estimate — like thinking in latent space.
- future: predict h[t+d] from h[t] with learned offset queries. it's a simpler task.

decoder — projection to token space. auxiliary. if latents collapse to a uniform vector, the decoder can't predict different tokens, so it acts as an anti-collapse signal. doesn't dominate training.

collapse prevention: VICReg + data2vec target centering.


future loss has warmup from 0. without warmup, training diverges within the first 2k steps.

diagnostics
-----------

we log these metrics every step:

nextlat / nextlat-rank (Microsoft Research)
effective_rank
participation_ratio
condition_number
numerical_rank
rank_utilization
coherence

i-jepa (Assran et al., CVPR 2023)
collapsed_dim_ratio
sv_entropy
representation_stability

c-jepa / byol (Grill et al., NeurIPS 2020)
svd_sharpness

lecun (2022) — jepa position paper
alpha_norm

ansuini et al. (NeurIPS 2019)
intrinsic_dim

dinov2 (Oquab et al., 2024)
mean_pairwise_cosine
cov_trace

wang & isola (ICLR 2022)
uniformity

barlow twins (Zbontar et al., ICML 2021)
cross_corr_redundancy

kornblith et al. (ICML 2019)
cka_linear
cka_rbf


code structure
--------------

```
src/models/       encoder, predictor etc.
src/masks/        span masking
src/datasets/     wikitext-103 / bookcorpus (for kaggle)
src/utils/        schedulers, logging
src/eval/         probes and geometry metrics
baselines/        data2vec, MLM
configs/          per-size YAML configs
defaults.yaml     default config
scripts/          training scripts
tests/            84 tests
train_probe.py    probe 
```

the differences between text-span jepa, data2vec, and MLM are best understood by reading their respective `compute_loss()` functions.

provenance
----------

code patterns from reference implementations (variable names changed):

I-JEPA

data2vec

NextLat

VICReg

Barlow Twins

Kornblith et al.

cite
----

```bibtex
@article{textspanjepa2026,
  title={Text-Span JEPA: Latent Predictive Learning for Language Representations},
  author={Slyatski Ilya},
  year={2026}
}
```

license
-------

Apache 2.0
