text-span-jepa
==============

latent prediction at masked spans + future positions.

not token reconstruction. predict in latent space — that's the whole point of JEPA (LeCun 2022). the encoder learns what matters because it never has to waste capacity on useless details.

something like twist: span-level masking forces the model to use broader context. future latent prediction gives it a reason to encode directionality.

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

everything in the YAML, nothing on the CLI.

resume: set `meta.load_checkpoint: true` in the config. picks up from `checkpoint-latest.pth.tar`.

configs: `debug.yaml` (sanity), `small-100m.yaml` (~90M, 16GB), `base-200m.yaml` (~140M, 24GB), `large-350m.yaml` (~280M, 40GB), `kaggle.yaml` (tuned for T4).

architecture
------------

encoder — bidirectional transformer. same architecture for online and target. target encoder is EMA copy with scheduled tau.

predictor — narrow transformer. takes encoder output, do what it think matters and predicts target latent. two modes:
- span: mask contiguous blocks, predict their latents
- future: predict future positions from current with learned offset queries. it's a simpler task.

decoder — weight-tied projection. if latents collapse to a uniform vector, the decoder can't predict different tokens, so it's an implicit anti-collapse framework.

the differences between text-span jepa, data2vec,  and MLM are best understood by reading their respective compute_loss() functions.


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

novel mechanisms (16 in total)

each mechanism addresses a specific failure mode of standard JEPA, with a mathematical guarantee, which I hope will help to Large JEPA models
