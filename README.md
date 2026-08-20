This repo was a bit edited by LLM due to my English, but i really tried to do it maximally readable, and reviewed it a lot of times :)

text-span-jepa
==============

latent prediction at masked spans + future positions.

not token reconstruction. predict in latent space — that's the point of JEPA (LeCun). the encoder learns what matters because it shouldn't waste capacity on useless details.

something like twist: span masking forces the model to use broader context. future latent prediction gives it a reason to encode directionality.

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

resume: set `meta.load_checkpoint: true` in the config. picks up from `checkpoint-latest.pth.tar`.

configs: `debug.yaml` (something like sanity), `small-100m.yaml` (almost 100M), `base-200m.yaml` (something aroud 140M, but i did it because of GPU efficiency), `large-350m.yaml` (almost 300M), `kaggle.yaml` (created for T4 in Kaggle).


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

novel mechanisms (16)

each mechanism addresses a specific failure mode of standard JEPA, which I hope will help to large JEPA models
