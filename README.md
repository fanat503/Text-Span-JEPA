text-span-jepa
==============

latent prediction at masked spans + future positions.

not token reconstruction. predict in latent space — that's the whole point of JEPA (LeCun 2022). the encoder learns what matters because it never has to waste capacity on low-level token details or noise.

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


architecture
------------

three components.

encoder — bidirectional transformer. same architecture for online and target. target encoder is EMA copy with scheduled tau = from 0.996 to 1.0 (constant tau doesn't work, I-JEPA showed this).

predictor — narrow transformer. takes encoder output, inserts mask tokens at span positions, predicts target latent. two modes: span and future

decoder — projection to token space. auxiliary. if latents collapse to a uniform vector, the decoder can't predict different tokens. doesn't dominate training.

collapse prevention: VICReg + data2vec target centering.


future loss has warmup from 0. without warmup, training diverges within the first 2k steps.


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
