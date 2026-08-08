text-span-jepa
==============

latent prediction at masked spans + future positions.

not token reconstruction. predict in latent space - that's the whole point of JEPA (LeCun 2022). the encoder learns what matters because it never has to look on token details or noise.


setup
-----

```
pip install -r requirements.txt
pip install -e .
```

python 3.9+, pytorch 2.0+.

training
--------

```
python -m src.train --fname configs/small-100m.yaml
```

resume: set `meta.load_checkpoint: true` in the config. picks up from `checkpoint-latest.pth.tar`.


architecture
------------

three components.

encoder — bidirectional transformer. same architecture for online and target. target encoder is EMA copy with scheduled tau.

predictor — narrow transformer.

decoder — projection to token space.


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
