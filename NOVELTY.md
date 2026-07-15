# Novelty

The novelty is not just the model class. It is the full training/evaluation package:

- masked-span latent prediction for text;
- EMA target encoder;
- predictor with latent refinement steps;
- tied decoder grounding predicted latents;
- variance/covariance collapse guards;
- target centering;
- representation-first evaluation protocol;
- explicit MLM baseline and ablations.

Safe claim: Text-Span JEPA is a representation-learning architecture for text, not an autoregressive LLM.
