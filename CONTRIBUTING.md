# Contributing

Before changing model or training code, run:

```bash
python scripts/audit.py
python -m py_compile *.py scripts/*.py baselines/*.py tests/*.py
```

Changes that alter objective, data order, masking, or evaluation must be reflected
in `EXPERIMENT_CARD.md` and `REVIEW_ATTACKS.md`.
