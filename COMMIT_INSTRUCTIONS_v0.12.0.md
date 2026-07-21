# Text-Span JEPA v0.12.0 — Commit Instructions

## What changed (v0.11.0 → v0.12.0)

### CRITICAL BUG FIXES (training would crash without these)

1. **model_name mapping** — Configs used suffixed names like
   `text_span_jepa_small`, `mlm_small`, `data2vec_base`, but
   `create_model()`, `get_param_groups()`, `do_ema_update()` only
   accepted exact names. Added `_normalize_model_name()` that strips
   suffixes to canonical form. **Every training run crashed at model creation.**

2. **save_checkpoint() crashes for MLM/data2vec** — Assumed model
   always has `.predictor`, `.decoder`, `.target_encoder`. Rewrote to
   dispatch by model_name. **Saving any MLM/data2vec checkpoint crashed.**

3. **get_param_groups() wrong param groups** — With suffixed name,
   fell through to `else` branch returning all params in one group
   (no WD_exclude). Now normalizes name first.

4. **do_ema_update() skipped data2vec** — Checked
   `model_name in ('data2vec', 'data2vec_baseline')` but config uses
   `data2vec_base`. EMA target was never updated.

### IMPORTANT FIXES

5. **Mask curriculum disabled** — `SpanMaskCollator` created without
   `mask_ratio_start`/`mask_ratio_end`/`curriculum_steps`. Train.py
   now passes them from model config.

6. **CSVLogger unused** — Now logs loss components + diagnostics
   every `log_freq` steps to `train_log.csv`.

7. **Individual loss components not logged** — Now logs
   `loss_span`, `loss_future`, `loss_decoder`, `loss_variance`,
   `loss_covariance`, `decoder_accuracy`.

8. **grad_accum_steps missing** — Added to defaults.yaml and
   all per-experiment configs.

9. **Gradient clipping per-component** — Changed to global
    `clip_grad_norm_` on all trainable params (I-JEPA pattern).

10. **Old configs/ directory** — Deleted.

11. **visualization.py numpy array** — `convergence_plot` crashed
    when passed numpy arrays instead of lists. Fixed with `list()`.

### NEW

12. **`_get_all_trainable_params()`** — Excludes target encoder.

13. **CSV loss logging** — 13-column CSV file.

14. **13 new tests** in `TestV012Bugfixes`.

15. **Version bumped** to 0.12.0.

## Total tests: 227 (120 model + 94 interp + 13 v0.12)

## Smoke test results

```
JEPA 200-step:  loss 1.2091 → 0.9640 ↓  eff_rank=77.9  no NaN  ckpt OK
MLM 100-step:   loss 7.1035 → 5.7608 ↓  no NaN  ckpt OK
data2vec 100-step: loss 8.6445 → 3.7363 ↓  no NaN  ckpt OK
Interp modules: 12/12 OK
227/227 tests passed
```

## How to commit and push

The commit already exists locally at `51308a9` (tag: v0.12.0).
Just push:

```bash
cd /workspaces/Text-Span-JEPA
git push origin main --tags
```

If starting from a clean v0.3.1 checkout:

```bash
cd /workspaces/Text-Span-JEPA
git checkout main
git pull origin main
# The v0.12.0 commit should already be there after push
```

## Next steps after pushing

1. `bash scripts/run_experiment.sh train_jepa`
2. `bash scripts/run_experiment.sh train_mlm`
3. `bash scripts/run_experiment.sh compare`
4. Return to NeurIPS Protocol Phase 1 with actual results
