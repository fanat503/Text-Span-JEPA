# Text-Span JEPA v0.12.0 — Полные инструкции по коммиту и очистке

## Текущее состояние

```
Локальный коммит: 9767b8b (tag v0.12.0) на ветке main
Удалённый сервер:  5b5f7d2 (только начальный LICENSE)
Ветка dev:        устаревшая (f7cadcc) — будет удалена
```

## Что содержит v0.12.0 (суммарно от v0.3.1)

### Критические баг-фиксы (без них обучение крашилось)

1. **model_name mapping** — `_normalize_model_name()` маппит
   `text_span_jepa_small` → `text_span_jepa`, `mlm_small` → `mlm`,
   `data2vec_base` → `data2vec`. Без этого КАЖДЫЙ training run крашился.

2. **save/load checkpoint для MLM/data2vec** — Прежде save_checkpoint()
   предполагал .predictor/.decoder/.target_encoder — крашился на MLM/data2vec.

3. **get_param_groups()** — С суффиксом имени выдавал 1 группу вместо 5.

4. **do_ema_update()** — Не обновлял EMA для data2vec_base.

### Важные фиксы

5. Mask curriculum была отключена — параметры не передавались в SpanMaskCollator
6. CSVLogger не использовался — теперь логирует 13 колонок
7. Компоненты loss не логировались — добавлено полное логирование
8. grad_accum_steps отсутствовал в defaults.yaml и конфигах
9. Gradient clipping по-компонентно → глобальный (I-JEPA)
10. Старая директория configs/ удалена

### Фиксы v0.12.0+2 (этот коммит)

11. **TextDataset** — добавлен `drop_last=False` для evaluation
    (513 токенов → 2 чанка вместо 1)
12. **CollapseDiagnostics NaN guard** — `std(dim=(0,1))` на (1,1,D)
    давал NaN. Исправлено через `nan_to_num(0.0)`
13. **visualization.py** — `convergence_plot` крашился на numpy arrays

### Новые фичи

- `_get_all_trainable_params()` — excludes target encoder
- CSV loss logging с 13 колонками
- 13 новых тестов для v0.12 bugfixes
- Версия 0.12.0

## Тесты: 227 (120 model + 94 interp + 13 v0.12)

## Smoke-тест

```
JEPA 200-step:     loss 1.194 → 0.810 ↓  rank=81.4  no NaN  ckpt ✓
MLM 100-step:      loss 7.103 → 5.761 ↓  no NaN  ckpt ✓
data2vec 100-step: loss 8.644 → 3.736 ↓  no NaN  ckpt ✓
TextDataset:       drop_last ✓  NaN guard ✓  seed ✓
```

## Инструкции

### Шаг 1: Проверить что локально всё чисто

```bash
cd /workspaces/Text-Span-JEPA
git status          # должен быть clean
git log --oneline -1  # должен показать 9767b8b v0.12.0
git tag -l          # должен показать v0.12.0
```

### Шаг 2: Удалить устаревшую ветку dev

```bash
git branch -D dev
```

### Шаг 3: Пуш на GitHub (force — перезапишет удалённый)

```bash
git push origin main --force --tags
```

> **Важно:** Нужен `--force` потому что удалённый сервер на 5b5f7d2,
> а локальный на 9767b8b. История переписана (v0.3.0/v0.3.1/v0.12.0
> не были запушены ранее с этого origin).

### Шаг 4: Проверить на GitHub

```bash
# На GitHub:
# - Открыть https://github.com/fanat503/Text-Span-JEPA
# - Убедиться что последний коммит: 9767b8b
# - Убедиться что тег v0.12.0 существует
# - Файлы: config/, src/interp/, scripts/run_experiment.sh — все на месте
```

### Шаг 5: Клонировать и запустить тесты (для проверки)

```bash
cd /tmp
git clone https://github.com/fanat503/Text-Span-JEPA.git Text-Span-JEPA-verify
cd Text-Span-JEPA-verify
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml numpy transformers pytest
python -m pytest tests/ -q
# Должно быть: 227 passed
```

### Шаг 6: Очистка после merge

```bash
# Удалить старые инструкции (они уже в коммите)
cd /workspaces/Text-Span-JEPA
rm -f COMMIT_INSTRUCTIONS_v0.10.0.md COMMIT_INSTRUCTIONS_v0.11.0.md
# (v0.12.0 инструкция оставлена как документация)

# Удалить старые patch-файлы (они больше не нужны — всё в git)
rm -f /home/user/Text-Span-JEPA-v0.6.0.patch
rm -f /home/user/Text-Span-JEPA-v0.7.0.patch
rm -f /home/user/Text-Span-JEPA-v0.8.0.patch
rm -f /home/user/Text-Span-JEPA-v0.9.0.patch
rm -f /home/user/Text-Span-JEPA-v0.10.0.patch
rm -f /home/user/Text-Span-JEPA-v0.11.0.patch

# Удалить egg-info (пересоздаётся при pip install -e .)
rm -rf text_span_jepa.egg-info/

git add -A
git commit -m "chore: cleanup old patch files and instructions"
git push origin main
```

## Что делать дальше (после успешного push)

1. Запустить обучение: `bash scripts/run_experiment.sh train_jepa`
2. Запустить базовые модели: `bash scripts/run_experiment.sh train_mlm`
3. Запустить сравнение: `bash scripts/run_experiment.sh compare`
4. Вернуться к NeurIPS Protocol Phase 1 с результатами
