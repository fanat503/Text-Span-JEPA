.PHONY: audit dummy train eval budget

audit:
	python scripts/audit.py
	python -m py_compile *.py scripts/*.py baselines/*.py tests/*.py

dummy:
	python scripts/make_dummy_data.py

budget:
	python scripts/estimate_budget.py --config configs/small.json

train:
	python train.py --config configs/small.json

eval:
	python eval.py --checkpoint runs/jepa_small/best_val_jepa_small.pt --out runs/jepa_small/eval.json
