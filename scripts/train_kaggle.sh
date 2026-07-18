#!/bin/bash
# Text-Span JEPA — Kaggle T4/P100 training script
set -euo pipefail
CONFIG=${1:-configs/kaggle.yaml}
echo "=== Text-Span JEPA — Kaggle Training ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
