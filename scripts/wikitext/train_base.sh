#!/bin/bash
# Text-Span JEPA — WikiText-103 base (single GPU, 24GB+)
set -euo pipefail

CONFIG=${1:-configs/base-200m.yaml}
echo "=== Text-Span JEPA — WikiText-103 base ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
