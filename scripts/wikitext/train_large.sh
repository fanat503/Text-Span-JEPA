#!/bin/bash
# Text-Span JEPA — WikiText-103 large (single GPU, 40GB+ or A100)
set -euo pipefail

CONFIG=${1:-configs/large-350m.yaml}
echo "=== Text-Span JEPA — WikiText-103 large ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
