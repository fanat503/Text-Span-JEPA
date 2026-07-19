#!/bin/bash
# Text-Span JEPA — WikiText-103 small (single GPU, 16GB+)
set -euo pipefail

CONFIG=${1:-configs/small-100m.yaml}
echo "=== Text-Span JEPA — WikiText-103 small ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
