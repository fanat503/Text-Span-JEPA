#!/bin/bash
set -e
NUM_CORES=${1:-8}
echo "=== Text-Span JEPA — WikiText-103 base (TPU V5-E8, ${NUM_CORES} cores) ==="
python -m src.train_tpu --fname config/wikitext/textspanjepa_wikitext_base.yaml
