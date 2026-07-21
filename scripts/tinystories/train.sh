#!/bin/bash
set -e
echo "=== Text-Span JEPA — TinyStories small ==="
CUDA_VISIBLE_DEVICES=0 python -m src.train --fname config/tinystories/textspanjepa_tinystories.yaml
