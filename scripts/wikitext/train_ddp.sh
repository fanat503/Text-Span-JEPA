#!/bin/bash
set -e
NUM_GPUS=${1:-4}
echo "=== Text-Span JEPA — WikiText-103 base (DDP, ${NUM_GPUS} GPUs) ==="
torchrun --nproc_per_node=${NUM_GPUS} -m src.train_multi_gpu --fname config/wikitext/textspanjepa_wikitext_base.yaml
