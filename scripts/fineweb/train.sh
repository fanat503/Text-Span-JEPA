#!/bin/bash
set -e
NUM_SHARDS=${1:-1}
echo "=== Text-Span JEPA — FineWeb-Edu (${NUM_SHARDS} shards) ==="
python -c "import yaml; c=yaml.safe_load(open('config/wikitext/textspanjepa_wikitext_small.yaml')); c['meta']['dataset']='fineweb_edu'; c['data']['root_path']='data/fineweb-edu'; c['data']['num_train_files']=${NUM_SHARDS}; c['logging']['folder']='output/fineweb/small-100m/'; yaml.dump(c,open('/tmp/fineweb_config.yaml','w'))"
CUDA_VISIBLE_DEVICES=0 python -m src.train --fname /tmp/fineweb_config.yaml
