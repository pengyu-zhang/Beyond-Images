#!/usr/bin/env bash
# Convenience wrapper around the pipeline CLI.
#
# Usage:
#   bash scripts/run_pipeline.sh <stage> [extra CLI args...]
# Examples:
#   bash scripts/run_pipeline.sh fuse --inputs data/processed/img_text_summary/MKG-W_original_img_summary_blip.json \
#        --journal outputs/MKG-W/fuse.jsonl --output outputs/MKG-W/fused.json
#   CONFIG=configs/paper.yaml bash scripts/run_pipeline.sh caption --images data/raw/MKG-W/images_new \
#        --output outputs/MKG-W/captions_new.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/default.yaml}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${1:?Usage: run_pipeline.sh <stage> [args...] (see: python -m beyond_images --help)}"
shift

exec python -m beyond_images "$STAGE" --config "$CONFIG" "$@"
