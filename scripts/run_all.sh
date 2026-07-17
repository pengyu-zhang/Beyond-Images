#!/usr/bin/env bash
# Full enrichment pipeline for one dataset. Mirrors the paper's three modules:
#   (1) Modality Extension  -> crawl new images
#   (2) Semantic Generation -> caption original + new images, merge per entity
#   (3) LLM-based Fusion    -> fused entity summaries
#   (4) Downstream exports  -> embeddings (.h5/.pth) + token ids (MyGO)
#
# Prerequisites (see data/README.md):
#   data/raw/<DATASET>/ent_links.tsv          entity alignment (3 columns)
#   data/raw/<DATASET>/images_original/       original images as QID_idx.jpg
#
# Usage:
#   bash scripts/run_all.sh MKG-W
#   CONFIG=configs/paper.yaml bash scripts/run_all.sh DB15K
#
# NOTE: full-scale runs are long (crawling 15k entities and captioning
# ~100k images). Every stage journals progress and resumes when re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?Usage: run_all.sh <DATASET>  e.g. MKG-W, MKG-Y, DB15K}"
CONFIG="${CONFIG:-configs/default.yaml}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

RAW="data/raw/$DATASET"
OUT="outputs/$DATASET"
LINKS="$RAW/ent_links.tsv"
mkdir -p "$OUT"

[ -f "$LINKS" ] || { echo "Missing $LINKS - see data/README.md"; exit 1; }

echo "== [1/6] Crawl new images from Wikipedia =="
python -m beyond_images crawl --config "$CONFIG" \
    --links "$LINKS" --images-dir "$RAW/images_new" \
    --journal "$OUT/crawl.jsonl" --metadata "$OUT/new_images_metadata.json"

echo "== [2/6] Caption original images =="
if [ -d "$RAW/images_original" ]; then
    python -m beyond_images caption --config "$CONFIG" \
        --images "$RAW/images_original" --output "$OUT/captions_original.jsonl"
    python -m beyond_images merge --config "$CONFIG" \
        --links "$LINKS" --captions "$OUT/captions_original.jsonl" \
        --output "$OUT/${DATASET}_original_img_summary.json"
else
    echo "   (no $RAW/images_original - skipping)"
fi

echo "== [3/6] Caption new images =="
python -m beyond_images caption --config "$CONFIG" \
    --images "$RAW/images_new" --output "$OUT/captions_new.jsonl"
python -m beyond_images merge --config "$CONFIG" \
    --links "$LINKS" --captions "$OUT/captions_new.jsonl" \
    --output "$OUT/${DATASET}_new_img_summary.json"

echo "== [4/6] LLM fusion =="
INPUTS=("$OUT/${DATASET}_original_img_summary.json")
[ -f "$OUT/${DATASET}_new_img_summary.json" ] && INPUTS+=("$OUT/${DATASET}_new_img_summary.json")
python -m beyond_images fuse --config "$CONFIG" \
    --inputs "${INPUTS[@]}" \
    --journal "$OUT/fuse.jsonl" --output "$OUT/${DATASET}_fused.json"

echo "== [5/6] Embeddings for MMRNS (.h5) / AdaMF (.pth) =="
python -m beyond_images embed --config "$CONFIG" \
    --input "$OUT/${DATASET}_fused.json" \
    --h5 "$OUT/${DATASET}_description_sentences.h5" \
    --pth "$OUT/${DATASET}-textual.pth"

echo "== [6/6] Token ids for MyGO (.json) =="
python -m beyond_images tokens --config "$CONFIG" \
    --input "$OUT/${DATASET}_fused.json" --output "$OUT/${DATASET}-tokens.json"

echo "Pipeline complete. Outputs in $OUT/"
