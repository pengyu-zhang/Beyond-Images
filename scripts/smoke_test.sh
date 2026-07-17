#!/usr/bin/env bash
# End-to-end smoke test: runs fusion -> embeddings -> token export on the
# bundled MKG-W sample using small, ungated models. Finishes in a few
# minutes on GPU (or CPU). Optionally exercises the Wikipedia crawler and
# a small captioner on one entity with --with-crawl (needs network).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG=configs/smoke.yaml
OUT=outputs/smoke
N=5

echo "== [1/5] Extract bundled sample =="
bash scripts/prepare_data.sh --sample

SAMPLE=data/sample/extracted/img_text_summary
mkdir -p "$OUT"

echo "== [2/5] LLM fusion on $N entities (flan-t5-base) =="
python -m beyond_images fuse --config $CONFIG \
    --inputs "$SAMPLE/MKG-W_original_img_summary_blip.json" "$SAMPLE/MKG-W_img_new_summary_blip.json" \
    --journal "$OUT/fuse.jsonl" --output "$OUT/fused.json" --limit $N

echo "== [3/5] Sentence embeddings (h5 + pth + manifest) =="
python -m beyond_images embed --config $CONFIG \
    --input "$OUT/fused.json" --h5 "$OUT/fused.h5" --pth "$OUT/fused.pth"

echo "== [4/5] MyGO token export =="
python -m beyond_images tokens --config $CONFIG \
    --input "$OUT/fused.json" --output "$OUT/tokens.json"

echo "== [5/5] Validate outputs =="
python - <<EOF
import json, h5py, torch

fused = json.load(open("$OUT/fused.json", encoding="utf-8"))
summaries = [r["images"].get("images_t5_descriptions", "") for r in fused.values()]
non_empty = [s for s in summaries if s.strip()]
assert non_empty, "no fused summaries produced"
print(f"fused.json: {len(fused)} entities, {len(non_empty)} non-empty summaries")
print("example summary:", non_empty[0][:120], "...")

with h5py.File("$OUT/fused.h5") as h5:
    keys = list(h5.keys())
    shape = h5[keys[0]].shape
assert shape[1] == 768, f"unexpected embedding dim {shape}"
print(f"fused.h5: {len(keys)} entities, dim {shape[1]}")

pth = torch.load("$OUT/fused.pth", map_location="cpu", weights_only=True)
manifest = open("$OUT/fused.pth.entities.txt", encoding="utf-8").read().splitlines()
assert pth.shape[0] == len(manifest), "pth rows != manifest entities"
print(f"fused.pth: {tuple(pth.shape)} rows aligned with manifest")

tokens = json.load(open("$OUT/tokens.json", encoding="utf-8"))
first = next(iter(tokens.values()))
assert first[0] == 101 and first[-1] == 102, "token sequences must be CLS...SEP"
print(f"tokens.json: {len(tokens)} entities, CLS/SEP OK")
print("SMOKE TEST PASSED")
EOF

if [ "${1:-}" = "--with-crawl" ]; then
    echo "== [extra] Crawl + caption one entity (needs network) =="
    printf 'http://dbpedia.org/resource/Amsterdam\tAmsterdam\thttp://www.wikidata.org/entity/Q727\n' > "$OUT/links_one.tsv"
    python -m beyond_images crawl --config $CONFIG \
        --links "$OUT/links_one.tsv" --images-dir "$OUT/images" \
        --journal "$OUT/crawl.jsonl" --metadata "$OUT/crawl_metadata.json" --limit 1
    python -m beyond_images caption --config $CONFIG \
        --images "$OUT/images" --output "$OUT/captions.jsonl" --limit 3
    python -m beyond_images merge --config $CONFIG \
        --links "$OUT/links_one.tsv" --captions "$OUT/captions.jsonl" \
        --output "$OUT/one_entity_summary.json"
    echo "CRAWL+CAPTION SMOKE PASSED"
fi
