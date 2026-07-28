"""Build demo/demo_data.js for the GitHub Pages live demo.

Reads real pipeline outputs (crawl journal + per-image captions + fused
summaries) produced by the small-model demo run and embeds them, with
Wikimedia thumbnail URLs, as a JS data file consumed by index.html.

Usage:
    PYTHONPATH=src python tools/demo/build_demo_data.py \
        --run outputs/demo --output demo/demo_data.js
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from beyond_images.utils.jsonl import load_json, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/demo", help="Demo pipeline output folder")
    parser.add_argument("--output", default="demo/demo_data.js")
    args = parser.parse_args()

    run = Path(args.run)
    fused = load_json(run / "fused.json")
    summary = load_json(run / "summary.json")

    image_meta: dict[str, dict] = {}
    for rec in read_jsonl(run / "crawl.jsonl"):
        for img in rec["images"]:
            image_meta[img["id"]] = img

    entities = []
    for name, record in fused.items():
        qid = record["entity_qid"]
        captions = summary.get(name, {}).get("images", {})
        images = []
        for image_id, value in captions.items():
            if image_id == "merged_descriptions" or not isinstance(value, dict):
                continue
            meta = image_meta.get(image_id, {})
            url = meta.get("download_url") or meta.get("image_url")
            if not url:
                continue
            images.append(
                {
                    "id": image_id,
                    "url": url,
                    "page": meta.get("page_url", ""),
                    "caption": value.get("image_description_detail", ""),
                    "credit": meta.get("summary", {}).get("Author", ""),
                    "license": meta.get("summary", {}).get("License", ""),
                }
            )
        images.sort(key=lambda item: int(item["id"].rsplit("_", 1)[-1]))
        entities.append(
            {
                "name": name.replace("_", " "),
                "qid": qid,
                "dbpedia": record.get("dbpedia_url", ""),
                "wikidata": record.get("wikidata_url", ""),
                "summary": record["images"].get("images_t5_descriptions", ""),
                "images": images,
            }
        )

    payload = {
        "note": (
            "Real output of the Beyond Images pipeline run end-to-end with the "
            "small smoke-test models (BLIP captioner + Flan-T5-base fusion) on 8 "
            "MKG-W entities. Images are hotlinked from Wikimedia; follow the "
            "per-image page link for full attribution."
        ),
        "entities": entities,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8",
    )
    total_images = sum(len(e["images"]) for e in entities)
    print(f"wrote {out} ({len(entities)} entities, {total_images} images)")


if __name__ == "__main__":
    main()
