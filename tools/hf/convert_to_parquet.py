"""Convert the released Beyond Images JSON files into Parquet for Hugging Face.

Input: the extracted img_text_summary/ folder from the Zenodo/Release zip.
Output: three viewer-friendly tables, one Parquet file per (dataset, source):

  captions/        one row per image caption
  entities/        one row per entity (merged caption text)
  image_metadata/  one row per crawled image (URLs + Wikipedia file metadata)

Usage:
    python tools/hf/convert_to_parquet.py --input <img_text_summary dir> --output <staging dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASETS = {"MKG-W": "mkg_w", "MKG-Y": "mkg_y", "DB15K": "db15k"}

BLIP_FILES = {
    ("MKG-W", "original"): "MKG-W_original_img_summary_blip.json",
    ("MKG-W", "new"): "MKG-W_img_new_summary_blip.json",
    ("MKG-Y", "original"): "MKG-Y_original_img_summary_blip.json",
    ("MKG-Y", "new"): "MKG-Y_img_new_summary_blip.json",
    ("DB15K", "original"): "DB15K_img_original_summary_blip.json",
    ("DB15K", "new"): "DB15K_img_new_summary_blip.json",
}

WIKI_FILES = {
    "MKG-W": "MKG-W_img_new_summary_wiki.json",
    "MKG-Y": "MKG-Y_img_new_summary_wiki.json",
    "DB15K": "DB15K_img_new_summary_wiki.json",
}


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    print(f"  {path.name}: {table.num_rows} rows, {path.stat().st_size / 1e6:.1f} MB")


def convert_blip(input_dir: Path, output_dir: Path) -> None:
    for (dataset, source), filename in BLIP_FILES.items():
        print(f"[blip] {dataset} / {source}")
        data = json.load(open(input_dir / filename, encoding="utf-8"))
        slug = DATASETS[dataset]

        caption_rows, entity_rows = [], []
        for entity_name, record in data.items():
            base = {
                "dataset": dataset,
                "source": source,
                "entity_name": str(entity_name),
                "entity_qid": str(record.get("entity_qid", "")),
                "dbpedia_url": str(record.get("dbpedia_url", "")),
                "wikidata_url": str(record.get("wikidata_url", "")),
            }
            images = record.get("images", {}) or {}
            n = 0
            for image_id, value in images.items():
                if image_id == "merged_descriptions" or not isinstance(value, dict):
                    continue
                caption = value.get("image_description_detail")
                if caption is None:
                    continue
                caption_rows.append({**base, "image_id": image_id, "caption": str(caption)})
                n += 1
            merged = images.get("merged_descriptions", "")
            entity_rows.append(
                {**base, "merged_descriptions": str(merged or ""), "num_images": n}
            )

        write_parquet(caption_rows, output_dir / "captions" / f"{slug}_{source}.parquet")
        write_parquet(entity_rows, output_dir / "entities" / f"{slug}_{source}.parquet")
        del data, caption_rows, entity_rows


def convert_wiki(input_dir: Path, output_dir: Path) -> None:
    for dataset, filename in WIKI_FILES.items():
        print(f"[wiki] {dataset}")
        data = json.load(open(input_dir / filename, encoding="utf-8"))
        slug = DATASETS[dataset]

        rows = []
        for rec in data:
            summary = rec.get("summary") or {}
            description = next(
                (v for k, v in summary.items() if k.startswith("Description")), ""
            )
            rows.append(
                {
                    "dataset": dataset,
                    "image_id": str(rec.get("id", "")),
                    "entity_qid": str(rec.get("id", "")).split("_", 1)[0],
                    "wikidata_url": str(rec.get("wikidata_url", "")),
                    "page_url": str(rec.get("page_url") or ""),
                    "image_url": str(rec.get("image_url") or ""),
                    "description": str(description),
                    "date": str(summary.get("Date", "")),
                    "author": str(summary.get("Author", "")),
                    "source_field": str(summary.get("Source", "")),
                    "metadata_json": json.dumps(summary, ensure_ascii=False),
                }
            )
        write_parquet(rows, output_dir / "image_metadata" / f"{slug}_new.parquet")
        del data, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Extracted img_text_summary directory")
    parser.add_argument("--output", required=True, help="Staging directory for the HF dataset")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    convert_blip(input_dir, output_dir)
    convert_wiki(input_dir, output_dir)
    print("done")


if __name__ == "__main__":
    main()
