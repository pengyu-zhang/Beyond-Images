"""Merge per-image captions with entity links into per-entity summary JSON.

Replaces original scripts 4.4-4.6. The output schema matches the released
dataset (entity_name -> {entity_name, entity_qid, dbpedia_url, wikidata_url,
images: {<image_id>: {image_description_detail}, merged_descriptions}}).

Bug fixes over the originals (documented in REFACTORING_NOTES):
  - QID matching uses the exact base QID (`Q42_1` -> `Q42`), never a string
    prefix test: `startswith("Q42")` also matched Q420/Q423 images in 4.4.
  - The caption file's first record is no longer dropped (4.4-4.6 passed
    `header=0` to pandas although the file has no header row).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..utils.jsonl import read_jsonl, save_json_atomic

DBPEDIA_PREFIX = "http://dbpedia.org/resource/"
WIKIDATA_PREFIX = "http://www.wikidata.org/entity/"


def load_entity_links(links_tsv: str | Path) -> list[dict[str, str]]:
    """Parse an ent_links TSV (2 or 3 columns) into entity records."""
    entities = []
    with open(links_tsv, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            dbpedia_url = parts[0]
            wikidata_url = parts[-1] if parts[-1].startswith(WIKIDATA_PREFIX) else ""
            entities.append(
                {
                    "entity_name": dbpedia_url.replace(DBPEDIA_PREFIX, ""),
                    "entity_qid": wikidata_url.replace(WIKIDATA_PREFIX, "") or "NAN",
                    "dbpedia_url": dbpedia_url,
                    "wikidata_url": wikidata_url or "NAN",
                }
            )
    return entities


def _captions_by_qid(captions_jsonl: str | Path) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in read_jsonl(captions_jsonl):
        caption = rec.get("caption")
        if not caption:
            continue
        image_id = Path(rec["image"]).stem  # e.g. Q42_1
        qid = image_id.split("_", 1)[0]
        grouped[qid].append((image_id, caption))
    for records in grouped.values():
        records.sort()
    return grouped


def merge_captions(
    links_tsv: str | Path,
    captions_jsonl: str | Path,
    output_json: str | Path,
) -> dict[str, int]:
    entities = load_entity_links(links_tsv)
    grouped = _captions_by_qid(captions_jsonl)

    result: dict[str, dict] = {}
    stats = {"entities": len(entities), "with_captions": 0, "captions": 0}
    for entity in entities:
        images: dict[str, object] = {}
        descriptions: list[str] = []
        for image_id, caption in grouped.get(entity["entity_qid"], []):
            images[image_id] = {"image_description_detail": caption}
            descriptions.append(caption)
        images["merged_descriptions"] = " ".join(descriptions)
        if descriptions:
            stats["with_captions"] += 1
            stats["captions"] += len(descriptions)
        result[entity["entity_name"]] = {**entity, "images": images}

    save_json_atomic(result, output_json)
    return stats
