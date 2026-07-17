"""Fuse per-entity caption sets into single summaries (resume-safe).

Replaces original scripts 8.2-8.4. Input: one or more entity summary JSON
files produced by the captioning merge step (typically the original-image and
new-image files). Output: the same entity schema with
`images.images_t5_descriptions` holding the fused paragraph, plus a JSONL
journal so interrupted runs resume where they stopped.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..utils.jsonl import JsonlWriter, load_json, read_jsonl, save_json_atomic
from .fusers import Fuser

FUSED_KEY = "images_t5_descriptions"  # kept for compatibility with released data


def clean_descriptions(descriptions: list[str]) -> list[str]:
    """De-duplicate and drop degenerate captions (as in the original scripts)."""
    cleaned = []
    for desc in dict.fromkeys(descriptions):
        if not desc or desc.isspace():
            continue
        words = desc.split()
        if len(words) < 3:
            continue
        if len(set(words)) < len(words) / 2 and len(words) > 5:
            continue  # highly repetitive caption ("person person person ...")
        cleaned.append(desc)
    return cleaned


def collect_descriptions(
    input_files: list[str | Path],
    priority_index: int = 0,
    max_per_entity: int = 500,
) -> dict[str, dict]:
    """Merge entity records from `input_files`, capping descriptions per entity.

    Descriptions from `input_files[priority_index]` fill the cap first
    (the original scripts prioritised original-image captions).
    """
    merged: dict[str, dict] = {}
    ordered = [input_files[priority_index]] + [
        f for i, f in enumerate(input_files) if i != priority_index
    ]
    for source in ordered:
        data = load_json(source)
        for entity_name, record in data.items():
            slot = merged.setdefault(
                entity_name,
                {
                    **{k: v for k, v in record.items() if k != "images"},
                    "_descriptions": [],
                },
            )
            images = record.get("images")
            if not isinstance(images, dict):
                continue
            captions = [
                value["image_description_detail"]
                for value in images.values()
                if isinstance(value, dict) and value.get("image_description_detail")
            ]
            remaining = max_per_entity - len(slot["_descriptions"])
            if remaining > 0:
                slot["_descriptions"].extend(clean_descriptions(captions)[:remaining])
    return merged


def fuse_entities(
    fuser: Fuser,
    merged: dict[str, dict],
    journal_jsonl: str | Path,
    output_json: str | Path,
    limit: int | None = None,
) -> dict[str, int]:
    items = list(merged.items())
    if limit:
        items = items[:limit]

    done = {rec["entity_name"]: rec[FUSED_KEY] for rec in read_jsonl(journal_jsonl)}
    todo = [(name, rec) for name, rec in items if name not in done]
    print(f"[fuse] {len(items)} entities, {len(done)} done, {len(todo)} to fuse")

    stats = {"entities": len(items), "fused": 0, "empty": 0, "errors": 0}
    started = time.time()
    with JsonlWriter(journal_jsonl) as writer:
        for idx, (entity_name, record) in enumerate(todo, 1):
            descriptions = record["_descriptions"]
            if not descriptions:
                stats["empty"] += 1
                continue
            try:
                fused = fuser.fuse(entity_name, descriptions)
            except Exception as exc:  # keep the queue moving; log and continue
                stats["errors"] += 1
                print(f"[fuse] error on {entity_name!r}: {exc}")
                continue
            writer.write({"entity_name": entity_name, FUSED_KEY: fused})
            done[entity_name] = fused
            stats["fused"] += 1
            if idx % 10 == 0:
                rate = stats["fused"] / max(time.time() - started, 1e-6)
                print(f"[fuse] {idx}/{len(todo)} ({rate:.2f} ent/s)")

    final = {}
    for entity_name, record in items:
        output_record = {k: v for k, v in record.items() if not k.startswith("_")}
        output_record["images"] = {}
        if entity_name in done:
            output_record["images"][FUSED_KEY] = done[entity_name]
        final[entity_name] = output_record
    save_json_atomic(final, output_json)
    return stats
