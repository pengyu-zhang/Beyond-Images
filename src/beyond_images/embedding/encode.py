"""Encode enriched entity text into embeddings for downstream MMKG models.

Replaces original script 5.5. Outputs:
  - .h5  (entity name -> (1, dim) float32 dataset)   e.g. for MMRNS
  - .pth (stacked (N, dim) float32 tensor)           e.g. for AdaMF
  - .entities.txt manifest listing the row order of the .pth tensor
    (the original relied silently on JSON key order; the manifest makes the
    entity-to-row alignment explicit and checkable).

Improvements: batched GPU encoding (the original encoded one text at a time)
and a consistent embedding dimension taken from the model, not hardcoded.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from ..utils.jsonl import load_json

TEXT_KEYS = ("images_t5_descriptions", "merged_descriptions")


def extract_texts(entity_json: str | Path, text_key: str = "auto") -> dict[str, str]:
    """Pull one text per entity from a summary/fused JSON file."""
    data = load_json(entity_json)
    texts: dict[str, str] = {}
    for entity_name, record in data.items():
        images = record.get("images", {}) if isinstance(record, dict) else {}
        if text_key != "auto":
            value = images.get(text_key, "")
        else:
            value = next((images[k] for k in TEXT_KEYS if images.get(k)), "")
        value = str(value).strip()
        if value:
            texts[entity_name] = value
    return texts


def encode_texts(
    texts: dict[str, str],
    model_name: str = "bert-base-uncased",
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[list[str], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    entities = list(texts.keys())
    embeddings = model.encode(
        [texts[name] for name in entities],
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    return entities, embeddings


def write_outputs(
    entities: list[str],
    embeddings: np.ndarray,
    h5_path: str | Path | None = None,
    pth_path: str | Path | None = None,
) -> None:
    if h5_path:
        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as h5:
            for name, vector in zip(entities, embeddings):
                h5.create_dataset(name, data=vector[None, :])
        print(f"[embed] wrote {h5_path} ({len(entities)} entities, dim={embeddings.shape[1]})")
    if pth_path:
        pth_path = Path(pth_path)
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(embeddings), pth_path)
        manifest = pth_path.with_suffix(pth_path.suffix + ".entities.txt")
        manifest.write_text("\n".join(entities) + "\n", encoding="utf-8")
        print(f"[embed] wrote {pth_path} + row manifest {manifest.name}")
