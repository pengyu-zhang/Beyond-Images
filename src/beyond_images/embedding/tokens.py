"""Export enriched text as BERT token ids for token-based MMKG models (MyGO).

Replaces original scripts 5.2 (tokenise) and 5.3 (merge into an existing
token file). 5.2 needlessly loaded the full BERT model just to tokenise;
only the tokenizer is loaded here.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.jsonl import load_json, save_json_atomic
from .encode import extract_texts

CLS, SEP = 101, 102  # bert-base-uncased special token ids


def tokenize_entities(
    entity_json: str | Path,
    output_json: str | Path,
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
    text_key: str = "auto",
) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = load_json(entity_json)
    texts = extract_texts(entity_json, text_key=text_key)

    output: dict[str, list[int]] = {}
    for entity_name, text in texts.items():
        record = data[entity_name]
        qid = record.get("entity_qid")
        if not qid or qid == "NAN":
            continue
        token_ids = tokenizer.encode(text, truncation=True, max_length=max_length)
        output[f"http://www.wikidata.org/entity/{qid}"] = token_ids

    save_json_atomic(output, output_json, indent=None)
    print(f"[tokens] wrote {output_json} ({len(output)} entities)")
    return len(output)


def merge_token_files(
    base_json: str | Path,
    extra_json: str | Path,
    output_json: str | Path,
) -> int:
    """Append `extra` token sequences into `base` per entity (MyGO format).

    Strips CLS/SEP from the extra sequence and splices it before the base
    sequence's final SEP, exactly as the original merge script did.
    """
    base = load_json(base_json)
    extra = load_json(extra_json)

    merged: dict[str, list[int]] = {}
    for key, base_tokens in base.items():
        if key in extra:
            extra_tokens = [t for t in extra[key] if t not in (CLS, SEP)]
            merged[key] = base_tokens[:-1] + extra_tokens + [SEP]
        else:
            merged[key] = base_tokens
    for key, extra_tokens in extra.items():
        if key not in base:
            merged[key] = [CLS] + [t for t in extra_tokens if t not in (CLS, SEP)] + [SEP]

    save_json_atomic(merged, output_json, indent=None)
    print(f"[tokens] merged -> {output_json} ({len(merged)} entities)")
    return len(merged)
