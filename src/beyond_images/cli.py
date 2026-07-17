"""Command-line entry point for the Beyond Images pipeline.

Usage:
    python -m beyond_images <stage> --config configs/default.yaml [options]

Stages (in pipeline order):
    links-transform   mmkb SameAsLink file -> dbpedia/mid TSV
    links-resolve     DBpedia URIs -> Wikidata QIDs (resume-safe)
    consolidate       original per-entity image folders -> flat QID_idx.jpg
    db15k-download    download DB15K images from mmkb URL lists
    crawl             crawl new entity images + metadata from Wikipedia
    caption           image folder -> captions JSONL (BLIP-2 / GIT / LLaVA)
    merge             captions + entity links -> per-entity summary JSON
    fuse              entity summaries -> LLM-fused paragraphs
    embed             entity JSON -> .h5 / .pth embeddings (+ row manifest)
    tokens            entity JSON -> BERT token-id JSON (MyGO format)
    tokens-merge      splice enriched tokens into an existing token file
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config
from .utils.jsonl import JsonlWriter
from .utils.runtime import resolve_device, set_all_seeds


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value (dot notation), e.g. --set fusion.model=google/flan-t5-large",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N items")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beyond_images", description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("links-transform", help="mmkb SameAsLink -> TSV")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    _add_common(p)

    p = sub.add_parser("links-resolve", help="DBpedia -> Wikidata QIDs")
    p.add_argument("--input", required=True, help="TSV with dbpedia_url [<TAB>dataset_id]")
    p.add_argument("--journal", required=True, help="Resolution journal JSONL (resume-safe)")
    p.add_argument("--output", required=True, help="Final 3-column ent_links TSV")
    _add_common(p)

    p = sub.add_parser("consolidate", help="Original images -> QID_idx.jpg")
    p.add_argument("--images-root", required=True)
    p.add_argument("--links", required=True, help="ent_links TSV with QIDs")
    p.add_argument("--output", required=True)
    p.add_argument("--log", default=None)
    _add_common(p)

    p = sub.add_parser("db15k-download", help="Download DB15K search-engine images")
    p.add_argument("--url-dir", required=True, help="Folder containing URLS_<provider>.txt")
    p.add_argument("--output", required=True)
    p.add_argument("--journal", required=True)
    p.add_argument("--num-images", type=int, default=20)
    _add_common(p)

    p = sub.add_parser("crawl", help="Crawl new Wikipedia images per entity")
    p.add_argument("--links", required=True, help="ent_links TSV with QIDs")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--journal", required=True)
    p.add_argument("--metadata", default=None, help="Optional flat metadata JSON export")
    p.add_argument("--no-download", action="store_true", help="Record metadata only")
    _add_common(p)

    p = sub.add_parser("caption", help="Caption an image folder")
    p.add_argument("--images", required=True)
    p.add_argument("--output", required=True, help="Captions journal JSONL")
    _add_common(p)

    p = sub.add_parser("merge", help="Captions + links -> entity summary JSON")
    p.add_argument("--links", required=True)
    p.add_argument("--captions", required=True)
    p.add_argument("--output", required=True)
    _add_common(p)

    p = sub.add_parser("fuse", help="Entity summaries -> LLM-fused paragraphs")
    p.add_argument("--inputs", nargs="+", required=True, help="Entity summary JSON file(s)")
    p.add_argument("--journal", required=True)
    p.add_argument("--output", required=True)
    _add_common(p)

    p = sub.add_parser("embed", help="Entity JSON -> h5/pth embeddings")
    p.add_argument("--input", required=True)
    p.add_argument("--h5", default=None)
    p.add_argument("--pth", default=None)
    _add_common(p)

    p = sub.add_parser("tokens", help="Entity JSON -> BERT token ids (MyGO)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    _add_common(p)

    p = sub.add_parser("tokens-merge", help="Splice enriched tokens into a base token file")
    p.add_argument("--base", required=True)
    p.add_argument("--extra", required=True)
    p.add_argument("--output", required=True)
    _add_common(p)

    return parser


def _load_entities_for_crawl(links_tsv: str) -> list[tuple[str, str]]:
    entities = []
    with open(links_tsv, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and "/entity/Q" in parts[2]:
                entities.append((parts[2], parts[2].rsplit("/", 1)[-1]))
    return entities


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config, overrides=args.set)

    seed = cfg.get("run.seed", 42)
    set_all_seeds(seed)
    device = resolve_device(cfg.get("run.device", "auto"), tf32=cfg.get("run.tf32", True))
    print(f"[run] stage={args.stage} config={args.config} seed={seed}")

    started = time.time()
    stats: dict = {}

    if args.stage == "links-transform":
        from .retrieval.entity_links import transform_sameas_links

        stats["rows"] = transform_sameas_links(args.input, args.output)

    elif args.stage == "links-resolve":
        from .retrieval.entity_links import export_links_tsv, resolve_wikidata_links

        resolved, failed = resolve_wikidata_links(
            args.input,
            args.journal,
            lookup=cfg.get("retrieval.dbpedia_lookup", "json"),
            max_workers=cfg.get("retrieval.max_workers", 10),
            timeout=cfg.get("retrieval.timeout", 20),
            retries=cfg.get("retrieval.retries", 3),
            user_agent=cfg.get("retrieval.user_agent"),
            limit=args.limit,
        )
        stats.update(resolved=resolved, failed=failed, exported=export_links_tsv(args.journal, args.output))

    elif args.stage == "consolidate":
        from .retrieval.entity_links import load_qid_map
        from .retrieval.original_images import consolidate_images

        stats = consolidate_images(
            args.images_root,
            load_qid_map(args.links),
            args.output,
            fuzzy_threshold=cfg.get("retrieval.fuzzy_threshold", 0.8),
            log_path=args.log,
        )

    elif args.stage == "db15k-download":
        from .retrieval.original_images import download_url_list_images

        url_dir = Path(args.url_dir)
        url_files = {
            provider: url_dir / f"URLS_{provider}.txt"
            for provider in ("google", "bing", "yahoo")
            if (url_dir / f"URLS_{provider}.txt").exists()
        }
        stats = download_url_list_images(
            url_files,
            args.output,
            args.journal,
            num_images_per_provider=args.num_images,
            max_workers=cfg.get("retrieval.max_workers", 32),
            timeout=cfg.get("retrieval.timeout", 30),
        )

    elif args.stage == "crawl":
        from .retrieval.new_images import crawl_new_images, export_metadata_json

        entities = _load_entities_for_crawl(args.links)
        if args.limit:
            entities = entities[: args.limit]
        stats = crawl_new_images(
            entities,
            args.images_dir,
            args.journal,
            lookup=cfg.get("retrieval.wikipedia_lookup", "api"),
            max_workers=cfg.get("retrieval.max_workers", 10),
            timeout=cfg.get("retrieval.timeout", 20),
            retries=cfg.get("retrieval.retries", 3),
            user_agent=cfg.get("retrieval.user_agent"),
            max_images_per_entity=cfg.get("retrieval.max_images_per_entity", 0),
            download=not args.no_download,
        )
        if args.metadata:
            stats["metadata_records"] = export_metadata_json(args.journal, args.metadata)

    elif args.stage == "caption":
        from .captioning.captioners import build_captioner
        from .captioning.run import caption_folder

        captioner = build_captioner(cfg.section("captioning"), device)
        stats = caption_folder(
            captioner,
            args.images,
            args.output,
            batch_size=cfg.get("captioning.batch_size", 8),
            limit=args.limit,
        )

    elif args.stage == "merge":
        from .captioning.merge import merge_captions

        stats = merge_captions(args.links, args.captions, args.output)

    elif args.stage == "fuse":
        from .fusion.fusers import build_fuser
        from .fusion.run import collect_descriptions, fuse_entities

        merged = collect_descriptions(
            args.inputs,
            priority_index=cfg.get("fusion.priority_index", 0),
            max_per_entity=cfg.get("fusion.max_descriptions_per_entity", 500),
        )
        fuser = build_fuser(cfg.section("fusion"), device)
        stats = fuse_entities(fuser, merged, args.journal, args.output, limit=args.limit)

    elif args.stage == "embed":
        from .embedding.encode import encode_texts, extract_texts, write_outputs

        texts = extract_texts(args.input, text_key=cfg.get("embedding.text_key", "auto"))
        if args.limit:
            texts = dict(list(texts.items())[: args.limit])
        entities, embeddings = encode_texts(
            texts,
            model_name=cfg.get("embedding.model", "bert-base-uncased"),
            device=device,
            batch_size=cfg.get("embedding.batch_size", 256),
        )
        write_outputs(entities, embeddings, h5_path=args.h5, pth_path=args.pth)
        stats = {"entities": len(entities), "dim": int(embeddings.shape[1])}

    elif args.stage == "tokens":
        from .embedding.tokens import tokenize_entities

        stats["entities"] = tokenize_entities(
            args.input,
            args.output,
            model_name=cfg.get("embedding.tokens_model", "bert-base-uncased"),
            max_length=cfg.get("embedding.max_token_length", 512),
            text_key=cfg.get("embedding.text_key", "auto"),
        )

    elif args.stage == "tokens-merge":
        from .embedding.tokens import merge_token_files

        stats["entities"] = merge_token_files(args.base, args.extra, args.output)

    elapsed = time.time() - started
    print(f"[done] stage={args.stage} elapsed={elapsed:.1f}s stats={json.dumps(stats)}")

    metrics_path = Path(cfg.get("run.output_root", "outputs")) / "metrics.jsonl"
    with JsonlWriter(metrics_path) as writer:
        writer.write(
            {
                "stage": args.stage,
                "config": str(args.config),
                "seed": seed,
                "device": device,
                "elapsed_sec": round(elapsed, 2),
                "stats": stats,
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
