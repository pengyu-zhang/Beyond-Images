"""Standardise original dataset images to the canonical `QID_index.jpg` naming.

Replaces original scripts 1.2-1.4 and 1.6 (rename/consolidate, with a fuzzy
second pass) and 1.5 (DB15K search-engine image download from mmkb URL lists).

Fixes over the originals:
  - deterministic image order (sorted file listing instead of os.listdir),
  - exact-name and normalised matching in one pass, fuzzy matching as fallback,
  - single implementation for all datasets.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image, ImageFile

from ..utils.jsonl import JsonlWriter, completed_keys
from ..utils.web import make_session

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _normalise(name: str) -> str:
    name = re.sub(r"[._,()]", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def consolidate_images(
    images_root: str | Path,
    qid_map: dict[str, str],
    output_dir: str | Path,
    fuzzy_threshold: float = 0.8,
    log_path: str | Path | None = None,
) -> dict[str, int]:
    """Copy per-entity image folders into one flat folder named `QID_idx.jpg`.

    `qid_map` maps DBpedia entity names to QIDs. Folders are matched exactly,
    then with the original `__` -> `:_` fix-up, then by normalised fuzzy ratio.
    """
    images_root, output_dir = Path(images_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalised_map = {_normalise(k): v for k, v in qid_map.items()}
    folders = sorted(p for p in images_root.iterdir() if p.is_dir())
    stats = {"folders": len(folders), "matched": 0, "fuzzy": 0, "unmatched": 0, "images": 0}
    unmatched: list[str] = []

    for folder in folders:
        qid = qid_map.get(folder.name)
        if qid is None and "__" in folder.name:
            qid = qid_map.get(folder.name.replace("__", ":_", 1))
        if qid is None:
            qid = normalised_map.get(_normalise(folder.name))
        if qid is None and fuzzy_threshold:
            candidate = _normalise(folder.name)
            best_ratio, best_qid = 0.0, None
            for name, mapped in normalised_map.items():
                ratio = SequenceMatcher(None, candidate, name).ratio()
                if ratio > best_ratio:
                    best_ratio, best_qid = ratio, mapped
            if best_ratio >= fuzzy_threshold:
                qid = best_qid
                stats["fuzzy"] += 1
        if qid is None:
            stats["unmatched"] += 1
            unmatched.append(folder.name)
            continue

        stats["matched"] += 1
        for idx, image_path in enumerate(sorted(p for p in folder.iterdir() if p.is_file())):
            shutil.copy2(image_path, output_dir / f"{qid}_{idx}.jpg")
            stats["images"] += 1

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(
                f"folders={stats['folders']} matched={stats['matched']} "
                f"(fuzzy={stats['fuzzy']}) unmatched={stats['unmatched']}\n"
            )
            fh.writelines(name + "\n" for name in unmatched)
    return stats


def download_url_list_images(
    url_files: dict[str, str | Path],
    output_dir: str | Path,
    progress_jsonl: str | Path,
    num_images_per_provider: int = 20,
    max_size: int = 500,
    max_workers: int = 32,
    timeout: float = 30.0,
) -> dict[str, int]:
    """Download DB15K images from mmkb URL lists (`URLS_google.txt` etc.).

    `url_files` maps provider name -> URL list path with `url<TAB>freebase_id/index`
    rows. Images are resized to `max_size`, converted to JPEG, and stored as
    `<freebase_id>/<provider>_<index>.jpg`. Progress is journalled to
    `progress_jsonl` so re-runs skip completed downloads.
    """
    output_dir = Path(output_dir)
    session = make_session(retries=2)
    done = completed_keys(progress_jsonl, "key")
    stats = {"requested": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    tasks: list[tuple[str, str, str, int]] = []
    for provider, url_file in url_files.items():
        with open(url_file, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                url, ident = parts
                freebase_id, index = ident.rsplit("/", 1)
                if int(index) >= num_images_per_provider:
                    continue
                key = f"{freebase_id}/{provider}_{index}"
                stats["requested"] += 1
                if key in done:
                    stats["skipped"] += 1
                    continue
                tasks.append((provider, url, freebase_id, int(index)))

    def fetch(task: tuple[str, str, str, int]) -> tuple[str, bool]:
        provider, url, freebase_id, index = task
        key = f"{freebase_id}/{provider}_{index}"
        target_dir = output_dir / freebase_id.strip("/").replace("/", ".")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{provider}_{index}.jpg"
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
            image.thumbnail((max_size, max_size), Image.LANCZOS)
            image.convert("RGB").save(target, "JPEG")
            return key, True
        except Exception:
            return key, False

    with JsonlWriter(progress_jsonl) as writer:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, (key, ok) in enumerate(pool.map(fetch, tasks), 1):
                writer.write({"key": key, "ok": ok})
                stats["downloaded" if ok else "failed"] += 1
                if idx % 200 == 0:
                    print(f"[db15k-images] {idx}/{len(tasks)} ({stats['failed']} failed)")
    return stats
