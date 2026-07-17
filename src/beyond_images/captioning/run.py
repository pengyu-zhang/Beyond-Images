"""Caption every image in a folder, writing an incremental JSONL journal.

Fixes over the original 4.x scripts:
  - sorted, deterministic file order (os.listdir order is arbitrary),
  - resume: images already present in the output journal are skipped,
    instead of appending duplicate lines on re-runs,
  - corrupt images are skipped per-image, not per-batch.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..utils.jsonl import JsonlWriter, completed_keys
from .captioners import Captioner, load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def caption_folder(
    captioner: Captioner,
    image_dir: str | Path,
    output_jsonl: str | Path,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict[str, int]:
    image_dir = Path(image_dir)
    files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        files = files[:limit]

    done = completed_keys(output_jsonl, "image")
    todo = [p for p in files if p.name not in done]
    print(f"[caption] {len(files)} images, {len(done)} done, {len(todo)} to caption")

    stats = {"images": len(files), "captioned": 0, "skipped_corrupt": 0}
    started = time.time()
    with JsonlWriter(output_jsonl) as writer:
        for start in range(0, len(todo), batch_size):
            batch_paths = todo[start : start + batch_size]
            batch_images, kept_paths = [], []
            for path in batch_paths:
                image = load_image(path)
                if image is None:
                    stats["skipped_corrupt"] += 1
                    writer.write({"image": path.name, "caption": None, "corrupt": True})
                    continue
                batch_images.append(image)
                kept_paths.append(path)
            if not batch_images:
                continue
            captions = captioner.caption_batch(batch_images)
            for path, caption in zip(kept_paths, captions):
                writer.write({"image": path.name, "caption": caption})
                stats["captioned"] += 1
            done_count = start + len(batch_paths)
            rate = stats["captioned"] / max(time.time() - started, 1e-6)
            print(f"[caption] {done_count}/{len(todo)} ({rate:.1f} img/s)")
    return stats


def export_captions_txt(output_jsonl: str | Path, txt_path: str | Path) -> int:
    """Export the journal to the original `name.jpg: caption` text format."""
    count = 0
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    from ..utils.jsonl import read_jsonl

    with open(txt_path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in read_jsonl(output_jsonl):
            if rec.get("caption"):
                fh.write(f"{rec['image']}: {rec['caption']}\n")
                count += 1
    return count
