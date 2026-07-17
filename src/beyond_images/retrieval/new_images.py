"""Modality Extension Module: crawl additional entity images from Wikipedia.

Replaces original scripts 2.1, 2.2, 2.5. For every entity QID:
  1. resolve its English Wikipedia page,
  2. list the images used on that page,
  3. download each image and record provenance metadata
     (page URL, image URL, and the file's Summary table when available).

Two backends, selected via config:
  - "api":  MediaWiki / Wikidata APIs (default; robust, returns structured
            extmetadata such as description, author, and date).
  - "html": scrape wikidata.org / wikipedia.org pages, reproducing the
            original scripts' behaviour.

Output schema per image matches the released dataset:
  {"id": "<QID>_<idx>", "wikidata_url": ..., "page_url": ...,
   "image_url": ..., "summary": {...}}
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..utils.jsonl import JsonlWriter, completed_keys, read_jsonl, save_json_atomic
from ..utils.web import make_session

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


# --------------------------------------------------------------------------- api
def _enwiki_title_api(session, qid: str, timeout: float) -> str | None:
    resp = session.get(
        WIKIDATA_API,
        params={
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    entity = resp.json().get("entities", {}).get(qid, {})
    return entity.get("sitelinks", {}).get("enwiki", {}).get("title")


def _page_images_api(session, title: str, timeout: float) -> list[dict[str, Any]]:
    """List images on an enwiki page with URL + extmetadata via the API."""
    endpoint = "https://en.wikipedia.org/w/api.php"
    files: list[str] = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "images",
        "imlimit": "max",
        "format": "json",
    }
    while True:
        resp = session.get(endpoint, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        for page in payload.get("query", {}).get("pages", {}).values():
            files.extend(img["title"] for img in page.get("images", []))
        cont = payload.get("continue")
        if not cont:
            break
        params.update(cont)

    results: list[dict[str, Any]] = []
    for chunk_start in range(0, len(files), 50):
        chunk = files[chunk_start : chunk_start + 50]
        resp = session.get(
            endpoint,
            params={
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": 800,  # rasterised thumb (SVG logos become PNG)
                "format": "json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("url")
            if not url or not url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg")):
                continue
            if info.get("width", 0) < 80 or info.get("height", 0) < 80:
                continue  # UI icons and decorations
            meta = info.get("extmetadata", {})
            summary = {
                key: (meta.get(source, {}) or {}).get("value", "")
                for key, source in (
                    ("Description", "ImageDescription"),
                    ("Date", "DateTime"),
                    ("Author", "Artist"),
                    ("License", "LicenseShortName"),
                )
                if (meta.get(source, {}) or {}).get("value")
            }
            results.append(
                {
                    "page_url": info.get("descriptionurl", url),
                    "image_url": url,
                    "download_url": info.get("thumburl") or url,
                    "summary": summary,
                }
            )
    return results


# -------------------------------------------------------------------------- html
def _enwiki_url_html(session, qid: str, timeout: float) -> str | None:
    resp = session.get(f"https://www.wikidata.org/wiki/{qid}", timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    sitelinks = soup.find("div", {"class": "wikibase-sitelinklistview"})
    if sitelinks:
        enwiki = sitelinks.find("li", {"class": "wikibase-sitelinkview-enwiki"})
        if enwiki and enwiki.find("a"):
            return enwiki.find("a")["href"]
    return None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _page_images_html(session, wikipedia_url: str, timeout: float) -> list[dict[str, Any]]:
    resp = session.get(wikipedia_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    detail_pages = [
        urljoin(wikipedia_url, anchor["href"])
        for anchor in soup.find_all("a", {"class": "mw-file-description"})
        if anchor.get("href")
    ]

    results: list[dict[str, Any]] = []
    for page_url in detail_pages:
        try:
            page = session.get(page_url, timeout=timeout)
            page.raise_for_status()
        except Exception:
            continue
        page_soup = BeautifulSoup(page.text, "html.parser")

        image_url = None
        container = page_soup.find("div", {"class": "fullImageLink"})
        if container and container.find("img") and container.find("img").get("src"):
            src = container.find("img")["src"]
            image_url = "https:" + src if src.startswith("//") else src

        summary: dict[str, str] = {}
        for table in page_soup.find_all("table"):
            for row in table.find_all("tr"):
                header = row.find("th") or row.find("td", {"class": "fileinfo-paramfield"})
                cells = row.find_all("td")
                if header and cells:
                    value = cells[1] if len(cells) > 1 else cells[0]
                    summary[_clean_text(header.text)] = _clean_text(value.text)

        results.append({"page_url": page_url, "image_url": image_url, "summary": summary})
    return results


# --------------------------------------------------------------------------- run
def crawl_new_images(
    entities: list[tuple[str, str]],
    images_dir: str | Path,
    metadata_jsonl: str | Path,
    lookup: str = "api",
    max_workers: int = 10,
    timeout: float = 20.0,
    retries: int = 3,
    user_agent: str | None = None,
    max_images_per_entity: int = 0,
    download: bool = True,
) -> dict[str, int]:
    """Crawl Wikipedia images for `entities` = [(wikidata_url, qid), ...].

    Appends one metadata record per entity to `metadata_jsonl` (resume-safe)
    and optionally downloads images to `images_dir` as `QID_idx.jpg`.
    """
    images_dir = Path(images_dir)
    if download:
        images_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(retries=retries, user_agent=user_agent or make_session().headers["User-Agent"])

    done = completed_keys(metadata_jsonl, "qid")
    todo = [e for e in entities if e[1] not in done]
    print(f"[new-images] {len(entities)} entities, {len(done)} done, {len(todo)} to crawl")
    stats = {"entities": 0, "images": 0, "failed_entities": 0, "failed_downloads": 0}

    def crawl_one(entity: tuple[str, str]) -> dict[str, Any]:
        wikidata_url, qid = entity
        try:
            if lookup == "api":
                title = _enwiki_title_api(session, qid, timeout)
                images = _page_images_api(session, title, timeout) if title else []
            else:
                page_url = _enwiki_url_html(session, qid, timeout)
                images = _page_images_html(session, page_url, timeout) if page_url else []
        except Exception:
            images = []

        records = []
        if max_images_per_entity:
            images = images[:max_images_per_entity]
        for idx, info in enumerate(images, start=1):
            record = {
                "id": f"{qid}_{idx}",
                "wikidata_url": wikidata_url,
                "page_url": info["page_url"],
                "image_url": info["image_url"],
                "summary": info["summary"],
            }
            if info.get("download_url") and info["download_url"] != info["image_url"]:
                record["download_url"] = info["download_url"]
            if download and info["image_url"]:
                try:
                    resp = session.get(info.get("download_url") or info["image_url"], timeout=timeout)
                    resp.raise_for_status()
                    (images_dir / f"{qid}_{idx}.jpg").write_bytes(resp.content)
                except Exception:
                    record["download_failed"] = True
            records.append(record)
        return {"qid": qid, "wikidata_url": wikidata_url, "images": records}

    with JsonlWriter(metadata_jsonl) as writer:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(crawl_one, entity): entity for entity in todo}
            for idx, future in enumerate(cf.as_completed(futures), 1):
                result = future.result()
                writer.write(result)
                stats["entities"] += 1
                stats["images"] += len(result["images"])
                if not result["images"]:
                    stats["failed_entities"] += 1
                stats["failed_downloads"] += sum(
                    1 for rec in result["images"] if rec.get("download_failed")
                )
                if idx % 25 == 0:
                    print(f"[new-images] {idx}/{len(todo)} entities crawled")
    return stats


def export_metadata_json(metadata_jsonl: str | Path, output_json: str | Path) -> int:
    """Flatten the per-entity crawl journal into the released flat JSON list."""
    flat: list[dict[str, Any]] = []
    for rec in read_jsonl(metadata_jsonl):
        flat.extend(rec["images"])
    flat.sort(key=lambda item: item["id"])
    save_json_atomic(flat, output_json)
    return len(flat)
