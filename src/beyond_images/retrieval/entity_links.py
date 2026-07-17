"""Entity alignment: map dataset entities (DBpedia URIs / Freebase MIDs) to Wikidata QIDs.

Replaces original scripts 1.1, 2.3, 2.4. Two lookup backends:
  - "json": query DBpedia's JSON endpoint (https://dbpedia.org/data/<name>.json)
            and read owl:sameAs links. Robust and fast (config default).
  - "html": scrape the DBpedia resource page for rel="owl:sameAs" anchors,
            reproducing the original scripts' behaviour.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from urllib.parse import quote, unquote

from bs4 import BeautifulSoup

from ..utils.jsonl import JsonlWriter, completed_keys, read_jsonl
from ..utils.web import make_session

WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def transform_sameas_links(input_file: str | Path, output_file: str | Path) -> int:
    """Convert mmkb DB15K_SameAsLink.txt lines into `dbpedia_uri<TAB>mid` rows."""
    count = 0
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as fin, open(
        output_file, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            parts = line.split()
            # Expected: /m/xxx <SameAs> <http://dbpedia.org/resource/yyy> .
            if len(parts) != 4:
                continue
            mid = parts[0]
            dbpedia_uri = parts[2].strip("<>")
            fout.write(f"{dbpedia_uri}\t{mid}\n")
            count += 1
    return count


def _wikidata_from_json(session, dbpedia_url: str, timeout: float) -> str | None:
    name = dbpedia_url.rsplit("/", 1)[-1]
    api_url = f"https://dbpedia.org/data/{quote(unquote(name), safe='')}.json"
    resp = session.get(api_url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    resource_key = f"http://dbpedia.org/resource/{unquote(name)}"
    node = data.get(resource_key) or next(iter(data.values()), {})
    for link in node.get("http://www.w3.org/2002/07/owl#sameAs", []):
        value = link.get("value", "")
        if "wikidata.org" in value:
            return value
    return None


def _wikidata_from_html(session, dbpedia_url: str, timeout: float) -> str | None:
    resp = session.get(dbpedia_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for anchor in soup.find_all("a", {"rel": "owl:sameAs"}):
        href = anchor.get("href", "")
        if "wikidata.org" in href:
            return href
    return None


def resolve_wikidata_links(
    input_file: str | Path,
    output_jsonl: str | Path,
    lookup: str = "json",
    max_workers: int = 10,
    timeout: float = 20.0,
    retries: int = 3,
    user_agent: str | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """Resolve each `dbpedia_url<TAB>dataset_id` row to a Wikidata URL.

    Appends {dbpedia_url, dataset_id, wikidata_url} records to `output_jsonl`;
    already-resolved URLs are skipped so interrupted runs resume for free.
    Returns (resolved, failed) counts for this invocation.
    """
    session = make_session(retries=retries, user_agent=user_agent or make_session().headers["User-Agent"])
    fetch = _wikidata_from_json if lookup == "json" else _wikidata_from_html

    rows: list[tuple[str, str]] = []
    with open(input_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    if limit:
        rows = rows[:limit]

    done = completed_keys(output_jsonl, "dbpedia_url")
    todo = [row for row in rows if row[0] not in done]
    print(f"[entity-links] {len(rows)} rows, {len(done)} already resolved, {len(todo)} to do")

    resolved = failed = 0
    with JsonlWriter(output_jsonl) as writer:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_safe_fetch, fetch, session, url, timeout): (url, ds_id)
                for url, ds_id in todo
            }
            for idx, future in enumerate(cf.as_completed(futures), 1):
                url, ds_id = futures[future]
                wikidata_url = future.result()
                writer.write(
                    {"dbpedia_url": url, "dataset_id": ds_id, "wikidata_url": wikidata_url}
                )
                if wikidata_url:
                    resolved += 1
                else:
                    failed += 1
                if idx % 50 == 0:
                    print(f"[entity-links] {idx}/{len(todo)} done ({failed} failed)")
    return resolved, failed


def _safe_fetch(fetch, session, url: str, timeout: float) -> str | None:
    try:
        return fetch(session, url, timeout)
    except Exception:
        return None


def export_links_tsv(jsonl_path: str | Path, tsv_path: str | Path) -> int:
    """Flatten the resolution JSONL into the classic 3-column ent_links TSV."""
    Path(tsv_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(tsv_path, "w", encoding="utf-8", newline="\n") as fout:
        for rec in read_jsonl(jsonl_path):
            wikidata = rec.get("wikidata_url") or ""
            fout.write(f"{rec['dbpedia_url']}\t{rec.get('dataset_id', '')}\t{wikidata}\n")
            count += 1
    return count


def load_qid_map(links_tsv: str | Path) -> dict[str, str]:
    """Read an ent_links TSV into {dbpedia_name: QID} (rows without QID skipped)."""
    mapping: dict[str, str] = {}
    with open(links_tsv, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2].startswith(WIKIDATA_ENTITY_PREFIX):
                name = parts[0].rsplit("/", 1)[-1]
                mapping[name] = parts[2].rsplit("/", 1)[-1]
    return mapping
