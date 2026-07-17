# Data

Only documentation and the small bundled sample live in git. Data bodies are
downloaded into `data/raw/` and `data/processed/` (both git-ignored).

## Released enriched datasets (ready to use)

The pipeline outputs for all three benchmarks — per-image BLIP-2 captions and
per-entity merged descriptions — are released on Zenodo and mirrored as a
GitHub Release asset:

| Source | Link |
| --- | --- |
| Zenodo (primary) | <https://doi.org/10.5281/zenodo.14847095> |
| GitHub Release (mirror) | `img_text_summary.zip` attached to release `v1.0` |

```bash
bash scripts/prepare_data.sh          # downloads (~128 MB), verifies MD5, extracts
```

`data/sample/img_text_summary.zip` (bundled, 4.4 MB) contains the two MKG-W
caption files used by the smoke test; `scripts/prepare_data.sh --sample`
extracts just this file.

MD5 checksums:

```
e5464dcb76501c10b692afd309139a50  img_text_summary.zip        (Zenodo / Release)
946951db0c58930d1108d50c8ab2d1e1  data/sample/img_text_summary.zip
```

License: the released enriched data is CC-BY-4.0. Captions were generated
from images collected from Wikipedia/Wikimedia and the source MMKG datasets;
retain their original attribution requirements when reusing.

## Source MMKG datasets (needed only to re-run the pipeline from scratch)

To regenerate everything (crawl + caption) you need the source datasets.
Place the files as follows:

### MKG-W and MKG-Y

From the [MMRNS repository](https://github.com/quqxui/MMRNS) (Google Drive
link in its README — manual download required):

```
data/raw/MKG-W/ent_links.tsv        # DBpedia <TAB> name <TAB> Wikidata URL
data/raw/MKG-W/images_original/     # original images, one folder per entity
data/raw/MKG-Y/...                  # same layout
```

If your copy has no DBpedia-to-Wikidata mapping (MKG-Y), build it with:

```bash
bash scripts/run_pipeline.sh links-resolve \
    --input data/raw/MKG-Y/ent_links_raw.tsv \
    --journal outputs/MKG-Y/links.jsonl --output data/raw/MKG-Y/ent_links.tsv
```

### DB15K

From the [mmkb repository](https://github.com/mniepert/mmkb) (BSD-3):

```bash
# 1. SameAsLink -> DBpedia/mid TSV
bash scripts/run_pipeline.sh links-transform \
    --input data/raw/DB15K/DB15K_SameAsLink.txt --output data/raw/DB15K/ent_links_raw.tsv
# 2. Resolve Wikidata QIDs
bash scripts/run_pipeline.sh links-resolve \
    --input data/raw/DB15K/ent_links_raw.tsv \
    --journal outputs/DB15K/links.jsonl --output data/raw/DB15K/ent_links.tsv
# 3. Download original images from the mmkb URL lists (URLS_google.txt, ...)
bash scripts/run_pipeline.sh db15k-download \
    --url-dir data/raw/DB15K/urls --output data/raw/DB15K/images_by_provider \
    --journal outputs/DB15K/downloads.jsonl
# 4. Consolidate to QID_idx.jpg naming
bash scripts/run_pipeline.sh consolidate \
    --images-root data/raw/DB15K/images_by_provider \
    --links data/raw/DB15K/ent_links.tsv --output data/raw/DB15K/images_original
```

## Layout summary

```
data/
├── README.md                  (this file)
├── sample/img_text_summary.zip  (bundled smoke-test sample, git-tracked)
├── raw/                       (git-ignored: source datasets + downloaded images)
└── processed/                 (git-ignored: released enriched data, extracted)
outputs/                       (git-ignored: pipeline outputs + metrics.jsonl)
```
