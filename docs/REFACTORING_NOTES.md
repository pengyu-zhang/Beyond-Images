# Refactoring Notes

This document records how the published code (`original_code/`) was
reorganised into the current repository, every behavioural difference, and
all bugs found during the rewrite. The archived scripts are kept verbatim
for reference and are never modified.

## 1. What this repository is (and is not)

Beyond Images is a **data-side enrichment pipeline**. It produces enriched
textual inputs for existing MMKG completion models; it does **not** contain
or train those models. The link-prediction experiments in the paper were run
in the four upstream repositories with their default hyperparameters,
feeding them the enriched files this pipeline produces:

| Model | Repository | Enriched input consumed |
| --- | --- | --- |
| MMRNS | <https://github.com/quqxui/MMRNS> | `*_description_sentences.h5` |
| MyGO | <https://github.com/zjukg/MyGO> | token-id JSON (merged) |
| NativE | <https://github.com/zjukg/NATIVE> | text embeddings (`.pth`) |
| AdaMF | <https://github.com/zjukg/AdaMF-MAT> | `*-textual.pth` |

The paper's full result tables (Table 2/3/5) and the supplementary
training-time table are the reference numbers; this repository does not
attempt to reproduce them locally.

## 2. Mapping from the original scripts

The original `code/` used a numbered layout with gaps (01, 02, 04, 05, 08,
09 — 03, 06, 07 and 8.1 never existed in the published version). The
rewrite reorganises by pipeline module instead:

| Original | New location | Notes |
| --- | --- | --- |
| 1.1, 2.4 (DBpedia→Wikidata scrape) | `retrieval/entity_links.py` | + JSON-endpoint backend |
| 1.2, 1.3, 1.4, 1.6 (rename/match) | `retrieval/original_images.py` | one implementation, fuzzy fallback |
| 1.5 (DB15K downloader) | `retrieval/original_images.py` | requests + resume journal |
| 2.1, 2.2, 2.5 (Wikipedia crawl) | `retrieval/new_images.py` | + MediaWiki API backend |
| 2.3 (SameAsLink transform) | `retrieval/entity_links.py` | |
| 2.6 (rename by metadata) | superseded | crawler names files `QID_idx.jpg` directly |
| 4.1–4.3 (BLIP-2 captioning) | `captioning/captioners.py` + `run.py` | + GIT & LLaVA backends |
| 4.4–4.6 (caption merge) | `captioning/merge.py` | bug fixes below |
| 5.1, 5.4 (inspection utilities) | dropped | one-off diagnostics |
| 5.2, 5.3 (BERT tokens + merge) | `embedding/tokens.py` | tokenizer only, no model load |
| 5.5 (text → h5/pth) | `embedding/encode.py` | batched GPU encoding |
| 8.2–8.4 (LLM fusion) | `fusion/fusers.py` + `run.py` | one config-driven implementation |
| 9 (checker) | `tools/checker/app.py` | near-verbatim port |

## 3. Bugs found in the original code (fixed in the rewrite)

1. **QID prefix collision** (`4.4_MKG-W_img_summary.py`): captions were
   matched to entities with `image_name.startswith(entity_qid)`, so `Q42`
   also collected captions belonging to `Q420`, `Q423`, etc. The MKG-Y and
   DB15K variants (4.5/4.6) already used exact base-QID matching; MKG-W did
   not. The rewrite always matches the exact QID before the underscore.
2. **First caption dropped** (4.4–4.6): the caption text file was read with
   `header=0` although it has no header row, silently discarding the first
   image's caption in every run.
3. **BLIP-2 loading crashes on current libraries** (4.1–4.3):
   `from_pretrained(..., device_map="auto").to(device)` raises in recent
   accelerate versions, and the manual `<image>` token surgery
   (`resize_token_embeddings`, `image_token_index` patching) is unnecessary
   in current transformers and can corrupt generation.
4. **Append-mode duplication** (4.1–4.3): output files were opened with
   mode `"a"`, so re-running a job duplicated caption lines. The rewrite
   journals to JSONL and skips completed images.
5. **Non-deterministic image order** (1.2, 1.4, 1.6, 4.1–4.3):
   `os.listdir` order is filesystem-dependent, so `QID_idx` assignments
   were not reproducible. All listings are now sorted.
6. **Embedding dimension mismatch** (5.5): the fallback path emitted
   384-dim zero vectors while `bert-base-uncased` produces 768-dim
   embeddings; comments also claimed 384. The dimension now always comes
   from the model.
7. **Implicit `.pth` row order** (5.5): the AdaMF tensor rows relied on
   JSON key order with no record of which row is which entity. The rewrite
   writes a `.entities.txt` manifest next to the tensor.
8. **Missing HTTP timeouts** (2.x): several `requests.get` calls had no
   timeout and could hang a worker forever. All requests now use a shared
   session with timeouts, bounded retries, and a descriptive User-Agent.
9. **Unnecessary full-model load** (5.2): the script loaded the entire BERT
   model to use only its tokenizer.
10. **Wrong requirements file**: the published `requirements.txt` listed
    packages from an unrelated project (flair, pysolr, segtok, nltk, emoji,
    …) and omitted actual dependencies (requests, beautifulsoup4,
    sentence-transformers, h5py, streamlit, Pillow). Rewritten from actual
    imports.

## 4. Components described in the paper but missing from the code

- **GIT and LLaVA captioners** (paper §3.2, Table 5): only BLIP-2 scripts
  were published. Implemented as `git` / `llava` captioning backends
  (`microsoft/git-large-coco`, `llava-hf/llava-1.5-7b-hf`). Note that GIT
  is a pure captioning model and ignores the text prompt; the paper's fixed
  prompt applies to BLIP-2 and LLaVA only.
- **NativE input conversion**: the original 05 scripts covered MMRNS (h5),
  AdaMF (pth), and MyGO (tokens). NativE consumes text embeddings in the
  same `.pth` form; use the `embed` stage output.

## 5. Choices the paper does not specify (now configurable)

- **Decoding strategy for fusion LLMs**: the original Mistral/Llama scripts
  sampled (`do_sample=True, temperature=0.6, top_p=0.9`), which is not
  reproducible run-to-run; the T5 script used beam search. The paper is
  silent on this. `fusion.deterministic` (default `true` in `paper.yaml` /
  `default.yaml`, `false` in `original.yaml`) switches between greedy/beam
  and the original sampling parameters.
- **Caption cap and priority** (`MAX_DESCRIPTIONS_PER_ENTITY = 500`,
  original-image captions filling the cap first): kept as
  `fusion.max_descriptions_per_entity` and `fusion.priority_index`.
- **Caption cleaning** (dedup, <3-word drop, repetition filter): kept
  exactly as in 8.x (`fusion/run.py:clean_descriptions`).
- **Fusion prompts**: the T5 path uses the paper's prompt verbatim; the
  chat-model path uses the original scripts' system+user chat prompt
  (slightly different wording from the paper's single prompt). Both are in
  `fusion/fusers.py`.
- **Fused output key**: the original scripts stored every model's output
  under `images.images_t5_descriptions` (even Mistral/Llama). Kept for
  compatibility with the released data and the checker.

## 6. Enhancements beyond the original (all switchable)

- **Lookup backends**: `retrieval.dbpedia_lookup: json|html` and
  `retrieval.wikipedia_lookup: api|html`. The `html` settings reproduce the
  original scraping; `json`/`api` (default) use the DBpedia JSON endpoint
  and MediaWiki APIs, which are faster, more robust, and return structured
  image metadata (description/author/date/license).
- **SVG-safe downloads**: Wikipedia pages frequently embed SVG logos and
  coats of arms; raw SVG bytes saved as `.jpg` are unreadable by PIL (in a
  demo run, 40 of 64 images failed this way). The API backend downloads an
  800 px rasterised thumbnail instead and skips icon-sized files (<80 px).
- **Quantization**: `captioning.quantization` / `fusion.quantization`
  (`none|8bit|4bit`, needs bitsandbytes) to fit blip2-flan-t5-xxl or
  Mistral-7B on consumer GPUs.
- **Resume everywhere**: crawl, caption, fusion, and download stages write
  JSONL journals and skip completed units on re-run.
- **TF32 toggle** (`run.tf32`, on in `default.yaml`, off in
  `original.yaml`/`paper.yaml`), fixed seeds, device banner, and per-stage
  metrics appended to `outputs/metrics.jsonl`.

## 7. Verification performed

All runs on Windows 11 / RTX 5060 Laptop (8 GB) / Python 3.13 /
PyTorch 2.13 cu132 / transformers 5.14, executed through Git Bash.

- **Smoke test** (`scripts/smoke_test.sh`): bundled MKG-W sample → fusion
  (flan-t5-base, 5 entities, 13.9 s) → embeddings (h5 + pth + manifest,
  768-dim) → MyGO tokens (CLS/SEP verified). PASSED.
- **Live crawl + caption + merge + fuse** (8 entities): MediaWiki API
  crawl retrieved 63 images with metadata in 25 s (0 failures); BLIP
  captioning 13.6 img/s on GPU (0 corrupt after the SVG fix); merge
  attached all 63 captions to the correct 8 entities; fusion produced 8
  non-empty summaries. This run's real output ships as the live demo data
  (`demo/demo_data.js`).
- **Checker**: `tools/checker/app.py` compiles and runs against the fused
  output; read fallback to `merged_descriptions` added.

Not verified locally: full-scale captioning with blip2-flan-t5-xxl and
fusion with Mistral-7B/Llama-3.1 (exceed 8 GB VRAM un-quantized; Mistral
and Llama are additionally gated on Hugging Face), and any downstream
link-prediction training (out of scope, see §1).

## 8. Reference numbers from the paper

Link prediction with the enriched data (Fusion row) vs. the original
datasets, as reported in the paper (Table 2):

| Model | MKG-W MRR | MKG-Y MRR | DB15K MRR |
| --- | --- | --- | --- |
| MMRNS | 35.03 → 37.04 (+5.74%) | 35.93 → 37.54 (+4.48%) | 32.68 → 34.47 (+5.47%) |
| MyGO | 36.10 → 38.05 (+5.41%) | 38.51 → 40.77 (+5.87%) | 37.72 → 39.38 (+4.40%) |
| NativE | 36.58 → 38.04 (+3.98%) | 39.04 → 40.38 (+3.43%) | 37.16 → 39.55 (+6.42%) |
| AdaMF | 35.85 → 38.04 (+6.11%) | 38.57 → 40.54 (+5.10%) | 35.14 → 36.74 (+4.56%) |

On the 20-entity logo/symbol subset (Table 3): MRR 13.89 → 41.87
(+201.35%), Hits@1 7.50 → 32.50 (+333.33%).

## 9. Repository history

The original layout (numbered scripts, SLURM `run.sh`, old
`requirements.txt`) is preserved unchanged under `original_code/`, moved
with `git mv` so file history is intact.
