# Image-Text Consistency Checker

Local-first Streamlit interface for auditing whether the LLM-fused entity
summaries match each entity's image set (paper Section 4.4 / supplementary
Section 7). A browser-only preview with bundled examples is hosted as the
project's [live demo](https://pengyu-zhang.github.io/Beyond-Images/).

## Run

```bash
pip install streamlit pillow
streamlit run tools/checker/app.py
```

On first launch the Setup page asks for:

- **Text JSON**: an entity summary file, e.g. the fused output
  `outputs/MKG-W/MKG-W_fused.json` or a released
  `*_img_summary_blip.json` file (the fused
  `images.images_t5_descriptions` field is shown when present, otherwise
  `images.merged_descriptions`).
- **Images folder**: a flat folder of `QID_index.jpg` files.

## Features

- Match / Mismatch / Uncertain verdicts with mismatch reasons and notes
- Per-image soft removal (physical deletion is an opt-in setting)
- Filters (Pending / Reviewed / Mismatch / Uncertain), quick jump, progress stats
- Keyboard shortcuts: `Ctrl+S` save, `←` / `→` navigate
- Autosave on navigation; curation stored as a `_curation` object in
  `dataset_curated.json` next to the input file
- Export "modified only" or the full curated dataset

All I/O is on local disk; no external services are called.
