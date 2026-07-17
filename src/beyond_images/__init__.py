"""Beyond Images: a data-centric enrichment pipeline for multi-modal knowledge graphs.

Pipeline stages (mirroring the paper):
  1. retrieval   - Modality Extension Module: align entities to Wikidata QIDs,
                   consolidate original images, crawl additional images from Wikipedia.
  2. captioning  - Semantic Generation Module: image-to-text descriptions
                   (BLIP-2 / GIT / LLaVA backends) and per-entity caption merging.
  3. fusion      - LLM-based Semantic Fusion Module: summarise all captions of an
                   entity into a single coherent paragraph (Flan-T5 / Mistral / Llama).
  4. embedding   - Export enriched text for downstream MMKG models:
                   sentence embeddings (.h5 for MMRNS, .pth for AdaMF) and
                   BERT token ids (.json for MyGO).
"""

__version__ = "1.0.0"
