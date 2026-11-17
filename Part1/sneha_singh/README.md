# Part 1 — Sneha Singh

Multimodal RAG over the three read-only economics PDFs (Lab 2 Task 1).

## Current graded / selected path

| Item | Value |
| --- | --- |
| Notebook | `notebooks/sneha_singh/p1_rag.ipynb` (self-contained `mmrag` package written by the notebook) |
| Selected practice CSV | `Part1/sneha_singh/submission.csv` (sha `65d699b5…`, public LB **0.927**) |
| Team | `Pair_Programming_Team_05` |
| Kaggle | https://www.kaggle.com/code/snehasjsu/lab2-notebook/notebook?scriptVersionId=370792951 (v24; must be public or shared with instructor/ISAs) |
| Tag | Retag `p1-v1.0` to this notebook commit before deadline |

### Pipeline (selected)

PyMuPDF text + figure/table crops → Qwen2.5-VL figure descriptions → ChromaDB **`BAAI/bge-small-en-v1.5`** + BM25 (RRF) → exact-label routing, trust filter, precedence notes → **`Qwen/Qwen2.5-VL-7B-Instruct`** grounded JSON → `submission.csv`.

Final flags used for the **0.927** run: hybrid on, **rerank off** (CE was tested; it lowered LB), label routing / trust / precedence / status rules / figure images on.

## Deliverables (lab PDF)

| Item | Location |
| --- | --- |
| Notebook (Kaggle + local) | `notebooks/sneha_singh/p1_rag.ipynb` |
| Selected practice CSV | `Part1/sneha_singh/submission.csv` (sha `65d699b5…`, public LB **0.927**) |
| Validator | `Part1/validate_submission.py` |
| Source / trust notes | `Part1/sneha_singh/source_notes.md` |
| Results | `Part1/sneha_singh/results.md` |
| Golden set (team) | `eval/golden_set.json` |
| Human groundedness (both members) | `eval/human_review.csv` |
| Architecture figure | `report/figures/part1/architecture.png` |
| Report section | `report/report.md` §II |
| Kaggle link | `links.txt` |
| Tag | `p1-v1.0` |

## How to run

**Kaggle:** open `p1_rag.ipynb`, attach competition data, **GPU T4**, **Internet ON**, Run All ≤90 min. Submit `/kaggle/working/submission.csv` (or `export/submission.csv` from `p1_outputs.zip`). Share the notebook with instructor/ISAs or make it public.

**Local:** same notebook; PDFs via `Part1/data/` / `KB_DIR`.

## Public LB

Selected practice score **0.927** (`Pair_Programming_Team_05`). Task 1.5 uses the hidden set through the tagged notebook.
