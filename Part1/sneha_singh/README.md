# Part 1 — Sneha Singh

Multimodal RAG over the three read-only economics PDFs (Lab 2 Task 1).

## Deliverables (lab PDF)

| Item | Location |
| --- | --- |
| Notebook (Kaggle + local) | `notebooks/sneha_singh/p1_rag.ipynb` |
| Practice / selected CSV | `Part1/sneha_singh/submission.csv` |
| Validator | `Part1/validate_submission.py` |
| Source / trust notes | `Part1/sneha_singh/source_notes.md` |
| Results + ablation | `Part1/sneha_singh/results.md` |
| Golden set (team) | `eval/golden_set.json` |
| Human groundedness form | `eval/human_review.csv` |
| Architecture figure | `report/figures/part1/architecture.png` |
| Report section | `report/report.md` §II |
| Kaggle link | `links.txt` → `lab2-notebook` |
| Tag | `p1-v1.0` (retag after executed Kaggle notebook is committed) |

## How to run

**Kaggle (graded path):** open `p1_rag.ipynb`, attach competition data + `validate_submission.py`, GPU + internet, Run All ≤90 min. Submit `/kaggle/working/submission.csv`. No question-ID or question-text hardcodes.

**Local:** Ollama `qwen2.5vl:3b` + `qwen2.5:7b`, PDFs in `Part1/data/`, Run All. Optional `GENERATE_LOCAL_SUBMISSION=1`.

## Public LB

Best practice score **0.860** (`Pair_Programming_Team_05`). Final Task 1.5 marks use the hidden set through the tagged notebook.

## Still for both members before Canvas zip

1. Fill `groundedness_1_to_5` in `eval/human_review.csv` (both raters).
2. After Kaggle Save Version finishes: commit executed notebook + notebook CSV, retag `p1-v1.0`.
