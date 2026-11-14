# Part 1 — Sneha Singh

Multimodal RAG over the three read-only economics PDFs (Lab 2 Task 1).

## Deliverables (lab PDF)

| Item | Location |
| --- | --- |
| Notebook (Kaggle + local) | `notebooks/sneha_singh/p1_rag.ipynb` |
| Selected CSV (best public LB) | `Part1/sneha_singh/submission.csv` (sha `881a9ffb…`, score **0.940**; distinct from this Save Version notebook output sha `47948d4e…`) |
| Validator | `Part1/validate_submission.py` |
| Source / trust notes | `Part1/sneha_singh/source_notes.md` |
| Results + ablation | `Part1/sneha_singh/results.md` |
| Golden set (team) | `eval/golden_set.json` (Sneha Singh and Ritika Neema) |
| Human groundedness (both members, done) | `eval/human_review.csv` (mean 4.88 each) |
| Architecture figure | `report/figures/part1/architecture.png` |
| Report section | `report/report.md` §II |
| Kaggle link | `links.txt` → `lab2-notebook` |
| Tag | `p1-v1.0` (executed Kaggle notebook + matching CSV) |

## How to run

**Kaggle (graded path):** open `p1_rag.ipynb`, attach competition data + `validate_submission.py`, GPU + internet, Run All ≤90 min. Submit `/kaggle/working/submission.csv`. No question-ID or question-text hardcodes.

**Local:** Ollama `qwen2.5vl:3b` + `qwen2.5:7b`, PDFs in `Part1/data/`, Run All. Optional `GENERATE_LOCAL_SUBMISSION=1`.

## Public LB

Best practice score **0.940** (`Pair_Programming_Team_05`). Final Task 1.5 marks use the hidden set through the tagged notebook.

## Status

Tag `p1-v1.0` points at the post-hardcode-cleanup executed Kaggle notebook. Reported public practice LB best is **0.940** from selected `submission.csv` (`881a9ffb…`), which is not the same bytes as this Save Version output hash (`47948d4e…`).
