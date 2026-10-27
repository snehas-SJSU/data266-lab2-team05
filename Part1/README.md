# Part 1 — Multimodal RAG

Shared notebook: `notebooks/p1_rag.ipynb`.  
Member notes: `sneha_singh/results.md`, `ritika_neema/`.  
Golden set: `eval/golden_set.json`.  
Kaggle helper: `validate_submission.py`.

## Status for teammates

What is done on Part 1 so far, and what is still open. Same numbers as the notebook print of the passage-reranker ablation.

### Done (Sneha)

- Built and ran `notebooks/p1_rag.ipynb` locally (Ollama `qwen2.5:7b` answers, `qwen2.5vl:3b` figure captions).
- Parses the three PDFs, quarantines Appendix C, answers from text/figures/tables, keeps conversation edits out of the index.
- Architecture figure: `report/figures/part1/architecture.png`.
- Passage-reranker OFF vs ON ablation saved under `sneha_singh/outputs/evaluation/reranker_ablation/`.
- Graded tag `p1-v1.0` points at commit `1311cf4`.

### Latest notebook print (not a Kaggle score)

| Measure | Reranker OFF | Reranker ON |
| --- | --- | --- |
| Answers correct | 20/25 | 19/25 |
| Retrieval Recall@5 | 100% | 100% |
| All required sources in top 5 | 100% | 100% |
| Citation support—model judge | 94.4% | 97.3% |
| Median seconds/question | 55.63 | 50.12 |
| Generation / judge errors | 2 / 0 | 2 / 0 |

Retrieval is strong (100% Recall@5). Prefer **reranker OFF** as the main system for now. The lab +2 bonus needs the reranker to improve answers; ON is currently 19/25 against 20/25. **Do not retrain.** Part 1 is RAG; remaining work is pipeline bug fixes.

### Bugs to fix next

| ID | Problem |
| --- | --- |
| G04 | Arithmetic verify rejects a neighboring year instead of Table B-2 2007 unemployment 4.6 |
| G15 | `session_override` drops the temporary 6.0% Germany assumption |
| G20 | Revision event month/year not grounded |

Shared on both arms. Reranker OFF also fails G12 and G14. Reranker ON also loses G02, G05, and G08.

### Still open for both members

1. Review `eval/golden_set.json` together.
2. Rate groundedness 1–5 on the 25 final answers (both members).
3. Write Appendix C safety examples in `report/report.md`.
4. Fill the Part 1 results table in the report from a notebook print.
5. Timed Kaggle GPU run (≤ 90 minutes) and put the link in `links.txt`.
6. Ritika: record her own Part 1 pipeline run under `ritika_neema/`.

### How to read more

- Sneha detail: `sneha_singh/results.md`
- Team contribution log: `../CONTRIBUTIONS.md`
