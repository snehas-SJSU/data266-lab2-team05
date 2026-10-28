# Ritika Neema - Part 1

My own Part 1 pipeline is `notebooks/ritika_neema/p1_rag.ipynb`. The golden set is
`eval/ritika_neema/p1_golden_set.json`, written before any tuning; none of its 25 questions come from
`Questions.csv`.

## Pipeline

| Step | What it does |
| --- | --- |
| Ingestion | PyMuPDF. Text chunks of about 120 words keep `doc_id`, `page` and `trust_level`. Each numbered Figure and Table is found from its caption (label set in a heading font) and saved as a cropped PNG with its label and caption. Raster charts (DOC2, DOC3) are cropped to the image under the caption; vector charts (DOC1) to the drawings below the caption. Tables are rebuilt from PyMuPDF's table detection as one labelled line per row (`row: column = value`), followed by the notes under the table and the page sentences that explain it (v2). |
| Trust | From `kb_manifest.json`, including the page override that makes DOC3 Appendix C (p7) low trust. |
| Figures | `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit) writes a data description of each figure. Low-trust figures are not described. |
| Index | `BAAI/bge-small-en-v1.5` embeddings in ChromaDB (metadata `doc_id`, `page`, `figure_id`, `trust_level`) and BM25 over the same records. |
| Retrieval | Baseline: dense only. Final: dense + BM25 fused by reciprocal rank, plus exact lookup of any Figure/Table named in the question. Optional passage reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top 20 fused candidates. Low-trust records are never returned. |
| Answer | The same Qwen2.5-VL model returns JSON with the answer, the status and the ids of the evidence blocks it used. Citations and figure references are built from those ids, so every citation points to a retrieved record. Revision, erratum and supersede sentences from the documents are always given to the model. |
| Memory | One `Conversation` object per `conversation_id`. User-supplied values stay in that object; the notebook asserts the ChromaDB count is unchanged after answering. |

No question-specific rules or answers are in the code.

## How to run on Kaggle

1. New notebook in the competition, Accelerator: GPU T4 (or P100), Internet on.
2. Add Input: the competition data. Also upload `eval/ritika_neema/p1_golden_set.json` as a small private
   dataset and add it, so the evaluation cell can find it.
3. Import `notebooks/ritika_neema/p1_rag.ipynb` and Run All. The notebook writes
   `/kaggle/working/submission.csv`, runs `validate_submission.py`, then the golden-set evaluation.
4. Submit from the Output tab. Download the executed notebook and commit it here with its outputs.

Environment switches: `P1_RERANK=1` makes hybrid + reranker the final system (default off; the evaluation
always reports dense, hybrid and hybrid + reranker, and lists the golden questions the reranker fixed or broke),
`P1_VLM_ID` (default `Qwen/Qwen2.5-VL-7B-Instruct`), `P1_LOAD_4BIT=0` for fp16
(use with `Qwen/Qwen2.5-VL-3B-Instruct`), `P1_RUN_EVAL=0` to skip evaluation, `P1_DRY_RUN=1` for a
laptop plumbing test with a stub model (never for reported numbers).

## Human evaluation

After the run, `p1_groundedness_to_rate.csv` holds the 25 final answers. Both members rate groundedness
1-5 in `rater_ritika` and `rater_sneha`, save it as `eval/ritika_neema/p1_groundedness_ratings.csv`, and
re-run the last cell for the mean and agreement.

## Results

### Version 1 (Kaggle Tesla T4, 28 Oct 2026, `p1_rag_v1.ipynb`)

Golden set (25 questions), printed by the evaluation cell. Whole run 25.7 min; the 40 Kaggle questions took
16.9 min including model loading and figure descriptions.

| System | Correct / 25 | Recall@5 | Citation accuracy | Median s / question |
| --- | --- | --- | --- | --- |
| Baseline (dense only) | 12 | 0.913 | 0.660 | 6.90 |
| Hybrid + exact figure lookup (final) | 13 | 1.000 | 0.770 | 6.36 |
| Hybrid + exact figure lookup + reranker | 13 | 0.913 | 0.687 | 7.04 |

Kaggle public leaderboard (about half of the 40 questions): **0.441** (submission `Ritika v1`).

Reranker ablation: it fixed G15 and G25 and broke G03 and G20, so ON ties OFF and is not used. It also lowered
Recall@5 and citation accuracy; the MS MARCO cross-encoder was trained on web search, not economic tables.

Retrieval was not the problem (Recall@5 = 1.000 for hybrid). The 12 misses were in reading the evidence:
six wrong values from tables (G04, G05, G06, G14, G22, G25), because PyMuPDF's page text gives one number per
line with the year headings apart from the values; and five wrong statuses (G07 conflict, G18, G19 and G21
false premise, G24 session override) plus one wrong value (G15).

### Version 2 changes

Made after seeing the version 1 failures, so the golden set is no longer an unbiased test for version 2; the
hidden Kaggle set is. Both changes are general (no question-specific rules or answers):

1. Tables as labelled rows (see Pipeline), with the notes and explaining sentences attached.
2. The answer prompt defines each of the five statuses, with session override taking priority over false
   premise.

Also: the index and model cells can be re-run in one session, and an out-of-memory prompt is retried with fewer
evidence blocks.

Kaggle public leaderboard: **0.495** (Kaggle notebook version 1, which runs this version 2 code), up from 0.441.

Golden set, version 2 (Kaggle Tesla T4, 28 Oct 2026, `p1_rag.ipynb`; whole run 27.4 min, the 40 Kaggle
questions 16.8 min):

| System | Correct / 25 | Recall@5 | Citation accuracy | Median s / question |
| --- | --- | --- | --- | --- |
| Baseline (dense only) | 13 | 0.913 | 0.860 | 7.94 |
| Hybrid + exact figure lookup (final) | 16 | 1.000 | 0.813 | 8.03 |
| Hybrid + exact figure lookup + reranker | 17 | 0.957 | 0.720 | 8.14 |

Hybrid went from 13 to 16 correct: G06, G24 and G25 now pass (table values and session override). Still
failing: G04, G05, G14, G15, G22 (values), G07 (conflict), G18, G19, G21 (false premise). With version 2 the reranker fixed G04, G14 and G15 and broke G20 and G24, one more correct
answer than hybrid, but with lower Recall@5 and citation accuracy. The reranker settings were fixed before either
run and were not changed.
