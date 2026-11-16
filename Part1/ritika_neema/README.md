# Ritika Neema - Part 1

My own Part 1 pipeline is `notebooks/ritika_neema/p1_rag.ipynb` (version 3 code with the passage reranker as
the final system; Kaggle public score **0.576**). The golden set is `eval/ritika_neema/p1_golden_set.json`,
written before any tuning; none of its 25 questions come from `Questions.csv`. The held-out set is
`eval/ritika_neema/p1_holdout_set.json`, 25 new questions written after versions 1-4 and never used to tune.

## Pipeline

| Step | What it does |
| --- | --- |
| Ingestion | PyMuPDF. Text chunks of about 120 words keep `doc_id`, `page` and `trust_level`. Each numbered Figure and Table is found from its caption (label set in a heading font) and saved as a cropped PNG with its label and caption. Raster charts (DOC2, DOC3) are cropped to the image under the caption; vector charts (DOC1) to the drawings below the caption. Tables are rebuilt from PyMuPDF's table detection as one labelled line per row (`row: column = value`), followed by the notes under the table and the page sentences that explain it (v2). |
| Trust | From `kb_manifest.json`, including the page override that makes DOC3 Appendix C (p7) low trust. |
| Figures | `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit) writes a data description of each figure. Low-trust figures are not described. |
| Index | `BAAI/bge-small-en-v1.5` embeddings in ChromaDB (metadata `doc_id`, `page`, `figure_id`, `trust_level`) and BM25 over the same records. |
| Retrieval | Baseline: dense only. Hybrid: dense + BM25 fused by reciprocal rank, plus exact lookup of any Figure/Table named in the question. Final: hybrid + passage reranker, `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scoring the top 20 fused candidates. Low-trust records are never returned. |
| Answer | The same Qwen2.5-VL model returns JSON with the answer, the status and the ids of the evidence blocks it used. Citations and figure references are built from those ids, so every citation points to a retrieved record. Revision, erratum and supersede sentences from the documents are always given to the model. |
| Memory | One `Conversation` object per `conversation_id`. User-supplied values stay in that object; the notebook asserts the ChromaDB count is unchanged after answering. |

No question-specific rules or answers are in the code.

## How to run on Kaggle

1. New notebook in the competition, Accelerator: GPU T4 (or P100), Internet on.
2. Add Input: the competition data. Also upload `eval/ritika_neema/p1_golden_set.json` and
   `eval/ritika_neema/p1_holdout_set.json` as a small private dataset and add it, so the evaluation cell can
   find them.
3. Import `notebooks/ritika_neema/p1_rag.ipynb` and Run All. The notebook writes
   `/kaggle/working/submission.csv`, runs `validate_submission.py`, then the golden-set evaluation.
4. Submit from the Output tab. Download the executed notebook and commit it here with its outputs.

Environment switches: `P1_RERANK=0` makes plain hybrid the final system (default on, hybrid + reranker; the
evaluation always reports dense, hybrid and hybrid + reranker, and lists the golden questions the reranker fixed
or broke),
`P1_VLM_ID` (default `Qwen/Qwen2.5-VL-7B-Instruct`), `P1_LOAD_4BIT=0` for fp16
(use with `Qwen/Qwen2.5-VL-3B-Instruct`), `P1_RUN_EVAL=0` to skip evaluation, `P1_DRY_RUN=1` for a
laptop plumbing test with a stub model (never for reported numbers).

## Human evaluation

Team Part 1 groundedness (both members, 1–5) is filled in `eval/human_review.csv`
(mean 4.88 each; within one point on 24/25). That sheet is what the report uses.
If a notebook run also writes `p1_groundedness_to_rate.csv`, optional per-pipeline
columns can still be saved under `eval/ritika_neema/`, but they are not required
for the submitted human score.

## Results

The executed notebooks for versions 1-4 (`p1_rag_v1.ipynb` to `p1_rag_v4.ipynb`) were removed from `main` in
commit `5aeb91f`. They are kept in git history at commit `91930bc`, e.g.
`git show 91930bc:notebooks/ritika_neema/p1_rag_v3.ipynb > p1_rag_v3.ipynb`.

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

Golden set, version 2 (Kaggle Tesla T4, 28 Oct 2026, `p1_rag_v2.ipynb`; whole run 27.4 min, the 40 Kaggle
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

### Version 3 changes

Also general, and made after seeing the version 2 answers:

1. The answer must state the exact evidence values with units and years, also for why/how questions.
2. Number grounding: each value in an answer is checked against the evidence. A value found nowhere in the
   evidence, and not the difference or sum of two evidence values, is unsupported and the answer becomes
   `not_found`. When the model cites no table or figure, the table or figure that holds the exact value is cited.
3. A `not_found` answer gives no number.
4. When tables from two or more documents are in the evidence, the prompt asks the model to check for a conflict.
5. The evaluation adds an approximate Kaggle-style score (0.4 facts, 0.2 status, 0.2 figure F1, 0.2 citations)
   per question and per component.

Golden set, version 3 (Kaggle Tesla T4, 28 Oct 2026, `p1_rag_v3.ipynb`; whole run 29.4 min):

| System | Correct / 25 | Recall@5 | Citation accuracy | Median s / question | Kaggle-style score |
| --- | --- | --- | --- | --- | --- |
| Baseline (dense only) | 13 | 0.913 | 0.837 | 8.50 | 0.684 |
| Hybrid + exact figure lookup (final) | 14 | 1.000 | 0.880 | 8.69 | 0.750 |
| Hybrid + exact figure lookup + reranker | 16 | 0.957 | 0.768 | 8.62 | 0.739 |

Kaggle-style components for hybrid: facts 0.667, status 0.800, figures 0.733, citations 0.883. Against
version 2, hybrid lost G20 (answered not_found) and G24 (session override answered as answered) and fixed none,
so correct answers fell from 16 to 14 while citations improved. With 25 questions a change of two is within
what a prompt change alone can move. Kaggle public score: **0.526** (Kaggle notebook version 2, which runs this
version 3 code), up from 0.495, so version 3 is the current final. A first attempt to submit it failed with
"Submission contains null values" because an evaluation CSV was submitted instead of `submission.csv`; the
evaluation files now go to an `eval/` subfolder and the notebook checks the submission for cells Kaggle would
read as null.

| Version | Main change | Golden correct (hybrid) | Kaggle public |
| --- | --- | --- | --- |
| v1 | hybrid retrieval, raw table text | 13 / 25 | 0.441 |
| v2 | tables as labelled rows, status definitions | 16 / 25 | 0.495 |
| v3 | exact values, number grounding, citation fill, not_found cleanup, conflict check | 14 / 25 | 0.526 |

### Version 4 changes (after Sneha's notebook)

General checks on the model's work, taken from the approach in `notebooks/sneha_singh/p1_rag.ipynb`:

1. Calculator: the model writes any arithmetic as an expression; the code evaluates it only if every operand is
   a value in the evidence or supplied by the user, and appends the checked result to the answer.
2. Conflict witness: table cells are read as (entity, quantity, year) facts. When two documents disagree on the
   asked entity, quantity and year, and the differing value is not the other document's bracketed forecast,
   the prompt states the conflict instead of a generic reminder.
3. Citations for text the answer quotes (eight consecutive words in common with a retrieved chunk).

Golden set, version 4 (Kaggle Tesla T4, 28 Oct 2026, `p1_rag_v4.ipynb`; whole run 29.7 min):

| System | Correct / 25 | Recall@5 | Citation accuracy | Median s / question | Kaggle-style score |
| --- | --- | --- | --- | --- | --- |
| Baseline (dense only) | 13 | 0.913 | 0.808 | 9.12 | 0.695 |
| Hybrid + exact figure lookup (final) | 15 | 1.000 | 0.773 | 9.05 | 0.747 |
| Hybrid + exact figure lookup + reranker | 17 | 0.957 | 0.771 | 9.24 | 0.758 |

Against version 3 (hybrid): correct 14 to 15 and status 0.800 to 0.880 (the calculator fixed G15; G18 is now
false premise and G24 session override), but citations fell from 0.883 to 0.800 and figures from 0.733 to 0.720,
so the Kaggle-style score is unchanged (0.750 and 0.747). The extra quoted-text citations add pages that are not
in the gold evidence. The conflict witness did not fire on G07 (its answer cites only Document 3's Table B-2; the
witness needs both documents' tables in the evidence). Not submitted to Kaggle (daily limit).

### Final: version 3 + reranker, checked on the held-out set

The reranker beat plain hybrid on the golden set in versions 2, 3 and 4 (17 vs 16, 16 vs 14, 17 vs 15
correct). Because the golden set had been seen while making versions 2-4, I wrote 25 new questions
(`p1_holdout_set.json`, committed before any run on it) and ran version 3 with the reranker as the final system,
fixed before the run (`P1_RERANK=1`). No code other than that default changed.

Golden set (Kaggle Tesla T4, 30 Oct 2026, `p1_rag.ipynb`; whole run 36.3 min, the 40 Kaggle questions 18.0 min):

| System | Correct / 25 | Recall@5 | Citation accuracy | Median s / question | Kaggle-style score |
| --- | --- | --- | --- | --- | --- |
| Baseline (dense only) | 13 | 0.913 | 0.837 | 8.31 | 0.684 |
| Hybrid + exact figure lookup | 14 | 1.000 | 0.880 | 8.48 | 0.750 |
| Hybrid + exact figure lookup + reranker (final) | 16 | 0.957 | 0.768 | 8.48 | 0.739 |

Reranker ON vs OFF: fixed G04, G14, G15; broke G23. Kaggle-style components, hybrid → reranker: facts 0.667 →
0.707, status 0.800 → 0.840, figures 0.733 → 0.640, citations 0.883 → 0.800.

Held-out set (same run, never used for tuning):

| System | Correct / 25 | Recall@5 | Kaggle-style score | Facts | Status | Figures | Citations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid + exact figure lookup | 19 | 0.826 | 0.834 | 0.88 | 0.84 | 0.747 | 0.822 |
| Hybrid + exact figure lookup + reranker (final) | 21 | 0.957 | 0.875 | 0.96 | 0.88 | 0.720 | 0.854 |

Kaggle public score: **0.576**, up from 0.526 for version 3 without the reranker.

The reranker gives more correct answers on both sets (+2 on each) and a higher leaderboard score. It is not a
win on every metric: on the golden set Recall@5, figure F1 and citations drop, so the Kaggle-style score falls
slightly (0.750 to 0.739); on the held-out set only figure F1 drops. A likely cause is that the cross-encoder,
trained on web search, ranks a passage that holds the value above the figure or table record, so fewer figure
and table records reach the evidence the model cites. Held-out misses: H16 (conflict), H19 (false premise
answered as conflict), H20 and H21 (false premise answered as answered); H08, H14 and H15 are correct but score
below 1.0 on figures or citations.

| Version | Main change | Golden correct (final system) | Kaggle public |
| --- | --- | --- | --- |
| v1 | hybrid retrieval, raw table text | 13 / 25 | 0.441 |
| v2 | tables as labelled rows, status definitions | 16 / 25 | 0.495 |
| v3 | exact values, number grounding, citation fill, not_found cleanup, conflict check | 14 / 25 | 0.526 |
| v4 | calculator, conflict witness, quoted-text citations | 15 / 25 | not submitted |
| **v3 + reranker (final)** | cross-encoder reranker over the top 20 | **16 / 25** (held-out 21 / 25) | **0.576** |
