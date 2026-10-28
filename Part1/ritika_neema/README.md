# Ritika Neema - Part 1

My own Part 1 pipeline is `notebooks/ritika_neema/p1_rag.ipynb`. The golden set is
`eval/ritika_neema/p1_golden_set.json`, written before any tuning; none of its 25 questions come from
`Questions.csv`.

## Pipeline

| Step | What it does |
| --- | --- |
| Ingestion | PyMuPDF. Text chunks of about 120 words keep `doc_id`, `page` and `trust_level`. Each numbered Figure and Table is found from its caption (label set in a heading font) and saved as a cropped PNG with its label and caption. Raster charts (DOC2, DOC3) are cropped to the image under the caption; vector charts (DOC1) to the drawings below the caption. Table text is read from the table area, cut at the `Source` line. |
| Trust | From `kb_manifest.json`, including the page override that makes DOC3 Appendix C (p7) low trust. |
| Figures | `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit) writes a data description of each figure. Low-trust figures are not described. |
| Index | `BAAI/bge-small-en-v1.5` embeddings in ChromaDB (metadata `doc_id`, `page`, `figure_id`, `trust_level`) and BM25 over the same records. |
| Retrieval | Baseline: dense only. Final: dense + BM25 fused by reciprocal rank, plus exact lookup of any Figure/Table named in the question. Low-trust records are never returned. |
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

Environment switches: `P1_VLM_ID` (default `Qwen/Qwen2.5-VL-7B-Instruct`), `P1_LOAD_4BIT=0` for fp16
(use with `Qwen/Qwen2.5-VL-3B-Instruct`), `P1_RUN_EVAL=0` to skip evaluation, `P1_DRY_RUN=1` for a
laptop plumbing test with a stub model (never for reported numbers).

## Human evaluation

After the run, `p1_groundedness_to_rate.csv` holds the 25 final answers. Both members rate groundedness
1-5 in `rater_ritika` and `rater_sneha`, save it as `eval/ritika_neema/p1_groundedness_ratings.csv`, and
re-run the last cell for the mean and agreement.
