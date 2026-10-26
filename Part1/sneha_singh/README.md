# Part 1 - Multimodal RAG (Sneha Singh)

Notebook: [`../../notebooks/p1_rag.ipynb`](../../notebooks/p1_rag.ipynb). All implementation, settings, prompts, tests, evaluation and CSV export are in its cells; there are no helper modules to import. Project overview: [`../../README.md`](../../README.md).

## What I did in Part 1
- Built the multimodal RAG pipeline in one notebook: page-aware ingestion of the three PDFs, figure and table crops, VLM figure descriptions, ChromaDB + BM25 retrieval, grounded JSON answers with separate memory per conversation.
- Compared a dense baseline with a hybrid final system (reranking, figure-label priority, neighbouring context) on a 25-question draft golden set: 19 vs 11 correct, Recall@5 93.8% vs 72.9% (automated development diagnostics, not a holdout). The hybrid has no generation or judge errors; two rejected baseline generations remain counted as incorrect.
- Not done yet: both members' review of the golden set and groundedness ratings, a fresh frozen question set, the Kaggle GPU run with its 90-minute check, and the final submission.

Numbers and discussion: [`results.md`](results.md).

## Layout

- `../data/`: supplied PDFs, `Questions.csv`, `kb_manifest.json` (read-only, not in Git).
- `../../eval/`: golden set (draft) and the two-member rating form.
- `source_notes.md`: my cross-document reading notes.
- `outputs/`: generated crops, index, raw evaluation records (not in Git).
- `../sample_submission.csv`, `../validate_submission.py`: professor's format example and checker.
- `../requirements.txt`: pinned Part 1 environment.

## Pipeline

1. Ingest the three PDFs: page-aware text chunks, separate figure/table crops, table-row records, manifest trust levels (low-trust Appendix C is quarantined and never used as evidence).
2. Describe trusted figures with a vision-language model (cached by image, caption and model digest).
3. Index with MiniLM embeddings in ChromaDB, plus BM25.
4. Retrieve: dense baseline vs hybrid final (dense + BM25 fusion, exact figure-label priority, CPU cross-encoder reranker, neighbouring context).
5. Answer as grounded JSON (`answer`, `status`, `citations`, `figure_references`) with a separate memory per conversation. User assumptions never enter the index.
6. Evaluate on the golden set: answer correctness, Recall@5, citation accuracy, median seconds per question, human groundedness (both members).

## Run locally (Mac)

1. Activate `.venv` (or `pip install -r requirements.txt` from the project root).
2. Start Ollama with `qwen2.5vl:3b` (figure descriptions) and `qwen2.5:7b` (text answers and provisional grading). The embedding model and reranker must be cached once at their pinned revisions. The earlier Llama run is retained in evaluation history, not silently reused for the new model.
3. Open the notebook from the project root or `notebooks/`, select **Lab 2 Part 1 (.venv)** (not the HW4 environment), and Run All.
   - `RUN_FULL_EVALUATION=0` skips the 50-answer evaluation.
   - `GENERATE_LOCAL_SUBMISSION=1` writes `outputs/submission.csv` from `Questions.csv`.

Validate a generated CSV:

```bash
python Part1/validate_submission.py Part1/sneha_singh/outputs/submission.csv Part1/data/Questions.csv  # run from project root
```

## Run on Kaggle (final submission)

The leaderboard takes `submission.csv`, produced by the notebook run at tag `p1-v1.0`.

1. Upload the notebook from the tagged commit. Attach the competition data (PDFs, manifest, questions) and `validate_submission.py`.
2. Enable a GPU and internet.
3. Run All. Kaggle mode loads `Qwen/Qwen2.5-VL-7B-Instruct` through `transformers` (pinned commit, NF4). No Ollama or hosted API is used.
4. The notebook reads `QUESTIONS_PATH` (default `Questions.csv`), writes `/kaggle/working/submission.csv`, runs the validator, and records runtime, model versions and GPU memory in `submission_run.json`. Limit: 90 minutes.
5. Submit the CSV from the Output tab and select it as the final submission.

Local diagnostics (golden set, report) are skipped on Kaggle.

## Reliability checks

The notebook keeps the figure focus through both retrieval and fact filtering. Source-only follow-ups retain the earlier topic but omit hypothetical numbers. Fresh turns and cached-turn replay use the same assumption-reset rules.

Text extraction follows the PDF's internal reading flow to keep main-body paragraphs separate from margin notes. Both remain searchable; visual crops and structured table rows retain their physical layout. The draft answer is checked against its actually cited quotations. If that review reveals a bad plan, a second bounded planning cycle can reselect facts and recalculate; validation is never bypassed. The model-based checks remain provisional.

Plans must pass source-scope, status and operand checks. A failed check triggers a bounded retry with the rejected output and specific feedback; exhausting retries records a failure, not an unchecked answer. Calculator operands must come from selected source facts or active user assumptions. An explicit hypothetical replacement does not need documentary proof. The final answer cannot change the checked status, and explicit reversed percentage comparisons are rejected. These checks do not establish complete semantic correctness.

Explicit single-year table calculations are also bound to the extracted year/vintage column. Symbolic equations stay in the answer, not the numeric calculator. Percent/fraction inputs and Unicode arithmetic are normalized consistently; share calculations show their percentage form. Recall@5 is the mean fraction of required sources found in the top five; any-source hit rate and all-source rate are separate. Earlier history used the hit rate under the recall name and is not directly comparable on that metric.

Implementation references: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [LangChain validation retries](https://github.com/langchain-ai/docs/blob/main/src/oss/langchain/structured-output.mdx), [Lost in the Middle](https://arxiv.org/abs/2307.03172), and [Qdrant's retrieval-versus-generation evaluation](https://qdrant.tech/documentation/search-evaluation/pipeline-output-quality/). These are engineering references, not additions to the document-only economics knowledge base.

## Status

- The Kaggle GPU run and 90-minute limit are not yet verified.
- The golden set is an AI-assisted draft; both members must review it and commit it before further tuning. Scores on it are development diagnostics, not an untouched holdout.
- Human ratings (50 required) are not yet entered.
- Local (Ollama) and Kaggle (Qwen 7B) use different answer models; their scores are not comparable.
- Current metrics are in `outputs/evaluation/metrics.json`; earlier runs are kept in `history/`.
