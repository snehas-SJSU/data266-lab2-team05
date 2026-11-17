# Part 1 - Multimodal RAG (Sneha Singh)

## Kaggle public leaderboard (practice)

| Item | Value |
| --- | --- |
| **Selected public score** | **0.927** |
| Team | `Pair_Programming_Team_05` |
| Selected CSV | `Part1/sneha_singh/submission.csv` (sha `65d699b5…`) |
| Notebook | `notebooks/sneha_singh/p1_rag.ipynb` |
| Final graded path | Repository notebook at git tag `p1-v1.0` (hidden set) |

### Public Kaggle score

Selected practice leaderboard score is **0.927** from `Part1/sneha_singh/submission.csv` (export from the executed Kaggle run’s `p1_outputs/submission.csv`). Practice LB is feedback only. Task 1.5 uses the tagged notebook on the hidden set.

### Selected system (current)

| Piece | Choice |
| --- | --- |
| Embeddings | `BAAI/bge-small-en-v1.5` (Chroma dense) + BM25, RRF hybrid |
| Reranker | `BAAI/bge-reranker-base` available; **OFF in final flags** (LB test: ON lowered score) |
| Answer / figures | `Qwen/Qwen2.5-VL-7B-Instruct` (NF4 on Kaggle T4) |
| Routing | Exact figure/table label lookup when named; trust filter; precedence notes |
| Package | Notebook writes `mmrag/` in-session (no external private HF repo) |

This replaces the earlier MiniLM + long `ConversationRAG` notebook as the **selected** practice / graded notebook path.

### Earlier practice note (not selected)

An older MiniLM / salvage pipeline produced a higher practice CSV (**0.940**, sha `881a9ffb…`) that is **not** the selected graded path. A clean re-run of that family without retrieval hardcodes scored **0.533** and was discarded.

## Team +2 bonus

The team claims Part 1 **+2** from Ritika’s reranker ablation (`Part1/ritika_neema/README.md`, public score 0.576 on her notebook). Sneha’s selected **0.927** path keeps rerank **off**.

## Local / historical ablation

Older local golden-set OFF/ON prints and repair notes for the previous notebook live under `Part1/sneha_singh/eval/` and prior git history. They are development history for the previous stack, not the selected **0.927** Kaggle CSV.
