# Contributions

Pair_Programming_Team_05: Sneha Singh, Ritika Neema

## Meetings

| | Date | Topic | Link |
| --- | --- | --- | --- |
| 1 | | | |
| 2 | | | |

## Sneha Singh

### Part 1

I built and ran `notebooks/p1_rag.ipynb` locally (Ollama `qwen2.5:7b` answers, `qwen2.5vl:3b` figure captions). It parses the three PDFs, quarantines Appendix C, answers from text, figures, and tables, and keeps conversation edits out of the index. Architecture figure: `report/figures/part1/architecture.png`. Notes: `Part1/sneha_singh/results.md`. Graded tag `p1-v1.0` is at commit `1311cf4`.

**Latest notebook print (passage-reranker ablation, not a Kaggle score):**

| Measure | Reranker OFF | Reranker ON |
| --- | --- | --- |
| Answers correct | 20/25 | 19/25 |
| Retrieval Recall@5 | 100% | 100% |
| All required sources in top 5 | 100% | 100% |
| Citation support—model judge | 94.4% | 97.3% |
| Median seconds/question | 55.63 | 50.12 |
| Generation / judge errors | 2 / 0 | 2 / 0 |

Source: notebook cell that loads `Part1/sneha_singh/outputs/evaluation/reranker_ablation/metrics.json`. Retrieval is strong. Prefer **reranker OFF** as the main system for now: the +2 bonus needs the reranker to improve answers, and ON is currently 19/25 against 20/25. No retraining: Part 1 is RAG; remaining work is pipeline bug fixes.

**Bugs to fix next (do not retrain):** G04 (Table B-2 year / arithmetic verify), G15 (`session_override` drops the temporary 6.0% assumption), G20 (revision event month/year). Shared failures on both arms. Reranker ON also loses G02, G05, G08; OFF also fails G12 and G14.

**Still open for both of us:** review `eval/golden_set.json`, fill human groundedness ratings (1–5) for the 25 answers, write Appendix C safety examples in the report, fill the report Part 1 results table from a notebook print, and run the timed Kaggle GPU submission (`links.txt` still TBD). Ritika still needs her own Part 1 pipeline run recorded under `Part1/ritika_neema/`.

### Part 2

I built and ran `notebooks/p2_finetune.ipynb` on a Colab L4: data prep, LoRA rank 4 and rank 16, and the 120 images. I scored KID, CLIP, and LPIPS, ran the safety checker and the memorization check, and uploaded both adapters and the images to Hugging Face. I made the blinded rating page and rated the 25 pairs. Ritika Neema and Rajesh rated the same 25 pairs. Rajesh is the rater from outside the team. My notes are in `Part2/sneha_singh/results.md` and section III of `report/report.md`. How to run it is in `README.md`. The prompt file is `eval/p2_prompts.json`.

### Part 3

I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list before the runs. I built the five agents, the orchestrator, the Gradio app, the rating page, and the results code in `notebooks/p3_agents.ipynb`. Text found in an image is data: the prompt agent does not copy it, and the critic keeps a check only when it is grounded in the request. I smoke-tested on my Mac at 384px and 15 steps, then ran the 15 scenarios and 10 red-team cases on an L4 on 26 Oct 2026 at 512px and 30 steps, seed 266, temperature 0. That run is in the notebook: 30 of 30 routes matched, median 15.1 s without the critique loop and 31.4 s with it. Ritika and I both rated that run and agreed on all 40. The Results cell printed 6/15 scenarios passed without the critique loop and 4/15 with it, critic agreement 34/48, and attack success 0/10. I did not retune the critic after seeing 4/15. The test inputs are at https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs. I deployed the same app as a Hugging Face Space on the free ZeroGPU plan: https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. The Space source is `Part3/sneha_singh/hf_space/`. The architecture figure is `report/figures/part3/architecture.png`. Notes are in `Part3/sneha_singh/results.md` and section IV of `report/report.md`.

## Ritika Neema

### Part 1

To be filled by Ritika (`Part1/ritika_neema/`).

### Part 2

Rated the 25 blinded image pairs. No other Part 2 work is recorded yet (`Part2/ritika_neema/`).

### Part 3

Rated the 26 Oct L4 run. She and I agreed on all 40. Her own pipeline run is not recorded yet (`Part3/ritika_neema/`).
