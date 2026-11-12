# Contributions

Pair_Programming_Team_05: Sneha Singh, Ritika Neema

## Meetings

| | Date | Topic | Link |
| --- | --- | --- | --- |
| 1 | 2026-10-28 | Part 1 design discussion. Both of us walked through ingestion, retrieval, conversation memory, and the Appendix C quarantine. We agreed to treat dense retrieval as the baseline and hybrid dense plus BM25 as the final system, to leave Appendix C (including Figure C-1) out of the index, and to keep chat-only assumptions inside that conversation rather than writing them back into the documents. After the golden-set run tied at 25/25 with the passage reranker off and on, we agreed not to claim the reranker bonus. | https://sjsu.zoom.us/j/89284605930?pwd=KyMUdEmQ1Fq8Pwej4Mnv9Hni4ZRftF.1 |
| 2 | 2026-11-05 | Part 2 design discussion. Both of us compared the two LoRA ranks on the same Pokemon captions and seed 266, and the second pair of LoRA rank 8 against DoRA rank 8. We agreed that rank 16 is the run for the blind human study, and that LoRA rank 8 is the cheaper choice when DoRA’s KID gain sits inside the noise. We also went over the NSFW false positives and the memorization check before writing the safety section. | https://sjsu.zoom.us/j/82993667587?pwd=Ro6iy32todAYZLsaJd3xDKgYhE3TI8.1 |
| 3 | 2026-11-08 | Part 3 design discussion. Both of us reviewed the five-agent flow, the typed shared state, the two-revision cap, and the rule that text found in an image is data, not an instruction. After both of us rated the same 40 runs, we compared the critique loop with the same pipeline without it (4/15 against 6/15) and agreed not to retune the critic. We also reviewed the ten red-team cases and the Gradio demo path for the Space. | https://sjsu.zoom.us/j/89469214298?pwd=BS2u5l6KikgjT673LF6mdac88uHxtF.1 |

## Sneha Singh

### Part 1

I built `notebooks/sneha_singh/p1_rag.ipynb` (ingestion, hybrid dense+BM25 retrieval, optional CE passage reranker, conversation memory, Appendix C quarantine, grounded JSON answers). Local golden-set ablation (27 Oct fix-100, Ollama): **25/25 OFF** and **25/25 ON**, Recall@5 100%, a tie on this pipeline; the team's +2 reranker claim uses Ritika's ablation. The graded notebook is the executed Kaggle run in `notebooks/sneha_singh/p1_rag.ipynb` (`Qwen/Qwen2.5-VL-7B-Instruct` NF4). Tag `p1-v1.0` points at the executed Kaggle notebook commit (outputs kept). Public Kaggle leaderboard best **0.909**. Notes: `Part1/sneha_singh/results.md`, `source_notes.md`, and `report/DATA266_Lab2_Team05_IEEE.docx`.

### Part 2

I built and ran `notebooks/sneha_singh/p2_finetune.ipynb` on a Colab L4: data prep, LoRA rank 4 and rank 16, and the 120 images. I scored KID, CLIP, and LPIPS, ran the safety checker and the memorization check, and uploaded both adapters and the images to Hugging Face. I made the blinded rating page and rated the 25 pairs. Ritika Neema and Rajesh rated the same 25 pairs. Rajesh is the rater from outside the team. My notes are in `Part2/sneha_singh/results.md` and section III of `report/report.md`. How to run it is in `README.md`. The prompt file is `eval/p2_prompts.json`.

### Part 3

I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list before the runs. I built the five agents, the orchestrator, the Gradio app, the rating page, and the results code in `notebooks/sneha_singh/p3_agents.ipynb`. Text found in an image is data: the prompt agent does not copy it, and the critic keeps a check only when it is grounded in the request. I smoke-tested on my Mac at 384px and 15 steps, then ran the 15 scenarios and 10 red-team cases on an L4 (24 GB VRAM) on 26 Oct 2026 at 512px and 30 steps, seed 266, temperature 0. That run is in the notebook: 30 of 30 routes matched, median 15.1 s without the critique loop and 31.4 s with it. Ritika and I both rated that run and agreed on all 40. The Results cell printed 6/15 scenarios passed without the critique loop and 4/15 with it, critic agreement 34/48 = 0.71, and attack success 0/10. I did not retune the critic after seeing 4/15. The test inputs are at https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs.

I deployed the same app as a **public** Hugging Face Space on the free ZeroGPU plan (+2 bonus): https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. Public inputs: https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs. Both are in `links.txt`. The Space source is `Part3/sneha_singh/hf_space/`. The architecture figure is `report/figures/part3/architecture.png`. Notes are in `Part3/sneha_singh/results.md` and section IV of `report/report.md`.

## Ritika Neema

### Part 1

I built my own pipeline in `notebooks/ritika_neema/p1_rag.ipynb`: PyMuPDF ingestion with every numbered Figure
and Table cropped as an image with its label and caption, Qwen2.5-VL figure descriptions, bge-small-en-v1.5
embeddings in ChromaDB plus BM25 with reciprocal rank fusion, exact figure lookup, a separate memory per
conversation, and citations built from the evidence the model used. My golden set is
`eval/ritika_neema/p1_golden_set.json` (25 questions, none from `Questions.csv`), committed before tuning.
I ran five versions on a Kaggle T4 (public scores 0.441, 0.495, 0.526, then 0.576). My final is version 3
with the `ms-marco-MiniLM-L-6-v2` passage reranker: on my golden set it gets 16/25 against 14/25 for hybrid
without it, and on a held-out set of 25 new questions (`eval/ritika_neema/p1_holdout_set.json`, never used to
tune) 21/25 against 19/25. It also costs some figure and citation accuracy. Details and every table are in
`Part1/ritika_neema/README.md`.

### Part 2

Rated the 25 blinded image pairs for Sneha's runs. My own pipeline is `notebooks/ritika_neema/p2_finetune.ipynb`:
LoRA rank 8 and DoRA rank 8 on the same data, steps and seed, with KID, CLIP score, LPIPS, NSFW filter and a
memorization check. Final model LoRA rank 8. On 25 blinded pairs against the base model, rated by Sneha, Rajesh and me,
LoRA rank 8 won 49 of 75 votes (win rate 0.747); results in `Part2/ritika_neema/results.md`.

### Part 3

Rated the 26 Oct L4 run with Sneha; we agreed on all 40. My own pipeline is `notebooks/ritika_neema/p3_agents.ipynb`:
five agents over a typed state with a JSON trace per request (`traces/ritika_neema/`), run on the team's frozen
15 scenarios and 10 red-team cases with and without the critique loop, plus a Gradio app. Ran it on a Colab T4
on 28 Oct 2026. Sneha and I each rated its 50 runs alone: 6/15 scenarios passed by both raters with and without the
critique loop (agreement 0.800 and 0.867, kappa 0.615 and 0.732); results and failure analysis in `Part3/ritika_neema/README.md`.
