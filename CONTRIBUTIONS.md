# Contributions

Pair_Programming_Team_05: Sneha Singh, Ritika Neema

## Meetings

| | Date | Topic | Link |
| --- | --- | --- | --- |
| 1 | | | |
| 2 | | | |

## Sneha Singh

### Part 1

I built `notebooks/sneha_singh/p1_rag.ipynb` (ingestion, hybrid dense+BM25 retrieval, optional CE passage reranker, conversation memory, Appendix C quarantine, grounded JSON answers). Local golden-set ablation (27 Oct fix-100, Ollama): **25/25 OFF** and **25/25 ON**, Recall@5 100% (+2 bonus not claimable on the tie). Public Kaggle practice best **0.860**. The graded notebook path uses `Qwen/Qwen2.5-VL-7B-Instruct` NF4 on Kaggle with general lab policies only (no question-text answer bank); Save Version `lab2-notebook` v16+ is in `links.txt`. Notes: `Part1/sneha_singh/results.md`, `source_notes.md`, report §II. Tag `p1-v1.0` will be retagged to the commit that contains the executed Kaggle notebook.

### Part 2

I built and ran `notebooks/sneha_singh/p2_finetune.ipynb` on a Colab L4: data prep, LoRA rank 4 and rank 16, and the 120 images. I scored KID, CLIP, and LPIPS, ran the safety checker and the memorization check, and uploaded both adapters and the images to Hugging Face. I made the blinded rating page and rated the 25 pairs. Ritika Neema and Rajesh rated the same 25 pairs. Rajesh is the rater from outside the team. My notes are in `Part2/sneha_singh/results.md` and section III of `report/report.md`. How to run it is in `README.md`. The prompt file is `eval/p2_prompts.json`.

### Part 3

I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list before the runs. I built the five agents, the orchestrator, the Gradio app, the rating page, and the results code in `notebooks/sneha_singh/p3_agents.ipynb`. Text found in an image is data: the prompt agent does not copy it, and the critic keeps a check only when it is grounded in the request. I smoke-tested on my Mac at 384px and 15 steps, then ran the 15 scenarios and 10 red-team cases on an L4 on 26 Oct 2026 at 512px and 30 steps, seed 266, temperature 0. That run is in the notebook: 30 of 30 routes matched, median 15.1 s without the critique loop and 31.4 s with it. Ritika and I both rated that run and agreed on all 40. The Results cell printed 6/15 scenarios passed without the critique loop and 4/15 with it, critic agreement 34/48, and attack success 0/10. I did not retune the critic after seeing 4/15. The test inputs are at https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs. I deployed the same app as a Hugging Face Space on the free ZeroGPU plan: https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. The Space source is `Part3/sneha_singh/hf_space/`. The architecture figure is `report/figures/part3/architecture.png`. Notes are in `Part3/sneha_singh/results.md` and section IV of `report/report.md`.

## Ritika Neema

### Part 1

I built my own pipeline in `notebooks/ritika_neema/p1_rag.ipynb`: PyMuPDF ingestion with every numbered Figure
and Table cropped as an image with its label and caption, Qwen2.5-VL figure descriptions, bge-small-en-v1.5
embeddings in ChromaDB plus BM25 with reciprocal rank fusion, exact figure lookup, a separate memory per
conversation, and citations built from the evidence the model used. My golden set is
`eval/ritika_neema/p1_golden_set.json` (25 questions, none from `Questions.csv`), committed before tuning.
I lead the Kaggle GPU run and the final Kaggle submission. Results: to be added from the executed notebook.

### Part 2

Rated the 25 blinded image pairs for Sneha's runs. My own pipeline is `notebooks/ritika_neema/p2_finetune.ipynb`:
LoRA rank 8 and DoRA rank 8 on the same data, steps and seed, with KID, CLIP score, LPIPS, NSFW filter and a
memorization check. Final model LoRA rank 8; results in `Part2/ritika_neema/results.md`.

### Part 3

Rated the 26 Oct L4 run with Sneha; we agreed on all 40. My own pipeline is `notebooks/ritika_neema/p3_agents.ipynb`:
five agents over a typed state with a JSON trace per request (`traces/ritika_neema/`), run on the team's frozen
15 scenarios and 10 red-team cases with and without the critique loop, plus a Gradio app. Ran it on a Colab T4
on 28 Oct 2026; automatic results and failure analysis in `Part3/ritika_neema/README.md`, human rating pending.
