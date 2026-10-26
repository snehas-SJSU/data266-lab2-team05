# DATA 266 Lab 2

Team 05: Sneha Singh and Ritika Neema. Canvas and Kaggle team: `Pair_Programming_Team_05`. Private repository: [data266-lab2-team05](https://github.com/snehas-SJSU/data266-lab2-team05).

Three parts: multimodal RAG over three economics PDFs, with a Kaggle competition (Part 1); fine-tuning Stable Diffusion 1.5 with LoRA on Pokemon BLIP captions (Part 2); a multimodal agent system (Part 3). Seed 266. One 16 GB GPU. Grading tags, created before the deadline: `p1-v1.0`, `p2-v1.0`, `p3-v1.0`.

No hosted API is used. Every model below runs locally, on Ollama, or on Kaggle or Colab. Both members use the same three notebooks. Each member's notes, runs, and links stay in her own section.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13. Open each notebook from the project root, or from `notebooks/`, and use Restart & Run All. The first cell prints the git commit, library versions, GPU, and seed. On Colab that commit prints `n/a` because the notebook is not inside this repository there.

Weights, datasets, generated images, PDFs, and secrets stay out of git (`.gitignore`). Model weights and generated images are on Hugging Face.

| Part | Notebook | Where it runs |
| --- | --- | --- |
| 1 Multimodal RAG | `notebooks/p1_rag.ipynb` | Mac with Ollama, or Kaggle GPU for the submission |
| 2 Diffusion fine-tuning | `notebooks/p2_finetune.ipynb` | Colab GPU. A short smoke test can run on CPU or Apple MPS |
| 3 Multimodal agents | `notebooks/p3_agents.ipynb` | Colab GPU for the final run. Mac with Ollama for development |

## Part 1

Shared notebook: `notebooks/p1_rag.ipynb`. Kaggle, for the graded submission: use the notebook from the `p1-v1.0` tag, attach the competition data and `Part1/validate_submission.py`, enable a GPU and internet, and Run All. The notebook reads `QUESTIONS_PATH` (default `Questions.csv`), writes `submission.csv`, and checks it with `validate_submission.py`. Submit that CSV from the Output tab. The lab limit is 90 minutes on one 16 GB GPU. Mac timings do not count for it.

### Sneha Singh

Local: start Ollama with `qwen2.5vl:3b` and `qwen2.5:7b`, then Run All. Step-by-step notes are in [Part1/sneha_singh/README.md](Part1/sneha_singh/README.md). Scores are in [Part1/sneha_singh/results.md](Part1/sneha_singh/results.md).

On the development golden set, the hybrid system is 23/25 correct and Recall@5 is 100%. The dense baseline is 13/25 and Recall@5 is 72.9%. Human ratings are 0/50. The timed Kaggle run is not done. The golden set still needs both members to review it.

### Ritika Neema

Her Part 1 notes are not in yet. They go in [Part1/ritika_neema/README.md](Part1/ritika_neema/README.md). She still needs to run this notebook herself and record that baseline. Her Kaggle link in `links.txt` is still open.

## Part 2

Shared notebook: `notebooks/p2_finetune.ipynb`. Colab: GPU runtime (T4, L4, or A100), upload the notebook, and Run All. It installs the pinned packages, mounts Drive, and saves to `MyDrive/data266_lab2_p2/`. A finished step is skipped, and training resumes from the checkpoint after a disconnect. The prompts are in `eval/p2_prompts.json`.

Smoke test, from the project root, after the dataset is cached in `Part2/data/`:

```bash
.venv/bin/python Part2/sneha_singh/smoke_test/smoke_test.py
```

That run uses 32 images and 5 LoRA steps. It checks data preparation, training, generation, and the metrics. It is not the graded run.

### Sneha Singh

The graded run is on an NVIDIA L4. Each LoRA run took about 15 minutes. Rank 16 is the model she kept: KID 0.0231 against the base model's 0.0602, CLIP 0.3155 against 0.322, 0.24 GPU-hours, 4.27 GB peak. The human win rate on her images is 38.7% (95% CI 28.5–50.0%, 75 votes from three raters). Numbers are in [Part2/sneha_singh/results.md](Part2/sneha_singh/results.md).

Demo, base model beside the fine-tuned model:

```bash
.venv/bin/python Part2/sneha_singh/demo_generate.py "prompt one" "prompt two"
```

Weights and images: [rank 16, final](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r16), [rank 4](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r4), [generated images](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p2-images). The same links are in `links.txt`.

### Ritika Neema

She rated the 25 blinded pairs. That vote is in `Part2/sneha_singh/human_eval/ratings/ratings_Ritika_Neema.csv`. Her own training run is not recorded yet. Notes for that run go in [Part2/ritika_neema/README.md](Part2/ritika_neema/README.md). Her model and image links in `links.txt` are still open.

## Part 3

Shared notebook: `notebooks/p3_agents.ipynb`. Colab: GPU runtime. Put `eval/` and an `inputs/` folder of the test images in `MyDrive/data266_lab2_p3/`, upload the notebook, and Run All. Finished cases are skipped on a later run, so delete `outputs/runs.jsonl` before a new graded run. The last cell starts the Gradio demo and, on Colab, prints a public link.

Local development: Ollama with `qwen2.5vl:3b`, inputs in `Part3/data/inputs/` (not committed). The Mac path uses 384px and 15 steps. The Colab run uses 512px and 30 steps.

### Sneha Singh

Notes are in [Part3/sneha_singh/README.md](Part3/sneha_singh/README.md). The 26 Oct L4 run routed 30 of 30 scenario runs. Median time was 31.4 s with the critique loop and 15.1 s without it. Both of us rated that run and agreed on all 40. The Results cell printed 6/15 passed without the loop and 4/15 with it, critic agreement 34/48, and red-team attack success 0/10. The test images are not on Hugging Face yet.

### Ritika Neema

Her Part 3 notes are not in yet. They go in [Part3/ritika_neema/README.md](Part3/ritika_neema/README.md). She still needs to run this notebook herself. Her rating CSV and her Hugging Face input link are still open.

## Models

| Part | Models | Hosted API |
| --- | --- | --- |
| 1, local | Ollama `qwen2.5vl:3b` for figure descriptions, `qwen2.5:7b` for answers; `all-MiniLM-L6-v2`; `ms-marco-MiniLM-L6-v2` reranker | none |
| 1, Kaggle | `Qwen/Qwen2.5-VL-7B-Instruct` through `transformers` | none |
| 2 | `stable-diffusion-v1-5/stable-diffusion-v1-5` with LoRA ranks 4 and 16; `openai/clip-vit-base-patch32`; LPIPS AlexNet; torchmetrics KID | none |
| 3 | `Qwen2.5-VL-3B` for language and vision; Stable Diffusion 1.5 img2img; InstructPix2Pix; the Part 2 rank-16 LoRA; CLIP ViT-B/32; the Stable Diffusion safety checker | none |

## Folder structure

```text
data266-lab2-team05/
├── README.md                 how to run each part, split by member
├── requirements.txt          pinned versions
├── notebooks/
│   ├── p1_rag.ipynb          RAG and the Kaggle run
│   ├── p2_finetune.ipynb     train and generate
│   └── p3_agents.ipynb       agents and Gradio
├── eval/                     golden set, prompts, scenarios, result sheets
├── traces/                   Part 3 JSON traces
├── Part1/ Part2/ Part3/      each member's notes (sneha_singh/ and ritika_neema/)
├── report/                   report draft and figures
├── CONTRIBUTIONS.md          who did what, by member and part
├── AI_USAGE.md               AI tools, by member and part
├── links.txt                 repo, tags, Hugging Face, Kaggle, by member and part
└── .gitignore                data, secrets, weights
```

There is no `src/` folder. The code is in the notebooks. `Part2/sneha_singh/outputs/` holds generated images and adapters and is not committed. The lab PDFs in `Lab-requirement/` stay on this machine.

## Status

The repository is on GitHub at commit `07f0705`. The three tags are not created yet. Who has done what:

| Part | Sneha Singh | Ritika Neema |
| --- | --- | --- |
| 1 | Local notebook run is in. Golden-set review, human ratings, and the timed Kaggle run are open. | Her run is not recorded yet. |
| 2 | Training, Hugging Face uploads, and the demo script are done. She rated the 25 pairs. | She rated the 25 pairs. Her own training run is not recorded yet. |
| 3 | The 26 Oct L4 run and both ratings are in. Results cell: 6/15 without the critique loop, 4/15 with it. | She rated this run. Her own pipeline run is not in yet. |
