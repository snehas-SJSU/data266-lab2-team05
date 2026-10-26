# DATA 266 Lab 2

Team 05: Sneha Singh and Ritika Neema. Canvas and Kaggle team: `Pair_Programming_Team_05`. Private repository: [data266-lab2-team05](https://github.com/snehas-SJSU/data266-lab2-team05).

Three parts: multimodal RAG over three economics PDFs, with a Kaggle competition (Part 1); fine-tuning Stable Diffusion 1.5 with LoRA on Pokemon BLIP captions (Part 2); a multimodal agent system (Part 3). Seed 266. One 16 GB GPU. Grading tags, created before the deadline: `p1-v1.0`, `p2-v1.0`, `p3-v1.0`.

No hosted API is used. Every model below runs locally, on Ollama, or on Kaggle or Colab.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13. Open each notebook from the project root, or from `notebooks/`, and use Restart & Run All. The first cell prints the git commit, library versions, GPU, and seed. On Colab that commit prints `n/a` because the notebook is not inside this repository there.

Weights, datasets, generated images, PDFs, and secrets stay out of git (`.gitignore`). Model weights and generated images are on Hugging Face.

## How to run

| Part | Notebook | Where it runs |
| --- | --- | --- |
| 1 Multimodal RAG | `notebooks/p1_rag.ipynb` | Mac with Ollama, or Kaggle GPU for the submission |
| 2 Diffusion fine-tuning | `notebooks/p2_finetune.ipynb` | Colab GPU. A short smoke test can run on CPU or Apple MPS |
| 3 Multimodal agents | `notebooks/p3_agents.ipynb` | Colab GPU for the final run. Mac with Ollama for development |

Ritika runs the same three notebooks for her own baseline. Her notes will go in `Part1/ritika_neema/`, `Part2/ritika_neema/`, and `Part3/ritika_neema/`.

### Part 1

Local: start Ollama with `qwen2.5vl:3b` and `qwen2.5:7b`, then Run All. Step-by-step notes are in [Part1/sneha_singh/README.md](Part1/sneha_singh/README.md).

Kaggle, for the graded submission: use the notebook from the `p1-v1.0` tag, attach the competition data and `Part1/validate_submission.py`, enable a GPU and internet, and Run All. The notebook reads `QUESTIONS_PATH` (default `Questions.csv`), writes `submission.csv`, and checks it with `validate_submission.py`. Submit that CSV from the Output tab. The lab limit is 90 minutes on one 16 GB GPU. That timed Kaggle run is still to be done. Mac timings do not count for it.

### Part 2

Colab: GPU runtime (T4, L4, or A100), upload `notebooks/p2_finetune.ipynb`, and Run All. It installs the pinned packages, mounts Drive, and saves to `MyDrive/data266_lab2_p2/`. A finished step is skipped, and training resumes from the checkpoint after a disconnect. Each LoRA run took about 15 minutes on an L4. The prompts it uses are already in `eval/p2_prompts.json`. Numbers are in [Part2/sneha_singh/results.md](Part2/sneha_singh/results.md).

Smoke test, from the project root, after the dataset is cached in `Part2/data/`:

```bash
.venv/bin/python Part2/sneha_singh/smoke_test/smoke_test.py
```

That run uses 32 images and 5 LoRA steps. It checks data preparation, training, generation, and the metrics. It is not the graded run.

Demo, base model beside the fine-tuned model:

```bash
.venv/bin/python Part2/sneha_singh/demo_generate.py "prompt one" "prompt two"
```

Weights and images: [rank 16, final](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r16), [rank 4](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r4), [generated images](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p2-images). The same links are in `links.txt`.

### Part 3

Colab: GPU runtime. Upload `eval/` and an `inputs/` folder of the 25 test images to `MyDrive/data266_lab2_p3/`, upload `notebooks/p3_agents.ipynb`, and Run All. Finished cases are skipped on a later run. Notes are in [Part3/sneha_singh/README.md](Part3/sneha_singh/README.md).

Local development: Ollama with `qwen2.5vl:3b`, inputs in `Part3/data/inputs/` (not committed). The Mac path uses 384px and 15 steps. The Colab run uses 512px and 30 steps.

The last notebook cell starts the Gradio demo. On Colab it prints a public link.

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
├── README.md                 how to run each part
├── requirements.txt          pinned versions
├── notebooks/
│   ├── p1_rag.ipynb          RAG and the Kaggle run
│   ├── p2_finetune.ipynb     train and generate
│   └── p3_agents.ipynb       agents and Gradio
├── eval/                     golden set, prompts, scenarios, result sheets
├── traces/                   Part 3 JSON traces
├── Part1/ Part2/ Part3/      data notes and each member's notes
├── report/                   report draft and figures
├── CONTRIBUTIONS.md          who did what, and the meeting recordings
├── AI_USAGE.md               AI tools, by member and part
├── links.txt                 repo, tags, Hugging Face, Kaggle (for the Canvas zip)
└── .gitignore                data, secrets, weights
```

There is no `src/` folder. The code is in the notebooks. `Part2/sneha_singh/outputs/` holds generated images and adapters and is not committed. The lab PDFs in `Lab-requirement/` stay on this machine.

## Where the numbers are

These are the current local results. The report copies them from the notebook outputs. Human ratings and the Kaggle run are still open, and Ritika's results are not in yet.

| Part | System | Result |
| --- | --- | --- |
| 1 | hybrid, against a dense baseline | 23/25 correct (baseline 13/25), Recall@5 100% (baseline 72.9%), on the development golden set. Human ratings 0/50. Kaggle GPU run not done. See `Part1/sneha_singh/results.md`. |
| 2 | LoRA rank 16, against the base model | KID 0.0231 (base 0.0602), CLIP 0.3155 (base 0.322), human win rate 38.7% (95% CI 28.5–50.0%, 75 votes), 0.24 GPU-hours, 4.27 GB peak. See `Part2/sneha_singh/results.md`. |
| 3 | five agents, with and without the critique loop | Median 22.6 s with the loop and 15.1 s without. 30/30 scenario runs routed. Pass rates and ratings are not filled in yet. See `Part3/sneha_singh/results.md`. |

## Status

`eval/p2_prompts.json` is the only file on GitHub so far. The rest of this folder is still local. The three tags are not created yet. Part 2 training, the Hugging Face uploads, and the demo script are done. Part 1 still needs both members' review of the golden set, the human ratings, and the timed Kaggle run. Part 3 has a Colab run in the notebook. The two rating files are not both in place, the agreed finals are blank, and the test inputs are not on Hugging Face yet.
