# DATA 266 Lab 2 — Team 05

**GitHub:** [data266-lab2-team05](https://github.com/snehas-SJSU/data266-lab2-team05)
**Canvas / Kaggle:** `Pair_Programming_Team_05`
**Seed:** 266
**GPU:** one 16 GB GPU
**Tags:** `p1-v1.0`, `p2-v1.0`, `p3-v1.0`, all at commit `0824415`. Later commits on `main` are documentation after those tags.

| Member | Folder |
| --- | --- |
| Sneha Singh | `sneha_singh/` under `Part1/`, `Part2/`, and `Part3/` |
| Ritika Neema | `ritika_neema/` under `Part1/`, `Part2/`, and `Part3/` |

Three parts, one shared notebook each: multimodal RAG over three economics PDFs, plus a Kaggle competition (Part 1); Stable Diffusion 1.5 fine-tuned with LoRA on Pokemon BLIP captions (Part 2); a multimodal agent system (Part 3). No hosted API is used. Every model runs locally, on Ollama, or on Kaggle or Colab. Each member's notes, runs, and links stay in her own folder. The full link list is `links.txt`.

---

## Setup

```bash
git clone https://github.com/snehas-SJSU/data266-lab2-team05.git
cd data266-lab2-team05
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13. Open each notebook from the project root, or from `notebooks/`, and use Restart & Run All. The first cell prints the git commit, library versions, GPU, and seed. On Colab that commit prints `n/a` because the notebook is not inside this repository there.

---

## Files not in git

The lab says not to commit PDFs, datasets, API keys, or tokens. Weights, generated images, and the Part 3 inputs are on Hugging Face. `.gitignore` keeps everything inside a `data/` folder except a short README, so those folders look empty on GitHub.

| What | Where it comes from | Where a run reads it |
| --- | --- | --- |
| Part 1 PDFs and `Questions.csv` | Supplied with the lab | `Part1/data/` (not in git) |
| Part 2 training set, 833 pairs | [reach-vb/pokemon-blip-captions](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions) | The notebook downloads it. Cache: `Part2/data/pokemon-blip-captions`. On Colab the cache is on the VM. |
| Part 2 weights (Sneha) | [rank 16, final](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r16), [rank 4](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r4) | Hugging Face |
| Part 2 generated images (Sneha) | [data266-lab2-p2-images](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p2-images) | Hugging Face |
| Part 3 test images (Sneha) | Chosen by us. Same files: [data266-lab2-p3-inputs](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs) | Mac: `Part3/data/inputs/` (not committed). Colab: `MyDrive/data266_lab2_p3/inputs/`, with `eval/` in that same Drive folder. |
| Part 3 bonus app (Sneha) | [data266-lab2-p3-agents](https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents) | Hugging Face Space, free ZeroGPU |

The Part 2 notebook drops corrupt and duplicate images and makes a 90/10 split with seed 266. Part 3 does not download its test images. If `IMG07.png` is missing, the notebook creates one silver sedan with seed 266. The file list, including the CC BY 2.0 COCO photos, is in [Part3/data/README.md](Part3/data/README.md). More on the empty folders: [Part2/data/README.md](Part2/data/README.md). Ritika's Part 2 models, Part 2 images, and Part 3 inputs are still open in `links.txt`.

---

## How to run

Both members use the same three notebooks. Commands below are from the repo root, after Setup.

| Part | Member | Where | What to do |
| --- | --- | --- | --- |
| 1 | Sneha | Mac, Ollama | Start `qwen2.5vl:3b` and `qwen2.5:7b`, then Restart & Run All on `notebooks/p1_rag.ipynb`. |
| 1 | Both | Kaggle GPU | Use the notebook from tag `p1-v1.0`. Attach the competition data and `Part1/validate_submission.py`. Turn on a GPU and internet. Run All. It reads `Questions.csv`, writes `submission.csv`, and checks it with `validate_submission.py`. Submit that CSV. Limit: 90 minutes on one 16 GB GPU. Mac timings do not count. |
| 2 | Both | Colab GPU (T4, L4, or A100) | Upload `notebooks/p2_finetune.ipynb` and Run All. It installs the pinned packages, mounts Drive, and saves to `MyDrive/data266_lab2_p2/`. A finished step is skipped, and training resumes from the checkpoint after a disconnect. Prompts: `eval/p2_prompts.json`. |
| 2 | Sneha | Mac, smoke test | `.venv/bin/python Part2/sneha_singh/smoke_test/smoke_test.py` after the dataset is cached. 32 images, 5 LoRA steps. This checks the pipeline. It is not the graded run. |
| 2 | Sneha | Mac, demo | `.venv/bin/python Part2/sneha_singh/demo_generate.py "prompt one" "prompt two"` shows the base model beside the fine-tuned model. |
| 3 | Sneha | Mac, Ollama | `qwen2.5vl:3b`, inputs in `Part3/data/inputs/`. This path uses 384px and 15 steps. It is not the graded run. |
| 3 | Both | Colab GPU | Put `eval/` and `inputs/` in `MyDrive/data266_lab2_p3/`, upload `notebooks/p3_agents.ipynb`, and Run All. This path uses 512px and 30 steps. Delete `outputs/runs.jsonl` before a new graded run. The last cell starts Gradio and, on Colab, prints a public link. |

Ritika still needs her own Part 1 run, her own Part 2 training run, and her own Part 3 pipeline run. Notes for those go in her folders.

---

## Results so far

| Part | Member | Run | Result |
| --- | --- | --- | --- |
| 1 | Sneha | Local notebook | Run is in. Human ratings are 0/50. The timed Kaggle run is not done. The golden set still needs both members. Detail: [Part1/sneha_singh/results.md](Part1/sneha_singh/results.md). |
| 1 | Ritika | — | Not recorded yet. |
| 2 | Sneha | NVIDIA L4. Rank 16 kept. | KID 0.02314 (base 0.06024). CLIP 0.3155 (base 0.3220). 0.2399 GPU-hours. 4.27 GB peak. About 15 minutes for each LoRA run. Human win rate 0.387 (29/75, 95% CI 0.285 to 0.5). Detail: [Part2/sneha_singh/results.md](Part2/sneha_singh/results.md). |
| 2 | Ritika | Ratings only | She rated the 25 pairs. Vote: `Part2/sneha_singh/human_eval/ratings/ratings_Ritika_Neema.csv`. Her training run is not recorded. |
| 3 | Sneha | 26 Oct, NVIDIA L4 | Routed 30 of 30. Passed 6/15 without the critique loop and 4/15 with it. Critic agreement 34/48. Red-team attack success 0/10. Median 15.1 s without the loop and 31.4 s with it. Both members agreed on all 40 ratings. Detail: [Part3/sneha_singh/results.md](Part3/sneha_singh/results.md). |
| 3 | Ritika | Ratings only | She rated this run. Her own pipeline run is not in yet. |

---

## Models

| Part | Models | Hosted API |
| --- | --- | --- |
| 1, local | Ollama `qwen2.5vl:3b` for figure descriptions, `qwen2.5:7b` for answers; `all-MiniLM-L6-v2`; `ms-marco-MiniLM-L6-v2` reranker | none |
| 1, Kaggle | `Qwen/Qwen2.5-VL-7B-Instruct` through `transformers` | none |
| 2 | `stable-diffusion-v1-5/stable-diffusion-v1-5` with LoRA ranks 4 and 16; `openai/clip-vit-base-patch32`; LPIPS AlexNet; torchmetrics KID | none |
| 3 | `Qwen2.5-VL-3B` for language and vision; Stable Diffusion 1.5 img2img; InstructPix2Pix; the Part 2 rank-16 LoRA; CLIP ViT-B/32; the Stable Diffusion safety checker | none |

---

## Repo layout

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
├── Part1/ Part2/ Part3/      sneha_singh/ and ritika_neema/
│                             data/ holds a README only; images and PDFs are not committed
├── report/                   report draft and figures
├── CONTRIBUTIONS.md          who did what
├── AI_USAGE.md               AI tools
├── links.txt                 repo, tags, Hugging Face, Kaggle
└── .gitignore                data, secrets, weights
```

There is no `src/` folder. The code is in the notebooks. `Part2/sneha_singh/outputs/` holds generated images and adapters and is not committed. The lab PDFs in `Lab-requirement/` stay on this machine.

---

## Part 1 — Multimodal RAG

Shared notebook: `notebooks/p1_rag.ipynb`.

![Part 1 architecture: ingestion, retrieval, conversation memory, and evaluation](report/figures/part1/architecture.png)

### Sneha Singh — `Part1/sneha_singh/`

**Status:** local run is in. Human ratings 0/50. Timed Kaggle run is not done. Golden set still needs both members.
**Notes:** [Part1/sneha_singh/README.md](Part1/sneha_singh/README.md). Scores: [Part1/sneha_singh/results.md](Part1/sneha_singh/results.md).

### Ritika Neema — `Part1/ritika_neema/`

**Status:** her run is not recorded. She still needs to run the notebook and fill her Kaggle link in `links.txt`.
**Notes:** [Part1/ritika_neema/README.md](Part1/ritika_neema/README.md).

## Part 2 — Diffusion fine-tuning

Shared notebook: `notebooks/p2_finetune.ipynb`.

![Part 2 architecture: data preparation, LoRA fine-tuning, generation, and evaluation](report/figures/part2/architecture.png)

### Sneha Singh — `Part2/sneha_singh/`

**Status:** graded L4 run is done. Rank 16 kept. KID 0.02314 (base 0.06024), CLIP 0.3155 (base 0.3220), human win rate 0.387.
**Weights:** [rank 16](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r16), [rank 4](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r4), [images](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p2-images).
**Notes:** [Part2/sneha_singh/results.md](Part2/sneha_singh/results.md).

### Ritika Neema — `Part2/ritika_neema/`

**Status:** she rated the 25 pairs. Her own training run is not recorded. Her model and image links in `links.txt` are still open.
**Notes:** [Part2/ritika_neema/README.md](Part2/ritika_neema/README.md).

## Part 3 — Multimodal agents

Shared notebook: `notebooks/p3_agents.ipynb`.

![Part 3 architecture: planner, vision, prompt, generation, and critique](report/figures/part3/architecture.png)

### Sneha Singh — `Part3/sneha_singh/`

**Status:** 26 Oct L4 run is in. 6/15 passed without the critique loop, 4/15 with it. Attack success 0/10.
**Bonus:** Gradio app on free ZeroGPU, [data266-lab2-p3-agents](https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents).
**Inputs:** [data266-lab2-p3-inputs](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs).
**Notes:** [Part3/sneha_singh/README.md](Part3/sneha_singh/README.md). Scores: [Part3/sneha_singh/results.md](Part3/sneha_singh/results.md).

### Ritika Neema — `Part3/ritika_neema/`

**Status:** she rated this run. Her own pipeline run is not in yet. Her rating CSV and her Hugging Face input link are still open.
**Notes:** [Part3/ritika_neema/README.md](Part3/ritika_neema/README.md).
