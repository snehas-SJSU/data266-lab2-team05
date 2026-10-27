# DATA 266 Lab 2 — Team 05

Sneha Singh and Ritika Neema. Canvas and Kaggle name: `Pair_Programming_Team_05`.

We share one notebook per part. Part 1 answers questions about three economics PDFs. Part 2 fine-tunes Stable Diffusion 1.5 with LoRA on Pokémon captions. Part 3 is five agents that read an image, make a new one, and check it. Seed 266. No hosted API. Models run on our machines, through Ollama, or on Kaggle or Colab.

| Member | Folder |
| --- | --- |
| Sneha Singh | `sneha_singh/` under `Part1/`, `Part2/`, and `Part3/` |
| Ritika Neema | `ritika_neema/` under `Part1/`, `Part2/`, and `Part3/` |

Repo: [data266-lab2-team05](https://github.com/snehas-SJSU/data266-lab2-team05). Tags: `p1-v1.0` at `1311cf4`, `p2-v1.0` at `6cbb16b`, `p3-v1.0` at `a5f796e`. Links for both of us are in `links.txt`.

---

## Setup

```bash
git clone https://github.com/snehas-SJSU/data266-lab2-team05.git
cd data266-lab2-team05
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13. Open a notebook from the repo root and use Restart & Run All. The first cell prints the commit, the library versions, the GPU, and the seed. On Colab that commit is `n/a`, because the notebook is not inside this repo there.

---

## Repo layout

```text
data266-lab2-team05/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── p1_rag.ipynb
│   ├── p2_finetune.ipynb
│   └── p3_agents.ipynb
├── eval/
├── traces/
├── Part1/  Part2/  Part3/
├── report/
├── CONTRIBUTIONS.md
├── AI_USAGE.md
├── links.txt
└── .gitignore
```

The code is in the notebooks. Each of us has a `sneha_singh/` or `ritika_neema/` folder under the three parts. `traces/` is the Part 3 log: one JSON file per request. The lab puts that folder here, next to `eval/`. `Part3/sneha_singh/hf_space/` is only the bonus app upload. PDFs and images stay out of git.

---

## Datasets

| Part | Sneha Singh | Ritika Neema |
| --- | --- | --- |
| 1 | Canvas and the Kaggle data tab: three economics PDFs, `Questions.csv`, `kb_manifest.json` | Same files |
| 2 | [reach-vb/pokemon-blip-captions](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions) | Same dataset. Her own run is not recorded yet. |
| 3 | [data266-lab2-p3-inputs](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs) | Not recorded yet |

---

## Models

| Part | Models |
| --- | --- |
| 1, local | Ollama `qwen2.5vl:3b` for figures, `qwen2.5:7b` for answers; `all-MiniLM-L6-v2`; `ms-marco-MiniLM-L6-v2` reranker |
| 1, Kaggle | `Qwen/Qwen2.5-VL-7B-Instruct` through `transformers` |
| 2 | `stable-diffusion-v1-5/stable-diffusion-v1-5`, LoRA ranks 4 and 16; `openai/clip-vit-base-patch32`; LPIPS AlexNet; torchmetrics KID |
| 3 | `Qwen2.5-VL-3B`; Stable Diffusion 1.5 img2img; InstructPix2Pix; the Part 2 rank-16 LoRA; CLIP ViT-B/32 |

---

## How to run

### Both

| Part | Where | What to do |
| --- | --- | --- |
| 1 | Kaggle | Open `notebooks/p1_rag.ipynb` from tag `p1-v1.0`. Attach the competition data and `Part1/validate_submission.py`. Turn on a GPU and internet. Run All. It writes `submission.csv` and checks it. Limit: 90 minutes on one 16 GB GPU. |
| 2 | Colab | Upload `notebooks/p2_finetune.ipynb` and Run All. It mounts Drive and saves to `MyDrive/data266_lab2_p2/`. A finished step is skipped. Training resumes from the checkpoint if the session drops. |
| 3 | Colab | Put `eval/` and `inputs/` in `MyDrive/data266_lab2_p3/`, upload `notebooks/p3_agents.ipynb`, and Run All. 512 pixels, 30 steps. The last cell starts Gradio. |

### Sneha Singh

| Part | Where | What to do |
| --- | --- | --- |
| 1 | Mac | Start Ollama `qwen2.5vl:3b` and `qwen2.5:7b`. Put the three PDFs in `Part1/data/`. Run All. [Scores](Part1/sneha_singh/results.md). |
| 2 | Mac | Weights: [rank 16](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r16), [rank 4](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r4). Images: [data266-lab2-p2-images](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p2-images). Two prompts: `.venv/bin/python Part2/sneha_singh/demo_generate.py "prompt one" "prompt two"`. [Scores](Part2/sneha_singh/results.md). |
| 3 | Space | Bonus app: [data266-lab2-p3-agents](https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents), free ZeroGPU. Upload folder: `Part3/sneha_singh/hf_space/`, own `requirements.txt` because that build is Python 3.10. [Notes](Part3/sneha_singh/README.md). [Scores](Part3/sneha_singh/results.md). |

### Ritika Neema

| Part | Where | What to do |
| --- | --- | --- |
| 1 | | |
| 2 | | |
| 3 | | |

---

## Part 1 — Multimodal RAG

Notebook: `notebooks/p1_rag.ipynb`.

The PDFs stay read-only. A fact added in one chat stays in that chat. Dense retrieval is the baseline. The final system mixes dense search with BM25.

![Part 1 architecture: ingestion, retrieval, conversation memory, and evaluation](report/figures/part1/architecture.png)

## Part 2 — Diffusion fine-tuning

Notebook: `notebooks/p2_finetune.ipynb`.

Two LoRA ranks, 4 and 16, on the same data and the same step budget. Rank 16 is the one we kept. Prompts are in `eval/p2_prompts.json`, written before training.

![Part 2 architecture: data preparation, LoRA fine-tuning, generation, and evaluation](report/figures/part2/architecture.png)

## Part 3 — Multimodal agents

Notebook: `notebooks/p3_agents.ipynb`.

The planner picks a route. Vision describes the image. The prompt agent rewrites the request. Generation makes the image. Critique can send it back, at most twice. A caption stops after vision. An unsafe request stops at the planner.

![Part 3 architecture: planner, vision, prompt, generation, and critique](report/figures/part3/architecture.png)
