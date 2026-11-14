# DATA 266 Lab 2 — Team 05

Sneha Singh and Ritika Neema. Canvas and Kaggle name: `Pair_Programming_Team_05`.

Each of us keeps her own notebooks under `notebooks/<name>/`. Part 1 answers questions about three economics PDFs. Part 2 fine-tunes Stable Diffusion 1.5 with LoRA on Pokémon captions. Part 3 is five agents that read an image, make a new one, and check it. Seed 266. No hosted API. Models run on our machines, through Ollama, or on Kaggle or Colab.

| Member | Folder |
| --- | --- |
| Sneha Singh | `sneha_singh/` under `Part1/`, `Part2/`, and `Part3/` |
| Ritika Neema | `ritika_neema/` under `Part1/`, `Part2/`, and `Part3/` |

Repo: [data266-lab2-team05](https://github.com/snehas-SJSU/data266-lab2-team05). Tags: `p1-v1.0`, `p2-v1.0`, `p3-v1.0` (same tip; see `git rev-parse p1-v1.0`). Links for both of us are in `links.txt`.

## Folder structure

```text
data266-lab2-team05/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── sneha_singh/
│   │   ├── p1_rag.ipynb
│   │   ├── p2_finetune.ipynb
│   │   └── p3_agents.ipynb
│   └── ritika_neema/
│       ├── p1_rag.ipynb
│       ├── p2_finetune.ipynb
│       └── p3_agents.ipynb
├── eval/
├── traces/
│   ├── sneha_singh/
│   └── ritika_neema/
├── Part1/  Part2/  Part3/
│   ├── sneha_singh/
│   └── ritika_neema/
├── report/
│   └── figures/
│       ├── part1/architecture.png
│       ├── part2/architecture.png
│       └── part3/architecture.png
├── CONTRIBUTIONS.md
├── AI_USAGE.md
├── links.txt
└── .gitignore
```

The code is in the notebooks. Each of us has a `sneha_singh/` or `ritika_neema/` folder under the three parts. `traces/sneha_singh/` and `traces/ritika_neema/` hold each member's Part 3 request logs (one JSON per request). `Part3/sneha_singh/hf_space/` is only the bonus app upload. PDFs and images stay out of git.

## Architecture images

Part 1 — Multimodal RAG:

![Part 1 architecture: ingestion, retrieval, conversation memory, and evaluation](report/figures/part1/architecture.png)

Part 2 — Diffusion fine-tuning:

![Part 2 architecture: data preparation, LoRA fine-tuning, generation, and evaluation](report/figures/part2/architecture.png)

Part 3 — Multimodal agents:

![Part 3 architecture: planner, vision, prompt, generation, and critique](report/figures/part3/architecture.png)

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

## Datasets

| Part | Sneha Singh | Ritika Neema |
| --- | --- | --- |
| 1 | Canvas and the Kaggle data tab: three economics PDFs, `Questions.csv`, `kb_manifest.json`. Team golden set: `eval/golden_set.json`. | Same competition files. Own golden/holdout: `eval/ritika_neema/`. |
| 2 | [reach-vb/pokemon-blip-captions](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions). Generated: [data266-lab2-p2-images](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p2-images). | Same train set. Generated: [data266-lab2-p2-ritika-images](https://huggingface.co/datasets/neema30/data266-lab2-p2-ritika-images). |
| 3 | [data266-lab2-p3-inputs](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs) (public). Bonus Space: [data266-lab2-p3-agents](https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents). | Same team inputs + Space. |

---

## Models

| Part | Sneha Singh | Ritika Neema |
| --- | --- | --- |
| 1 | Local: Ollama `qwen2.5:7b` / `qwen2.5vl:3b`. Kaggle: `Qwen/Qwen2.5-VL-7B-Instruct` (NF4). Public LB best **0.940**. | Kaggle: `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit). Public LB best **0.576** (reranker ON). |
| 2 | LoRA ranks **4** and **16** (kept 16). [r16](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r16), [r4](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r4). | LoRA **r8** and DoRA **r8** (kept LoRA r8). [LoRA](https://huggingface.co/neema30/data266-lab2-p2-ritika-lora-r8), [DoRA](https://huggingface.co/neema30/data266-lab2-p2-ritika-dora-r8). |
| 3 | `Qwen2.5-VL-3B`; SD 1.5 img2img; InstructPix2Pix; Part 2 rank-16 LoRA. | Same stack with her Part 2 LoRA r8. |

---

## How to run

### Sneha Singh

| Part | Where | What to do |
| --- | --- | --- |
| 1 | Kaggle / Mac | Graded: `notebooks/sneha_singh/p1_rag.ipynb` on Kaggle GPU ≤90 min. Local Ollama path measured 25/25 OFF and ON. [Scores](Part1/sneha_singh/results.md). |
| 2 | Colab | `notebooks/sneha_singh/p2_finetune.ipynb` on L4. [Scores](Part2/sneha_singh/results.md). |
| 3 | Colab | `notebooks/sneha_singh/p3_agents.ipynb` on L4 (26 Oct). [Scores](Part3/sneha_singh/results.md). |

**Bonus (Part 3 Space, +2, public):** [data266-lab2-p3-agents](https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents) from `Part3/sneha_singh/hf_space/`.

### Ritika Neema

| Part | Where | Status |
| --- | --- | --- |
| 1 | Kaggle T4 | Executed `notebooks/ritika_neema/p1_rag.ipynb`. Public LB **0.576** with reranker (team +2 claim: 16/25 vs 14/25 golden, 21/25 vs 19/25 holdout). Notes: `Part1/ritika_neema/`. |
| 2 | Colab T4 | Executed `notebooks/ritika_neema/p2_finetune.ipynb`. LoRA r8 vs DoRA r8; human win rate **0.747**. Notes: `Part2/ritika_neema/`. |
| 3 | Colab T4 | Executed `notebooks/ritika_neema/p3_agents.ipynb` (28 Oct). Both raters: 6/15 with and without critique. Notes: `Part3/ritika_neema/`. |

---

## Part 1 — Multimodal RAG

Sneha notebook: `notebooks/sneha_singh/p1_rag.ipynb`. Ritika notebook: `notebooks/ritika_neema/p1_rag.ipynb`. Architecture image is at the top of this README.

The PDFs stay read-only. Dense retrieval is the baseline. The final system mixes dense search with BM25 and an optional passage reranker.

| Measure | Sneha (local golden, 27 Oct) | Ritika (her golden / Kaggle) |
| --- | --- | --- |
| Answers correct | **25/25** OFF and ON | Golden **16/25** ON vs **14/25** OFF; holdout 21/25 vs 19/25 |
| Kaggle public LB | **0.940** | **0.576** (reranker ON) |
| +2 reranker bonus | Tie on this pipeline | Claimed on Ritika’s sets |

Full notes: [Part1/sneha_singh/results.md](Part1/sneha_singh/results.md), [Part1/ritika_neema/README.md](Part1/ritika_neema/README.md).

## Part 2 — Diffusion fine-tuning

Sneha: LoRA ranks 4 and 16 (`notebooks/sneha_singh/p2_finetune.ipynb`); kept rank 16.  
Ritika: LoRA r8 vs DoRA r8 (`notebooks/ritika_neema/p2_finetune.ipynb`); kept LoRA r8.  
Prompts: `eval/p2_prompts.json`.

## Part 3 — Multimodal agents

Sneha: `notebooks/sneha_singh/p3_agents.ipynb` (L4, 26 Oct) — 6/15 vs 4/15, critic **34/48**, attack **0/10**.  
Ritika: `notebooks/ritika_neema/p3_agents.ipynb` (T4, 28 Oct) — both raters 6/15 either way.  

Bonus (+2): public Space [data266-lab2-p3-agents](https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents). Inputs: [data266-lab2-p3-inputs](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs).
