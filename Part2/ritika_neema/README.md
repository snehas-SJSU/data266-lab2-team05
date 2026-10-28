# Ritika Neema - Part 2

My own pipeline is `notebooks/ritika_neema/p2_finetune.ipynb`. Two runs on Stable Diffusion 1.5 with the same
data, steps, batches, noise and seed: **LoRA rank 8** and **DoRA rank 8** on the UNet attention projections
(`to_q`, `to_k`, `to_v`, `to_out.0`). Sneha's runs compare LoRA ranks 4 and 16; mine compare LoRA with DoRA at
one rank. Prompts are the team file `eval/p2_prompts.json` (committed before tuning).

| Step | What the notebook does |
| --- | --- |
| Data | `reach-vb/pokemon-blip-captions` (CC BY-NC-SA 4.0). Removes unreadable images and images without a caption, exact duplicates (pixel MD5) and near duplicates (256-bit average hash), and prints each count. 90/10 split with seed 266 (fixed by the lab), centre crop to 512x512. Training batches, noise and the blinded-pair sampling also use seed 266. |
| Training | VAE latents and text embeddings cached once. 1200 steps, batch 2 x 2 accumulation, LR 1e-4, AdamW, fp16 UNet with fp32 adapter weights. Prints trainable parameters, time, GPU-hours, peak memory. Checkpoints pushed to Hugging Face every 400 steps. |
| Generation | 20 prompts x seeds 0 and 1 for base, LoRA r8 and DoRA r8 (120 images). The Stable Diffusion safety checker screens every image; flagged ones are blacked out and counted. 8 side-by-side grids. |
| Evaluation | KID (in-domain generations vs the test split), CLIP score by prompt type, LPIPS between seeds, nearest-training-image CLIP similarity (memorization), blinded 25-pair sheet (final vs base). |
| Publish | Both adapters with model cards and all images to Hugging Face. |

## How to run (Colab T4)

1. Colab secrets: `HF_TOKEN` (write access) and `GH_TOKEN` (read access to the team repo).
2. Optional: mount Drive and set `P2_OUT=/content/drive/MyDrive/p2_ritika` so outputs survive a disconnect.
3. Run All. If one session is too short for both runs, set `P2_RUNS=lora_r8` in the first session and
   `P2_RUNS=dora_r8` in the second; a finished run is downloaded back from Hugging Face.
4. Three or more raters fill `human_eval/ratings_template.csv` with L, R or T per pair. Save it as
   `eval/ritika_neema/p2_human_ratings.csv` and re-run the win-rate cell.
5. Download the executed notebook and commit it with its outputs.

## Result and choice

Final run: **LoRA rank 8**
([neema30/data266-lab2-p2-ritika-lora-r8](https://huggingface.co/neema30/data266-lab2-p2-ritika-lora-r8)).
Data preparation removed 0 corrupted images, 0 exact duplicates and 2 near duplicates: 833 pairs became 831,
split into 748 train and 83 test. Cell 7b compares all models on the same 37 images after removing the 3 images the safety checker blacked out.
Both adapters lower KID well below the base model, but LoRA and DoRA are tied within one standard deviation,
and their CLIP and LPIPS differences are small and point in opposite directions. LoRA r8 trains and generates
faster, so it is the final run. Exact numbers are in the executed notebook (Cells 5, 7 and 7b).

Failure modes seen in the grids: (1) compositional prompts do not bind colours or counts to the right objects
in any model; (2) the safety checker blacks out harmless cartoon prompts, which also distorts the raw metrics.
