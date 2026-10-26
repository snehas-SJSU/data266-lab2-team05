"""Upload Part 2 model cards only. Does not retrain and does not re-upload weights.

The live rank-16 card rounds CLIP 0.3155 to 0.316. This writes the digits printed
in notebooks/p2_finetune.ipynb and uploads README.md for both adapters.

    huggingface-cli login
    .venv/bin/python Part2/sneha_singh/push_model_cards.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HF_USER = "Sneha-subbu"
RES = pd.read_csv(ROOT / "eval" / "p2_results.csv").set_index("model")
ROWS = {
    run: json.loads((ROOT / "Part2" / "sneha_singh" / "outputs" / run / "train_row.json").read_text())
    for run in ("lora_r4", "lora_r16")
}
HUMAN = (
    "Blinded human evaluation (25 pairs, 3 raters, 75 votes): rank 16 won 29/75 "
    "(win rate 0.387, 95% CI 0.285 to 0.5). "
    "Per rater: Rajesh 0.4, Ritika Neema 0.44, Sneha Singh 0.32. "
    "By prompt type: in-domain 0.41 (39 votes), out-of-domain 0.444 (18), compositional 0.278 (18)."
)


def card(run):
    t = ROWS[run]
    b, m = RES.loc["base"], RES.loc[run]
    final = " (selected final model)" if run == "lora_r16" else ""
    usage = (
        "```python\n"
        "import torch\n"
        "from diffusers import StableDiffusionPipeline\n"
        "pipe = StableDiffusionPipeline.from_pretrained("
        "'stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16).to('cuda')\n"
        f"pipe.load_lora_weights('{HF_USER}/data266-lab2-p2-{run}')\n"
        "```"
    )
    return f"""---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
library_name: diffusers
tags: [text-to-image, lora, stable-diffusion, diffusers]
datasets: [reach-vb/pokemon-blip-captions]
---
# {run}{final}

LoRA adapter (rank {t['rank']}) for Stable Diffusion 1.5, fine-tuned on Pokemon BLIP captions.
DATA 266 Lab 2, Part 2 - Team 05 (Sneha Singh, Ritika Neema).

## Intended use and limits
Non-commercial research and coursework only. The training images show copyrighted characters; do not use commercially.

## Training
750 train / 83 test images (seed 266), 512px, {t['steps']} steps, batch {t['batch']}, lr {t['lr']},
{t['trainable_params']:,} trainable parameters, peak GPU memory {t['peak_mem_gb']} GB, {t['gpu_hours']} GPU-hours (NVIDIA L4).

## Evaluation (20 fixed prompts, seeds 0 and 1, 512px, 30 steps, guidance 7.5)
| model | KID (lower is better) | CLIP score | LPIPS diversity |
|---|---|---|---|
| base | {b.KID_mean:.5f} +/- {b.KID_std:.5f} | {b.CLIP:.4f} | {b.LPIPS_diversity:.4f} |
| {run} | {m.KID_mean:.5f} +/- {m.KID_std:.5f} | {m.CLIP:.4f} | {m.LPIPS_diversity:.4f} |

{HUMAN}

## Known failure modes
Compositional prompts often lose objects, attributes or counts; the same creature look repeats across unrelated prompts;
out-of-domain prompts drift toward the cartoon style. A CLIP plus tile-distance check on 10 generations found no near-copies of training images.

## Usage
{usage}
"""


def main():
    from huggingface_hub import HfApi

    api = HfApi()
    written = {}
    for run in ("lora_r4", "lora_r16"):
        text = card(run)
        path = ROOT / "Part2" / "sneha_singh" / "outputs" / run / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written[run] = text
        print("wrote", path)
    for run, text in written.items():
        repo = f"{HF_USER}/data266-lab2-p2-{run}"
        api.upload_file(
            path_or_fileobj=text.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message="Align the model card with the notebook results table",
        )
        print("https://huggingface.co/" + repo)


if __name__ == "__main__":
    main()
