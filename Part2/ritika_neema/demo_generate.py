"""Part 2 demo: base Stable Diffusion 1.5 next to my fine-tuned adapter, same prompt and seed.

Usage (repo root):
  python Part2/ritika_neema/demo_generate.py "a drawing of a green pokemon with red eyes" "a red sports car"
Options: --adapter <HF repo or local folder> (default: my final LoRA rank 8), --seed 0, --steps 30, --res 512

The adapter is the peft format written by notebooks/ritika_neema/p2_finetune.ipynb
(adapter_config.json + adapter_model.safetensors). CUDA, Apple MPS or CPU, whichever is available.
"""
import argparse, json, os, textwrap

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from PIL import Image, ImageDraw
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("prompts", nargs="+")
ap.add_argument("--adapter", default="neema30/data266-lab2-p2-ritika-lora-r8")
ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--out", default="Part2/ritika_neema/outputs/demo")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if dev == "cuda" else torch.float32
folder = a.adapter if os.path.isdir(a.adapter) else snapshot_download(a.adapter)
cfg = json.load(open(os.path.join(folder, "adapter_config.json")))

def load(with_adapter):
    pipe = StableDiffusionPipeline.from_pretrained(a.base, torch_dtype=dtype, safety_checker=None,
                                                   requires_safety_checker=False).to(dev)
    if with_adapter:
        pipe.unet.add_adapter(LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"], init_lora_weights="gaussian",
                                         target_modules=cfg["target_modules"], use_dora=cfg["use_dora"]))
        state = load_file(os.path.join(folder, "adapter_model.safetensors"))
        set_peft_model_state_dict(pipe.unet, {k: v.to(dtype) for k, v in state.items()})
    pipe.set_progress_bar_config(disable=True)
    return pipe

def generate(pipe, prompt):
    g = torch.Generator("cpu").manual_seed(a.seed)   # CPU generator: same noise on every device
    return pipe(prompt, num_inference_steps=a.steps, guidance_scale=7.5, height=a.res, width=a.res,
                generator=g).images[0]

kind = f"{'DoRA' if cfg['use_dora'] else 'LoRA'} r{cfg['rank']}"
os.makedirs(a.out, exist_ok=True)
images = {}
for name, with_adapter in (("base", False), (kind, True)):
    pipe = load(with_adapter)
    images[name] = [generate(pipe, p) for p in a.prompts]
    del pipe
    if dev == "cuda":
        torch.cuda.empty_cache()

for i, prompt in enumerate(a.prompts):
    sheet = Image.new("RGB", (2 * a.res, a.res + 60), "white")
    draw = ImageDraw.Draw(sheet)
    for j, name in enumerate(images):
        sheet.paste(images[name][i], (j * a.res, 60))
        draw.text((j * a.res + 8, 6), name, fill="black")
    draw.text((8, 24), textwrap.shorten(prompt, 120), fill="black")
    path = os.path.join(a.out, f"demo_{i}_seed{a.seed}.png")
    sheet.save(path)
    print(f"{path}  |  {prompt}")
print(f"device {dev}, {a.steps} steps, {a.res}px, seed {a.seed}, adapter {a.adapter}")
