"""Part 2 demo: generate images for chosen prompts with the base model and the fine-tuned LoRA model, side by side.

Usage (repo root):
  .venv/bin/python Part2/sneha_singh/demo_generate.py "a drawing of a green pokemon with red eyes" "a photograph of a mountain lake"
Options: --lora <HF repo or local folder> (default: final rank-16 model), --seed 0, --steps 30, --res 512, --no-safety
Weights come from Hugging Face (base SD 1.5 + the LoRA adapter); a CUDA GPU or Apple MPS is used if available.
"""
import argparse, os, textwrap
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("prompts", nargs="+")
ap.add_argument("--lora", default="Sneha-subbu/data266-lab2-p2-lora_r16")
ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--out", default="Part2/sneha_singh/outputs/demo")
ap.add_argument("--no-safety", action="store_true", help="skip the NSFW check (default: check, report, and still show the image)")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if dev == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(a.base, torch_dtype=dtype, variant="fp16", safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True)

checker = None
if not a.no_safety:   # same NSFW checker as the notebook: it reports a flag, it does not blank the image
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor
    import numpy as np
    checker = StableDiffusionSafetyChecker.from_pretrained(a.base, subfolder="safety_checker", variant="fp16").to(dev).eval()
    fe = CLIPImageProcessor.from_pretrained(a.base, subfolder="feature_extractor")

def nsfw_flag(im):
    if checker is None: return "not checked"
    inp = fe(im, return_tensors="pt").to(dev)
    with torch.no_grad():
        _, f = checker(images=[np.array(im)], clip_input=inp.pixel_values.to(checker.dtype))
    return "flagged (shown anyway)" if f[0] else "ok"

def gen(prompt):
    g = torch.Generator("cpu").manual_seed(a.seed)               # same seed for base and fine-tuned
    return pipe(prompt, num_inference_steps=a.steps, guidance_scale=7.5, height=a.res, width=a.res, generator=g).images[0]

base_imgs = [gen(p) for p in a.prompts]                          # 1) base model (no adapter)
pipe.load_lora_weights(a.lora)                                   # 2) add the fine-tuned LoRA adapter
tuned_imgs = [gen(p) for p in a.prompts]

os.makedirs(a.out, exist_ok=True)
for n, (p, b, t) in enumerate(zip(a.prompts, base_imgs, tuned_imgs)):
    H = 56
    canvas = Image.new("RGB", (2 * a.res, a.res + H), "white"); d = ImageDraw.Draw(canvas)
    canvas.paste(b, (0, H)); canvas.paste(t, (a.res, H))
    d.text((6, 4), "base: " + textwrap.shorten(p, 90), fill="black"); d.text((a.res + 6, 4), "fine-tuned: " + a.lora.split("/")[-1], fill="black")
    d.text((6, 22), "NSFW check: " + nsfw_flag(b), fill="gray"); d.text((a.res + 6, 22), "NSFW check: " + nsfw_flag(t), fill="gray")
    f = f"{a.out}/demo_{n}.png"; canvas.save(f); print("saved", f, "|", p)
