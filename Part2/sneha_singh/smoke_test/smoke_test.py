"""Part 2 smoke test: tiny end-to-end pass (data -> LoRA steps -> generate -> KID/CLIP/LPIPS).

Not a result. Checks the pipeline runs on this machine before moving to a CUDA GPU.
Run from repo root:  .venv/bin/python Part2/sneha_singh/smoke_test/smoke_test.py
"""
import hashlib, random, time
import numpy as np, torch
from datasets import load_from_disk
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig
from torchvision import transforms

SEED, RES, N_IMG, STEPS, GEN_STEPS = 266, 256, 32, 5, 10
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print({"device": dev, "torch": torch.__version__, "seed": SEED})

# 2.1 data: drop corrupt + duplicate, 90/10 split, resize
ds = load_from_disk("Part2/data/pokemon-blip-captions").select(range(N_IMG))
seen, keep, corrupt, dup = set(), [], 0, 0
for i, ex in enumerate(ds):
    try:
        ex["image"].convert("RGB").load()
    except Exception:
        corrupt += 1; continue
    h = hashlib.md5(ex["image"].convert("RGB").resize((32, 32)).tobytes()).hexdigest()
    if h in seen:
        dup += 1; continue
    seen.add(h); keep.append(i)
print({"kept": len(keep), "corrupt": corrupt, "duplicate": dup})
rng = random.Random(SEED); rng.shuffle(keep)
n_test = max(1, round(0.1 * len(keep)))
test_idx, train_idx = keep[:n_test], keep[n_test:]
tf = transforms.Compose([transforms.Resize(RES), transforms.CenterCrop(RES)])
rgb = lambda i: tf(ds[i]["image"].convert("RGB"))
train = [(rgb(i), ds[i]["text"]) for i in train_idx]
test = [(rgb(i), ds[i]["text"]) for i in test_idx]
print({"train": len(train), "test": len(test)})

# 2.2 LoRA training (a few steps)
pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32, variant="fp16",
                                       safety_checker=None).to(dev)
vae, unet, te, tok = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
for m in (vae, unet, te):
    m.requires_grad_(False)
unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian",
                            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
params = [p for p in unet.parameters() if p.requires_grad]
print({"trainable_params": sum(p.numel() for p in params)})
opt = torch.optim.AdamW(params, lr=1e-4)
to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
unet.train(); t0 = time.time()
for step in range(STEPS):
    img, cap = train[step % len(train)]
    x = to_t(img).unsqueeze(0).to(dev)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        ids = tok(cap, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        emb = te(ids)[0]
    noise = torch.randn_like(lat)
    t = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=dev)
    pred = unet(noise_sched.add_noise(lat, noise, t), t, emb).sample
    loss = torch.nn.functional.mse_loss(pred, noise)
    loss.backward(); opt.step(); opt.zero_grad()
    print(f"step {step+1}/{STEPS} loss {loss.item():.4f}")
unet.eval()
print({"train_seconds": round(time.time() - t0, 1)})

# 2.3 generate: base (adapter off) vs lora (adapter on), seeds 0 and 1
prompt = test[0][1]
def gen(adapter_on):
    unet.enable_adapters() if adapter_on else unet.disable_adapters()
    out = []
    for s in (0, 1):
        g = torch.Generator("cpu").manual_seed(s)
        out.append(pipe(prompt, num_inference_steps=GEN_STEPS, height=RES, width=RES,
                        generator=g).images[0])
    return out
base_imgs, lora_imgs = gen(False), gen(True)
print({"prompt": prompt})

# 2.4 metrics
from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def clip_score(imgs, text):
    inp = proc(text=[text], images=imgs, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        o = clip(**inp)
    return (o.logits_per_image / clip.logit_scale.exp()).mean().item()
print({"clip_base": clip_score(base_imgs, prompt), "clip_lora": clip_score(lora_imgs, prompt)})

import lpips
lp = lpips.LPIPS(net="alex").to(dev)
tt = lambda im: to_t(im).unsqueeze(0).to(dev)
with torch.no_grad():
    print({"lpips_base": lp(tt(base_imgs[0]), tt(base_imgs[1])).item(),
           "lpips_lora": lp(tt(lora_imgs[0]), tt(lora_imgs[1])).item()})

from torchmetrics.image.kid import KernelInceptionDistance
u8 = lambda ims: torch.stack([torch.from_numpy(np.array(i.resize((299, 299)))).permute(2, 0, 1) for i in ims])
real = u8([i for i, _ in test] * 2)
kid = KernelInceptionDistance(subset_size=2).to("cpu")  # tiny n: value is noise, only checks it runs
kid.update(real, real=True); kid.update(u8(lora_imgs), real=False)
print({"kid_mean_std(noise at n=2)": [float(v) for v in kid.compute()]})

# memorization check mechanics: nearest training image by CLIP similarity
def emb(ims):
    inp = proc(images=ims, return_tensors="pt").to(dev)
    with torch.no_grad():
        e = clip.get_image_features(**inp)
    e = e if torch.is_tensor(e) else e.pooler_output
    return torch.nn.functional.normalize(e, dim=-1)
sim = emb(lora_imgs) @ emb([i for i, _ in train]).T
print({"max_clip_sim_to_train": sim.max(dim=1).values.tolist()})

for n, im in zip(["base0", "base1", "lora0", "lora1"], base_imgs + lora_imgs):
    im.save(f"Part2/sneha_singh/smoke_test/smoke_{n}.png")
print("SMOKE TEST OK")
