# Ritika Neema - Part 2 results

Every number here is printed in `notebooks/ritika_neema/p2_finetune.ipynb` (Colab, Tesla T4, seed 266).

## Setup

Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), frozen in fp16. Adapters on the UNet
attention projections `to_q`, `to_k`, `to_v`, `to_out.0`, alpha equal to rank, adapter weights in fp32.
Dataset `reach-vb/pokemon-blip-captions` (CC BY-NC-SA 4.0). Both runs share the data, the sampled batches,
the noise, 1200 steps, batch 2 x 2 accumulation (4), AdamW, learning rate 0.0001 and seed 266; the only
change is LoRA versus DoRA.

Data preparation (Cell 3): 833 pairs, 0 corrupted, 0 exact duplicates, 2 near duplicates (average hash)
removed, 831 kept, 90/10 split with seed 266: 748 train, 83 test, centre-cropped to 512x512.

## Training (Cell 5)

| Run | Trainable parameters | Steps | Batch | LR | Time | GPU-hours | Peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA r8 (final) | 1,594,368 | 1200 | 4 | 0.0001 | 25.34 min | 0.4223 | 2.55 GB |
| DoRA r8 | 1,694,208 | 1200 | 4 | 0.0001 | 31.57 min | 0.5262 | 2.62 GB |

DoRA adds one magnitude vector per adapted layer (99,840 parameters) and costs about 25% more training time.

## Evaluation

KID: the 20 in-domain generations (10 prompts x seeds 0 and 1) against the 83 test images. CLIP score: cosine
between image and prompt embeddings (ViT-B/32), clamped at 0. LPIPS (AlexNet): seed 0 against seed 1 for each
prompt.

The Stable Diffusion safety checker blacked out 3 images (LoRA: prompts 3 and 6; DoRA: prompt 17; base: none).
They are cartoon prompts, so they are likely false positives, but the originals were not kept. A black image
still counts in KID and CLIP, so Cell 7b recomputes every metric with those prompt/seed pairs removed from all
three models (37 images each, 18 in-domain). That is the table used for the comparison.

| Model | KID (in-domain) | CLIP all | CLIP in-domain | CLIP out-of-domain | CLIP compositional | LPIPS seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Base | 0.08577 +/- 0.01124 | 0.3215 | 0.3231 | 0.3111 | 0.3299 | 0.7375 |
| LoRA r8 (final) | 0.05606 +/- 0.00671 | 0.3147 | 0.3219 | 0.2984 | 0.3183 | 0.7110 |
| DoRA r8 | 0.05352 +/- 0.00673 | 0.3167 | 0.3238 | 0.3072 | 0.3132 | 0.7033 |

As generated, before removing the blacked-out images (Cell 7): KID 0.08639 / 0.06081 / 0.05169 and CLIP all
0.3223 / 0.3113 / 0.3156 for base / LoRA r8 / DoRA r8. Generation took 5.51 / 5.72 / 6.41 s per image.

Memorization (Cell 8): nearest training image by CLIP image embedding for the 20 in-domain generations. LoRA r8
max 0.900 (mean 0.776), DoRA r8 max 0.912 (mean 0.798); 0 of 20 above 0.95 for both. The closest matches are
the "pikachu" prompt against Pikachu training images, the same character rather than a copy
(`memorization_*.png` in the notebook outputs).

Human evaluation (Cell 10): blinded 25 pairs, LoRA r8 against base, 3 raters. Pending.

## Choice

Both adapters move the images toward the dataset: KID falls from 0.0858 to 0.0561 and 0.0535, about three
base standard deviations, while in-domain CLIP stays at 0.322-0.324. LoRA and DoRA are tied: the KID gap
(0.0025) is inside one standard deviation, and the CLIP and LPIPS differences are small and point in opposite
directions (DoRA keeps out-of-domain prompts closer to the base model; LoRA does a little better on
compositional prompts and seed diversity). LoRA r8 is the final run because it is cheaper: 0.4223 versus
0.5262 GPU-hours and 5.72 versus 6.41 s per image, with the same peak memory.

## Failure modes

1. Compositional prompts do not bind colours or counts to objects, in any model. "Two pokemon, one red and
   one blue" puts the red on the background; "three small round pokemon, the left one purple" gets the count
   wrong. Fine-tuning does not fix it, and compositional CLIP drops (0.3299 to 0.3183).
2. The safety checker blacks out harmless cartoon prompts (a dragon, a star with a red eye, round creatures),
   which also distorts the raw metrics until the affected pairs are removed.

## Demo

`python Part2/ritika_neema/demo_generate.py "<prompt>" ...` writes base and LoRA r8 side by side for the same
seed. `smoke_test/` holds a 256px, 10-step check on a Mac that the adapter loads from Hugging Face.
