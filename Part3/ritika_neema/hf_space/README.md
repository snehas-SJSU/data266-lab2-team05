---
title: DATA 266 Lab 2 Part 3 Agents (Ritika)
emoji: "🎨"
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: other
hardware: zero-a10g
---

# Part 3 multimodal agents (Ritika Neema)

DATA 266 Lab 2, Team 05 (Sneha Singh, Ritika Neema). Bonus deployment of my Part 3 pipeline
(`notebooks/ritika_neema/p3_agents.ipynb`): Planner, Vision, Prompt, Generation and Critique, with the critique
loop revising at most twice. Seed 266.

Models: Qwen2.5-VL-3B-Instruct for the planner, vision, prompt and critique agents; Stable Diffusion 1.5
(txt2img and img2img); InstructPix2Pix for edits; my Part 2 LoRA rank 8
(`neema30/data266-lab2-p2-ritika-lora-r8`) for Pokemon-style requests; CLIP ViT-B/32 and the SD safety checker.

Hosted on Sneha Singh's Hugging Face account because mine is too new for free ZeroGPU hosting; the code is
mine, from `Part3/ritika_neema/hf_space/` in the team repo.

The first request after the Space wakes up loads every model, so it can take several minutes. The graded results
come from the Colab T4 run in the notebook, not from this Space.

Examples: `IMG05.jpg` is an MS COCO val2017 photo (CC BY 2.0); `IMG07.png` was generated with base Stable
Diffusion 1.5. The Part 2 LoRA was trained on reach-vb/pokemon-blip-captions (CC BY-NC-SA 4.0); non-commercial
research use only.
