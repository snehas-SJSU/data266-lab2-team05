---
title: DATA 266 Lab 2 Part 3 Agents
emoji: "🖼️"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: other
hardware: zero-a10g
---

# Part 3 multimodal agents

DATA 266 Lab 2, Team 05 (Sneha Singh, Ritika Neema).

Gradio app for the five agents: Planner, Vision Understanding, Prompt Engineering, Image Generation, and Critique. The critique loop revises at most twice. Seed 266.

The first request downloads Qwen2.5-VL-3B and the image model, so it can take several minutes. Later requests reuse the loaded models. A GPU is required. The Part 2 LoRA is `Sneha-subbu/data266-lab2-p2-lora_r16`. The training images show copyrighted characters; non-commercial research use only.
