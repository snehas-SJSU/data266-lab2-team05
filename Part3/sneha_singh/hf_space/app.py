"""Hugging Face Space entry for the Part 3 Gradio demo.

The five agents are in agents.py, the same code as the notebook. Models load on the
first request: Qwen2.5-VL-3B, then one diffusion pipeline at a time.
"""
import os

for folder in ("/tmp/p3_out", "/tmp/p3_traces", "/tmp/p3_inputs"):
    os.makedirs(folder, exist_ok=True)
os.environ["P3_OUT_DIR"] = "/tmp/p3_out"
os.environ["P3_TRACES_DIR"] = "/tmp/p3_traces"
os.environ["P3_INPUT_DIR"] = "/tmp/p3_inputs"

from agents import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
