"""Hugging Face Space entry for my Part 3 Gradio app. The agents are in agents.py (built from the notebook)."""
from agents import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
