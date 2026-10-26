# Part 3 - Multimodal agents (Sneha Singh)

The notebook is [`../../notebooks/p3_agents.ipynb`](../../notebooks/p3_agents.ipynb). The five agents, the evaluation, and the Gradio app are in that notebook. [`results.md`](results.md) has the numbers the notebook printed.

## What I built

I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list before I ran anything. They are in [`../../eval/p3_scenarios.json`](../../eval/p3_scenarios.json), [`../../eval/p3_redteam.json`](../../eval/p3_redteam.json), and [`../../eval/p3_inputs.json`](../../eval/p3_inputs.json).

The flow is the one in the lab. The planner checks the request and picks a route. Vision describes the image. The prompt agent rewrites the request. Generation makes the image. Critique checks it and can send it back, at most twice. Caption questions stop after vision. An unsafe, vague, or unsupported request stops at the planner. Every request writes a JSON trace under `../../traces/`. A bad file or a crashed step becomes a short message, not a stack trace.

Generation has four modes. Style change is Stable Diffusion 1.5 img2img. Edits use InstructPix2Pix. Enhancement is ordinary image operations, judged by brightness and sharpness. A Pokemon-style request uses my Part 2 rank-16 LoRA from Hugging Face.

The Gradio app shows each agent's output. The CLIP cutoff, 0.26, was fit on the 120 Part 2 images, not on these 15 scenarios. The rating page is built by the notebook. How I use it is in [`human_eval/README.md`](human_eval/README.md).

`logic_tests.py` checks the critic rules with stand-in images. It does not call the real models.

## Run

Colab, for the graded run: a GPU runtime. Put `eval/` and an `inputs/` folder with the 25 files (`IMG01` through `FILE24`) in `MyDrive/data266_lab2_p3/`. Upload the notebook and Run All. A finished case is skipped if you run it again. The notebook I already ran used an NVIDIA L4, 512px, and 30 steps.

On my Mac I use Ollama with `qwen2.5vl:3b` and the inputs in `Part3/data/inputs/`. That path is not in git. The Mac uses 384px and 15 steps, so those pictures are not the graded ones.

The last cell starts Gradio. On Colab it prints a public link.

## What is done

The 26 Oct 2026 Colab run is in the notebook. Both rating files for that run are in `human_eval/ratings/`. The scores are in [`results.md`](results.md). The test images are at https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs. I deployed the same Gradio app as a Hugging Face Space for the Part 3 bonus: https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. The upload folder is `hf_space/`. It runs on the free ZeroGPU plan. Ritika's own pipeline is not in `../ritika_neema/`.
