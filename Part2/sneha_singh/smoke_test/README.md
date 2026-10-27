# Part 2 smoke test (local Mac, MPS)

Tiny end-to-end check before the real Colab run. Not results (32 images, 256px, 5 steps).

- `smoke_test.py`: script (run from repo root). Output images `smoke_base*.png`, `smoke_lora*.png`.
- `p2_smoke_executed_notebook_v1.ipynb`: first notebook version run in smoke mode.
- `p2_smoke_resume_test_executed.ipynb`: final notebook in smoke mode with a simulated disconnect at step 3 and resume from checkpoint.

Real run: `notebooks/sneha_singh/p2_finetune.ipynb`; artifacts in `Part2/sneha_singh/outputs/`.
