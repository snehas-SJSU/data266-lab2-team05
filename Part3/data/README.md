# Part 3 test images

This folder is empty on GitHub on purpose. `.gitignore` ignores everything under `data/` except this README and `.gitkeep`. The photos, the broken test files, and `inputs_sources.json` stay on the machine that prepared them. They are not a Hugging Face dataset the notebook downloads.

`notebooks/p3_agents.ipynb` only reads a folder that is already there:

- On a Mac, `Part3/data/inputs/`.
- On Colab, upload that same folder to `MyDrive/data266_lab2_p3/inputs/` before Run All. Also upload `eval/`.

If `IMG07.png` is missing, the notebook creates it on the GPU: one silver sedan, seed 266. The other files have to be in the folder already. The notebook stops if `inputs/` is missing.

What is in the local `inputs/` folder:

| Files | Where they come from |
| --- | --- |
| IMG01, IMG02, IMG03, IMG05, IMG06, IMG08, IMG11, IMG13, IMG15, IMG20, IMG21, IMG22 | MS COCO val2017, CC BY 2.0 |
| IMG09 | A COCO photo darkened and slightly blurred |
| IMG14 | A COCO photo reduced to 32×32 with noise |
| IMG04 | Base Stable Diffusion 1.5, seed 0 |
| IMG07 | Created by the notebook if no photo is supplied |
| IMG12 | One image from the Part 2 test split, seed 266 |
| IMG16, IMG17, IMG18, IMG19 | Synthetic images with text drawn on them |
| FILE23, FILE24 | A broken file and an empty file |

The full source list, with COCO ids and URLs, is `Part3/data/inputs_sources.json` next to the images. That file is ignored by the same `data/` rule.

The only Hugging Face dataset this notebook loads by itself is the Part 2 generated-image set, `Sneha-subbu/data266-lab2-p2-images`, and it uses that only to set the CLIP cutoff. The same files are also at [Sneha-subbu/data266-lab2-p3-inputs](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p3-inputs).
