# Part 2 dataset

This folder is empty on GitHub on purpose. `.gitignore` ignores everything under `data/` except this README and `.gitkeep`.

The training set is not stored in the repository. `notebooks/p2_finetune.ipynb` loads it itself:

- Hugging Face dataset: `reach-vb/pokemon-blip-captions` (833 image-caption pairs).
- Local cache, created on the first run: `Part2/data/pokemon-blip-captions`.
- If that cache is already there, the notebook uses it and does not download again.

On Colab the cache is on the VM, under the notebook working directory. It is not part of the git clone. The notebook then drops corrupt and duplicate images and makes a 90/10 split with seed 266. Test images are used only for evaluation.

The generated images from the graded run are a different dataset: [Sneha-subbu/data266-lab2-p2-images](https://huggingface.co/datasets/Sneha-subbu/data266-lab2-p2-images). The LoRA weights are [rank 16](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r16) and [rank 4](https://huggingface.co/Sneha-subbu/data266-lab2-p2-lora_r4).
