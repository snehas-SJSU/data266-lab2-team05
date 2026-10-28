# Part 2 smoke test (local Mac, MPS)

Check that the final adapter `neema30/data266-lab2-p2-ritika-lora-r8` downloads from Hugging Face, loads into
Stable Diffusion 1.5 and changes the output. Not results: 256px and 10 steps, so the images are rough.

```bash
python Part2/ritika_neema/demo_generate.py "a drawing of a green pokemon with red eyes" \
  "a red sports car on a city street" --res 256 --steps 10 --out Part2/ritika_neema/smoke_test
```

`demo_0_seed0.png` and `demo_1_seed0.png`: base on the left, LoRA r8 on the right, same seed. Run on
28 Oct 2026, about 2 minutes on Apple MPS including model loading.
