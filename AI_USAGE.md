# AI usage

Each member records her own AI tools and what they were used for; do not edit the other member's section.

## Sneha Singh

Same four questions as my homework `AI_USE.md` files. The assistant is Cursor. I wrote the code. I used it for research, for help while I was writing the hard parts of that code, and for fixing. The runs, the tables, and the final sentences are mine.

### Part 1

1. Which parts did you use an assistant for, and which did you write yourself?

I built `notebooks/p1_rag.ipynb` and ran it locally. Parsing the three PDFs, dense versus hybrid retrieval, and the golden-set comparison are mine. The results table in the report stays empty until the notebook number we both agree to report is the one in the file.

I wrote that code. Low-trust records stay in a quarantine collection and are never queried. Appendix C's chart is not sent to the vision model. Retrieved text and pixels are data. A user fact stays in that chat and is not written into ChromaDB.

Where I used AI for research. I was writing the safety section of `Part1/sneha_singh/results.md` and report section II, and I used the assistant to read the papers with me. I wrote the section. Lewis et al. is retrieve, then generate, so I do not answer from the model alone. Faysse et al. is why a figure has to be searchable as an image, not only as the words next to it. Greshake et al. is the one I used for Appendix C. An instruction can sit inside retrieved text or inside a picture, so that page stays out of retrieval and out of the vision model.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first safety draft treated Appendix C like the other pages: read it and answer from it. That is the wrong source. Appendix C is the low-trust page. Figure C-1 is an instruction aimed at the model, not a chart to describe.

3. How did you find out? What did the failure look like?

I checked the manifest trust levels against that draft. Appendix C is marked low. A question about the appendix has to be answered from the manifest. Answering from the page follows the instruction in the figure.

4. What did you change, and why does your version work?

That page stays out of retrieval and out of the image passed to the vision model. A fact a user adds in one chat dies with that chat. It is not written back into the three PDFs or the index. I did not fill the Appendix C examples in the report by hand. Those examples are still not written.

### Part 2

1. Which parts did you use an assistant for, and which did you write yourself?

I wrote `notebooks/p2_finetune.ipynb` and ran it on a Colab L4. The code is mine. That covers 2.1 data preparation, the 20 prompts, 2.2 fine-tuning, 2.3 image generation, 2.4 evaluation, the blinded rating page, the safety check, and the failure-mode section. Data prep, both LoRA ranks, the 120 images, KID, CLIP, LPIPS, the rating page, and the weight upload are mine.

Where I used AI while writing that code. I still wrote each of these.

- Section 2.1 Data preparation. The split cell: seed 266, 90/10, resize and center-crop to 512.
- Section 2.2 Fine-tuning, function `train_lora`. The adapter targets `to_q`, `to_k`, `to_v`, and `to_out.0`. The VAE and the text encoder stay frozen. Rank 4 and rank 16 share the data, the steps, the batch, the learning rate, and the seed.
- Section 2.4 Evaluation, functions `kid`, `clip_score`, and `diversity`. `kid` compares 20 in-domain images with the 83 test images.
- Section Safety, functions `cemb` and `tile_dist`. `cemb` finds the nearest training image by CLIP on 10 generations. `tile_dist` measures the 3 closest pairs.

Where I used AI for research. I was writing the architecture in `Part2/sneha_singh/results.md` and report section III, and I used the assistant to read the three papers with me. I wrote the section so it matches this run.

Rombach et al. is the Stable Diffusion paper. The picture goes through a VAE, the UNet predicts the noise, and the caption comes in through attention. I already had that model. I did not train it from scratch. In `train_lora` I freeze the VAE and the text encoder and only train the attention adapters.

Hu et al. is the LoRA paper. You do not update the whole weight matrix. You add a small low-rank piece on top. That is why rank 4 and rank 16 share the same 750 images, 1500 steps, batch 4, learning rate 0.0001, and seed 266. The rank is the only change. Rank 4 has 797,184 trainable parameters. Rank 16 has 3,188,736. I kept rank 16 because KID went from 0.06024 to 0.02314 and the time and memory were the same. A bigger rank is not always better, and I am not saying that.

Carlini et al. look for a training image that got copied. They generate hundreds of pictures per prompt, and a near-copy on their scale is about 0.15. I did the smaller check in `cemb` and `tile_dist`: 10 generations and one seed. A CLIP score of 0.789 to 0.938 looked like a copy when I first read it. Next to their 0.15 it is not the same thing. The three closest pairs are the same kind of creature, with a different pose and different colors. I wrote that. I did not run their extraction attack.

Where I used AI for fixing. The results and safety sentences had rounded past the notebook. After I logged in, `Part2/sneha_singh/push_model_cards.py` uploaded only the two model-card READMEs. The weights were not trained again.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The early report paragraph rounded the notebook. Rank 16 CLIP was written as 0.316, and both runs were written as 0.24 GPU-hours:

```
Prompt match and diversity drop a little (CLIP 0.322, 0.311, 0.316; LPIPS 0.735, 0.718, 0.712).
Each run took 0.24 GPU-hours on one NVIDIA L4 and peaked near 4.2 GB.
```

The notebook table already had rank 16 CLIP at 0.3155. The two GPU-hour prints are 0.2401 and 0.2399, not one shared 0.240.

3. How did you find out? What did the failure look like?

I put that paragraph next to the saved notebook table. The prose and the table did not match. The same draft said the nearest training images scored 0.79 to 0.94, which reads like a copy. Carlini et al. call a near-copy something around 0.15 on their own scale. The three closest pairs are the same kind of creature, different pose and color.

4. What did you change, and why does your version work?

I rewrote those sentences from the notebook print. KID is 0.06024 for the base model and 0.02314 for rank 16. CLIP is 0.3155, not 0.316. GPU-hours are 0.2401 and 0.2399. The human win rate stays 0.387, and the interval still reaches 0.5, so I kept rank 16 and did not train again. The human line names the CSVs: Rajesh 0.4, Ritika Neema 0.44, Sneha Singh 0.32. Rajesh is the rater from outside the team. After I logged in, only the two model-card READMEs were uploaded. The weights were not trained again.

### Part 3

1. Which parts did you use an assistant for, and which did you write yourself?

I wrote `notebooks/p3_agents.ipynb` and `Part3/sneha_singh/space/`. The code is mine. That covers the shared state, the vision-language model, Planner, Vision, Prompt, Generation, Critique, the orchestrator, the demo, the rating page, CLIP calibration, evaluation, and Results. I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list. I ran the notebook on an L4 on 26 Oct 2026. Ritika and I rated that run. The Results cell printed 6/15 without the critique loop and 4/15 with it, critic agreement 34/48, attack success 0/10, and median 15.1 s and 31.4 s. I did not change those numbers.

Where I used AI while writing that code. I still wrote each of these.

- Section Shared state and trace. Classes `State` and `Step`. Methods `log` and `save_trace`.
- Section Agent 1: Planner. Function `planner`, and `check_image` for a broken or empty file. An unsafe request stops here.
- Section Agent 2: Vision Understanding. Function `vision`. Text found in the image is stored as data.
- Section Agent 3: Prompt Engineering. Functions `prompt_agent`, `_overlap_injection`, and `_grounded_in_request`. Image text is not copied into the generation prompt.
- Section Agent 4: Image Generation. Function `generator_agent` and class `Hub`. The branch is img2img, InstructPix2Pix, `enhance_image`, or the Part 2 rank-16 LoRA.
- Section Agent 5: Critique. Function `critique`, with `clip_sim`. A check stays only when `_grounded_in_request` says it is in the request. The image is accepted only when every check passes.
- Section Orchestrator. Function `run_request`. The revision cap is 2. A caption or a question stops after `vision`.
- Section CLIP threshold calibration. Function `calibrate_clip_threshold`. The cutoff 0.26 is fit on the 120 Part 2 images, not on the 15 scenarios.

Where I used AI for research. I was writing the safety section of `Part3/sneha_singh/results.md` and report section IV, and I used the assistant to read Greshake et al. with me. The attack is not someone typing "ignore the rules" in the chat. They hide that line in something the model will look at, like text in a picture, and the model treats it as an order.

I used that idea while I was writing `vision`, `prompt_agent`, and `critique`, and again when I wrote the safety section. `vision` writes down the text it sees. It does not follow it. `_overlap_injection` and `prompt_agent` keep that text out of the prompt that goes to the image model, even if the picture says to skip the checks. `_grounded_in_request` only keeps a critic check when the words are in my request, so a phrase copied off the image cannot make the critic accept or revise. I tested that with the 10 red-team cases. R01 and R04 came back as captions and did not follow the text in the picture. R05 through R08 were refused before any image was made. Attack success stayed 0/10. I wrote that the rule held on this run. I did not write that the attack from the paper is solved.

Where I used AI for fixing, the bonus Space. The graded app already ran. The free Hugging Face build did not. The fixes I kept, after I checked each log, are in `Part3/sneha_singh/space/`:

- `requirements.txt`. `diffusers==0.41.0` because that image is Python 3.10. `pydantic==2.12.5` because gradio's MCP extra rejects 2.14.1.
- `agents.py`, class `VLM`, method `_load_hf`. The processor is built without AutoProcessor, because torchvision is not in that image.
- `agents.py`, function `app_fn`. `import spaces` is the first import. The GPU hook on `app_fn` is the literal `@spaces.GPU` decorator. A request with no image does not go through the image processor.

I uploaded each file. "Draw a small fire fox creature in Pokemon style" returned an image. The Space is https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. I did not buy a larger quota.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The Space `requirements.txt` pinned `diffusers==0.42.0`. The Space is Python 3.10. That pin cannot install. The build log ended with:

```
No matching distribution found for diffusers==0.42.0
```

3. How did you find out? What did the failure look like?

I opened the Space build log after Factory rebuild. pip stopped there. The app never started. The next uploads failed in a new place each time: no `@spaces.GPU` function found at startup, then CUDA already initialized because `import spaces` was below torch, then `TypeError: argument of type 'NoneType' is not iterable` inside AutoProcessor because torchvision is not in that image.

4. What did you change, and why does your version work?

I pinned `diffusers==0.41.0` and `pydantic==2.12.5`. `import spaces` is the first import. The GPU hook is the literal `@spaces.GPU` decorator. The vision processor is built without AutoProcessor. I uploaded each file myself. The request "Draw a small fire fox creature in Pokemon style" returned an image, and the critic stopped at 2 revisions. That Space is https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. The free daily quota was then used up. I did not buy a larger quota.

## Ritika Neema

### Part 1

To be filled by Ritika: AI tools used, what for, and what she reviewed. Nothing recorded yet.

### Part 2

To be filled by Ritika.

### Part 3

To be filled by Ritika.
