# AI usage

Each member records her own AI tools and what they were used for; do not edit the other member's section.

## Sneha Singh

Same four questions as my homework `AI_USE.md` files. The coding assistant is Cursor. Each part lists where it was used in the code, then in the writing. The runs, the tables, and the final sentences are mine.

### Part 1

1. Which parts did you use an assistant for, and which did you write yourself?

I built `notebooks/p1_rag.ipynb` and ran it locally. Parsing the three PDFs, dense versus hybrid retrieval, and the golden-set comparison are mine. The results table in the report stays empty until the notebook number we both agree to report is the one in the file.

Coding the assistant touched in that notebook:

- The quarantine collection. Low-trust records are stored and never queried.
- The caption step. Appendix C's chart is not sent to the vision model.
- The answer rule. Retrieved text and pixels are data, and Appendix C is not evidence.
- Conversation memory. A user fact stays in that chat and is not written into ChromaDB.

Writing. The safety section of `Part1/sneha_singh/results.md` and report section II. The research was Lewis et al. (retrieve, then generate), Faysse et al. (a figure has to be searchable as an image), and Greshake et al. (an instruction can sit inside retrieved text or inside a picture).

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first safety draft treated Appendix C like the other pages: read it and answer from it. That is the wrong source. Appendix C is the low-trust page. Figure C-1 is an instruction aimed at the model, not a chart to describe.

3. How did you find out? What did the failure look like?

I checked the manifest trust levels against that draft. Appendix C is marked low. A question about the appendix has to be answered from the manifest. Answering from the page follows the instruction in the figure.

4. What did you change, and why does your version work?

That page stays out of retrieval and out of the image passed to the vision model. A fact a user adds in one chat dies with that chat. It is not written back into the three PDFs or the index. I did not fill the Appendix C examples in the report by hand. Those examples are still not written.

### Part 2

1. Which parts did you use an assistant for, and which did you write yourself?

I ran `notebooks/p2_finetune.ipynb` on a Colab L4. Data prep, both LoRA ranks, the 120 images, KID, CLIP, LPIPS, the safety checker, the memorization figure, the blinded rating page, and the weight upload are mine.

Coding the assistant touched:

- The LoRA targets in the training notebook: `to_q`, `to_k`, `to_v`, and `to_out.0`. The VAE and the text encoder stay frozen. Rank 4 and rank 16 share the same data, steps, batch, learning rate, and seed.
- The memorization check: nearest training image by CLIP on 10 in-domain generations, then the tile distance on the 3 closest pairs.
- `Part2/sneha_singh/push_model_cards.py`. After I logged in, it uploaded only the two model-card READMEs. The weights were not trained again.

Writing. The architecture in `Part2/sneha_singh/results.md` and report section III, while reading the papers. Rombach et al. train a full latent diffusion model. Hu et al. add a low-rank adapter. Carlini et al. hunt for a pixel copy with a large extraction attack. The results and safety sentences were the other writing pass, where the draft had rounded past the notebook.

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

I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list. I built the five agents and ran them on an L4 on 26 Oct 2026. Ritika and I rated that run. The Results cell printed 6/15 without the critique loop and 4/15 with it, critic agreement 34/48, attack success 0/10, and median 15.1 s and 31.4 s. I did not change those numbers.

Coding the assistant touched, in `notebooks/p3_agents.ipynb` and in `Part3/sneha_singh/space/`:

- The critic. A check is kept only when it is grounded in the request. The image is accepted only when every check passes. The CLIP cutoff of 0.26 is fit on the 120 Part 2 images, not on the 15 scenarios.
- `requirements.txt` on the Space: `diffusers==0.41.0` and `pydantic==2.12.5`.
- `agents.py`. `import spaces` is the first import. The GPU hook is the literal `@spaces.GPU` decorator. The vision processor is built without AutoProcessor.
- The text-only path. A request with no image does not go through the image processor.

Writing. The safety section of `Part3/sneha_singh/results.md` and report section IV. The research was Greshake et al. again: text inside an uploaded image is data, so the prompt agent does not copy it. The bonus is that same app as a Hugging Face Space, and the free build would not start.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The Space `requirements.txt` pinned `diffusers==0.42.0`. The Space is Python 3.10. That pin cannot install. The build log ended with:

```
No matching distribution found for diffusers==0.42.0
```

3. How did you find out? What did the failure look like?

I opened the Space build log after Factory rebuild. pip stopped there. The app never started. The next uploads failed in a new place each time: no `@spaces.GPU` function found at startup, then CUDA already initialized because `import spaces` was below torch, then `TypeError: argument of type 'NoneType' is not iterable` inside AutoProcessor because torchvision is not in that image.

4. What did you change, and why does your version work?

I pinned `diffusers==0.41.0` and `pydantic==2.12.5`. `import spaces` is the first import. The GPU hook is the literal `@spaces.GPU` decorator. The vision processor is built without AutoProcessor. I uploaded each file myself. The request "Draw a small fire fox creature in Pokemon style" returned an image, and the critic stopped at 2 revisions. That Space is https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. The free daily quota was then used up. I did not buy a larger quota.

### Part 2: about 70 percent mine, about 30 percent where I used AI

Out of the whole Part 2 notebook, about 70 percent is work I did, and about 30 percent is where I used Cursor. I can point at the notebook for both.

The 70 percent is the Colab L4 run. I downloaded the 833 pairs and read the counts the notebook printed: 0 corrupt, 0 duplicates, 750 train, 83 test. I wrote the 20 prompts in `eval/p2_prompts.json` before training. I trained both LoRA ranks, about 15 minutes each. I generated the 120 images, seeds 0 and 1, 512px, 30 steps. I ran KID, CLIP, and LPIPS and kept the saved table. I built the blinded page and rated the 25 pairs. Ritika and Rajesh rated them too. I looked at the 10 images the safety checker flagged. I chose rank 16 and did not train again. I uploaded the weights.

The 30 percent is the hard code and the hard writing, not a second training run. Cursor helped with the 90/10 split at seed 266 and the 512 crop, the LoRA targets `to_q`, `to_k`, `to_v`, and `to_out.0` with the VAE and text encoder frozen, the KID call of 20 in-domain images against 83 test images, and the memorization check of nearest training image by CLIP plus the tile distance. In the writing, Cursor was used on the architecture in `Part2/sneha_singh/results.md` and report section III, from Rombach et al., Hu et al., and Carlini et al., and on the results sentences that had rounded the notebook. Those sentences now say CLIP 0.3155, GPU-hours 0.2401 and 0.2399, and KID 0.06024 to 0.02314. The human win rate stays 0.387. `push_model_cards.py` only uploaded the two model-card READMEs after I logged in.

| Notebook section | About 70 percent, mine | About 30 percent, where I used Cursor |
| --- | --- | --- |
| 2.1 Data preparation | The download of 833 pairs and the printed counts: 0 corrupt, 0 duplicates, 750 train, 83 test | The split code: seed 266, 90/10, resize and center-crop to 512 |
| 2.3 Prompts, written before training | The 20 prompts in `eval/p2_prompts.json`: 10 in-domain, 5 out-of-domain, 5 compositional | None |
| 2.2 Fine-tuning | Both LoRA runs, about 15 minutes each, 0.2401 and 0.2399 GPU-hours | The adapter code: `to_q`, `to_k`, `to_v`, `to_out.0` only. VAE and text encoder frozen. Same steps, batch 4, learning rate 0.0001, seed 266 for both ranks |
| 2.3 Image generation | Seeds 0 and 1, 512px, 30 steps, guidance 7.5, 40 images per model | None |
| 2.4 Evaluation | The metric run and the saved table | The KID setup in code: 20 in-domain images against the 83 test images, plus the CLIP and LPIPS calls |
| Human evaluation | The blinded page, my 25 votes, Ritika's, and Rajesh's | None |
| Safety | I ran the checker on all 120 images and looked at the 10 flags | The memorization code: nearest training image by CLIP on 10 generations, then the tile distance on the 3 closest pairs |
| Failure modes and final model choice | I picked rank 16 and did not train again | The writing of that choice in `Part2/sneha_singh/results.md` and report section III |

Writing where Cursor was used. The architecture section, from Rombach et al., Hu et al., and Carlini et al. It says what this run trained. It is not a full diffusion retrain, and it is not Carlini's extraction attack. The results sentences were the other writing pass. The draft had rounded past the notebook: CLIP 0.316 and 0.24 GPU-hours. The file now has 0.3155, 0.2401, and 0.2399. KID 0.06024 to 0.02314 stays. The human win rate of 0.387 still reaches 0.5.

`Part2/sneha_singh/push_model_cards.py` is the same kind of help. After I logged in, it uploaded the two model-card READMEs only.

### Part 3: about 70 percent mine, about 30 percent where I used AI

Part 3 is counted on its own, the same way. About 70 percent is work I did. About 30 percent is where I used Cursor, including the bonus Space. The 30 percent does not include the graded run.

The 70 percent is the 26 Oct L4 run of `notebooks/p3_agents.ipynb`. I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list before that run. I loaded Qwen2.5-VL-3B, ran the 40 requests at 512px and 30 steps, and kept the traces. The routes matched 30 of 30. I opened the Colab demo. Ritika and I rated all 40 and agreed. The Results cell printed 6/15 without the critique loop, 4/15 with it, critic agreement 34/48, attack success 0/10, and median 15.1 s and 31.4 s. I did not type those numbers in, and I did not retune the critic after seeing 4/15.

The 30 percent is the hard code inside that notebook, the safety writing, and the bonus build. Cursor helped with the typed state and the JSON trace, the planner's early stop, the rule that text in an image is data, the prompt code that does not copy that text, the generation branch among img2img, InstructPix2Pix, image operations, and the Part 2 LoRA, the critic checks that stay only when they are grounded in the request, the revision cap of 2, and the CLIP cutoff of 0.26 fit on the 120 Part 2 images rather than on the 15 scenarios. The writing in that 30 percent is the safety section of `Part3/sneha_singh/results.md` and report section IV, from Greshake et al.

The bonus is inside this same 30 percent. The graded app already ran. The extra marks are that app on a free Hugging Face Space, and the Space would not build. Cursor was used on `Part3/sneha_singh/space/requirements.txt` (`diffusers==0.41.0` for Python 3.10, `pydantic==2.12.5`) and on `agents.py` (`import spaces` before torch, the literal `@spaces.GPU` decorator, the processor built without AutoProcessor, and the text-only path). I uploaded each file. "Draw a small fire fox creature in Pokemon style" returned an image. The Space is https://huggingface.co/spaces/Sneha-subbu/data266-lab2-p3-agents. I did not buy a larger quota.

| Notebook section | About 70 percent, mine | About 30 percent, where I used Cursor |
| --- | --- | --- |
| Shared state and trace | The saved JSON traces from the L4 run | The typed state and the one-trace-per-request code |
| Vision-language model | The L4 load of Qwen2.5-VL-3B | None in the graded notebook. The Space load is under the bonus, below |
| Planner | The 30 of 30 routes that matched | The route and refuse code, including the early stop for an unsafe or broken request |
| Vision Understanding | The captions from the run | The rule in code that text in an image is data, not an instruction |
| Prompt Engineering | The prompts the run actually sent | The code that does not copy image text into the generation prompt |
| Image Generation | The 512px, 30-step images | The branch: img2img, InstructPix2Pix, image operations, or the Part 2 rank-16 LoRA |
| Critique | The 15 revisions and the 4/15 result | The check code: grounded in the request, accept only when every check passes |
| Orchestrator | I started the run and did not retune it after 4/15 | The revision cap of 2, and the path that stops after vision when no image is asked for |
| Demo app, in the notebook | I opened it on Colab | None. The public Space is the bonus |
| Rating page | Ritika and I rated all 40 and agreed | None |
| CLIP threshold calibration | The printed means, 0.316 matched and 0.179 shuffled | The cutoff code: 0.26, fit on the 120 Part 2 images, not on the 15 scenarios |
| Evaluation, IMG07, summary, Results | The Results cell. I did not type the scores in | None |
| Findings, failure modes, safety summary | The 4/15 and 6/15 stay as printed | The safety writing in `Part3/sneha_singh/results.md` and report section IV, from Greshake et al. |

## Ritika Neema

### Part 1

To be filled by Ritika: AI tools used, what for, and what she reviewed. Nothing recorded yet.

### Part 2

To be filled by Ritika.

### Part 3

To be filled by Ritika.
