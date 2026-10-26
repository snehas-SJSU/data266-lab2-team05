# AI usage

Each member records her own AI tools and what they were used for; do not edit the other member's section.

## Sneha Singh

An AI assistant helped me design. I read the papers. For each part the help was the design of the hard piece, and on top of that a brainstorm of what to take from the papers.

### Part 1

AI helped me design the retrieval and the grounded answer in `notebooks/p1_rag.ipynb`.

Retrieval. `HybridRetriever.retrieve` mixes a dense MiniLM score with BM25. `PassageReranker.score` reorders the passages for the question. `expand_context` brings in the chunks that sit next to a hit, so the answer is not stuck on one short passage. `retrieve_dense` stays beside it as the dense-only baseline.

Figures and the answer. `OllamaVLM.describe` turns a figure into words with Qwen2.5-VL. `OllamaReasoner.generate_grounded` writes the answer from those words and from the retrieved text. Qwen2.5 7B does not look at the picture again.

That design is what went into `Part1/sneha_singh/results.md` and the Part 1 write-up in `report/report.md`.

On top of that design, AI helped me brainstorm what to take from the papers.

Lewis et al. The order is retrieve, then generate. Their retriever is a dense encoder over Wikipedia. I kept that order. The retriever here is the hybrid one above.

Faysse et al. (ColPali). A page should be searchable as an image, by embedding patches with a vision model. I did not train that model. I kept a smaller version: describe the figure, then retrieve the description.

Greshake et al. An instruction can hide in a page or a picture. I turned that into a rule for this corpus. Retrieved text and the pixels of a figure are data. `describe` can say what a chart shows. `generate_grounded` does not follow a line inside it.

### Part 2

AI helped me design which layers the adapter trains in `notebooks/p2_finetune.ipynb`.

`train_lora` puts the adapter on `to_q`, `to_k`, `to_v`, and `to_out.0`. The VAE, the text encoder, and the original UNet weights stay frozen. The loss is the mean squared error on the noise. `lora_alpha` equals the rank.

That design is what went into `Part2/sneha_singh/results.md` and the Part 2 write-up in `report/report.md`.

AI also helped me fix the results sentences. An early paragraph rounded rank 16 CLIP to 0.316 and both runs to 0.24 GPU-hours. The notebook prints are CLIP 0.3155, and GPU-hours 0.2401 and 0.2399. `Part2/sneha_singh/push_model_cards.py` uploaded only the two model-card READMEs. The weights were not part of that upload.

On top of that work, AI helped me brainstorm what to take from the papers.

Rombach et al. The picture goes through a VAE, the UNet predicts the noise, and the caption enters through cross-attention. Stable Diffusion 1.5 is already that model, so the design leaves it frozen and trains only the attention adapters.

Hu et al. A small low-rank piece carries most of the change. That is why rank 4 and rank 16 share the 750 images, 1500 steps, batch size 4, learning rate 0.0001, and seed 266. Rank 4 trains 797,184 parameters. Rank 16 trains 3,188,736. KID moved from 0.06024 to 0.02314, and the gap between the ranks sits inside the noise. A bigger rank is not always better.

Carlini et al. They draw hundreds of images per prompt and call a near-copy something near 0.15 on their scale. I kept the question and used a smaller check. `cemb` finds the nearest training image by CLIP on 10 generations from one seed. `tile_dist` measures the three closest pairs. The largest per-tile RMS distance over 16 tiles came out 0.584 to 0.801. A CLIP score of 0.789 to 0.938 looks high next to their 0.15. The three closest pairs are the same kind of creature, with a different pose and different colors.

### Part 3

AI helped me design the prompt check, the critique check, and the revision loop in `notebooks/p3_agents.ipynb`.

`vision` writes down any text it sees in the picture and stores it as data. `prompt_agent`, with `_overlap_injection`, keeps that text out of the prompt that goes to the image model, even when the picture says to skip the checks. `_grounded_in_request` keeps a phrase only when the words are already in the request.

`critique` uses `clip_sim`. A check stays only when `_grounded_in_request` says it belongs to the request, so a phrase copied off the image cannot make the critic accept or revise. The image is accepted only when every remaining check passes.

`run_request` wraps those checks. The revision cap is 2. A caption or a question stops after `vision` and does not go on to generate.

That design is what went into `Part3/sneha_singh/results.md` and the Part 3 write-up in `report/report.md`.

On top of that design, AI helped me brainstorm what to take from the papers.

Yao et al. (ReAct). The model reasons about the request and then picks an action. I kept the reason step in `planner`. The later agents carry one fixed route. An unsafe or vague request stops at the planner.

Madaan et al. (Self-Refine). The loop is generate, critique, revise, with no extra training. I kept one concrete fix from `critique` and a stop at 2 revisions.

Brooks et al. (InstructPix2Pix). A higher image guidance stays closer to the source photo. I use that on the edit route. Guidance goes down when a revision still needs to change more, and goes up when a revision has drifted.

Greshake et al. An instruction can hide in a picture. That is the rule already in `vision` and `prompt_agent`. On the 10 red-team cases, R01 and R04 came back as captions and did not follow the text in the picture. R05 through R08 were refused before any image was made. Attack success was 0/10 on this run. That is the check in this lab. It is not the paper's own attack set.

AI also helped me brainstorm the Part 3 bonus, the same app on a free Hugging Face Space with ZeroGPU hardware `zero-a10g`, in `Part3/sneha_singh/space/`.

The Space image is Python 3.10, so `diffusers==0.42.0` has no build there. The log ended with `No matching distribution found for diffusers==0.42.0`. The pin that stayed is `diffusers==0.41.0`. `pydantic==2.12.5` stayed because gradio's MCP extra rejects 2.14.1.

`VLM._load_hf` builds the processor without AutoProcessor, because torchvision is not in that image. AutoProcessor raised `TypeError: argument of type 'NoneType' is not iterable`.

`app_fn` imports `spaces` first, before torch. The GPU hook is the literal `@spaces.GPU` decorator. A request with no image does not go through the image processor. An earlier upload failed because no `@spaces.GPU` function was found at startup. The one after that failed because CUDA was already initialized.

## Ritika Neema

### Part 1

To be filled by Ritika: AI tools used, what for, and what she reviewed. Nothing recorded yet.

### Part 2

To be filled by Ritika.

### Part 3

To be filled by Ritika.
