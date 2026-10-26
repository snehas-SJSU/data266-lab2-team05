# AI usage

Each member records her own AI tools and what they were used for; do not edit the other member's section.

## Sneha Singh

### Part 1

I used an assistant to read the three Part 1 papers (Lewis et al. on retrieval, Faysse et al. on document pages, Greshake et al. on indirect prompt injection) and to check what the lab actually requires from them. I used an assistant on the two spots the lab makes easy to get wrong. Appendix C is low trust: the text has false claims, and Figure C-1 has an instruction aimed at the model. That page has to stay out of retrieval and out of the image passed to the vision model, and a question about the appendix itself has to be answered from the manifest, not from the appendix. The other spot is conversation memory. A user can add or change a fact inside one chat, and that change has to die with the chat. It must not be written back into the three PDFs or the index.

### Part 2

I used an assistant to read the three Part 2 papers (Rombach et al. on latent diffusion, Hu et al. on LoRA, Carlini et al. on training-data extraction) and to separate what the lab asks for from what the papers do. I used an assistant on the LoRA setup and the memorization check. The lab wants a small adaptation, not a full retrain, so the adapter sits only on the UNet attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`). The VAE and the text encoder stay frozen, and rank 4 and rank 16 share the same data, steps, batch, learning rate, and seed. For memorization, Carlini et al. draw hundreds of images and look for a pixel copy. The lab asks for a smaller check: the nearest training image by CLIP, then the closest pairs. I also used help on the tile distance for those pairs, and on reading the safety-checker flags, which fire on ordinary cartoon creatures and are false positives.

### Part 3

I used an assistant to read the Part 3 safety paper again (Greshake et al.) and to check how an instruction hidden in an image should be treated. I used an assistant on the two safety spots that fail if they are written loosely. Text inside an uploaded image, including a line that says to skip the checks or to print a hidden instruction, has to be reported as content. The prompt agent must not copy it into the generation prompt. The critic was the other spot. A 3B model is too generous when asked for one overall pass, so the check is split into separate yes/no questions taken from the user's request. The CLIP cutoff of 0.26 was fit on the Part 2 images, not on the 15 scenarios, so the threshold is not tuned on the test set.

## Ritika Neema

### Part 1

To be filled by Ritika: AI tools used, what for, and what she reviewed. Nothing recorded yet.

### Part 2

To be filled by Ritika. (She rated 25 image pairs in the human evaluation; record any AI use for her Part 2 work here.)

### Part 3

To be filled by Ritika.
