# Contributions

Pair_Programming_Team_05: Sneha Singh, Ritika Neema

## Meetings

| | Date | Topic | Link |
| --- | --- | --- | --- |
| 1 | | | |
| 2 | | | |

## Sneha Singh

### Part 1

I built `notebooks/p1_rag.ipynb` and ran it locally. It parses the three PDFs, keeps Appendix C out of retrieval, answers from text, figures, and tables, and keeps conversation edits out of the documents. I compared a dense baseline with the hybrid system on the 25-question golden set and wrote the scores in `Part1/sneha_singh/results.md`. The golden set still needs both of us to review it. The human ratings and the timed Kaggle run are not done.

### Part 2

I built and ran `notebooks/p2_finetune.ipynb` on a Colab L4: data prep, LoRA rank 4 and rank 16, and the 120 images. I scored KID, CLIP, and LPIPS, ran the safety checker and the memorization check, and uploaded both adapters and the images to Hugging Face. I made the blinded rating page and rated the 25 pairs. My notes are in `Part2/sneha_singh/results.md` and section 2.A of `report/report.md`. How to run it is in `README.md`. The prompt file is `eval/p2_prompts.json`.

### Part 3

I wrote the 15 scenarios, the 10 red-team cases, and the 25-input list before the runs. I built the five agents, the orchestrator, the Gradio app, the rating page, and the results code in `notebooks/p3_agents.ipynb`. I smoke-tested them on my Mac, then ran the 15 scenarios and 10 red-team cases on an L4. That run is in the notebook. The two rating files are not both in place, the agreed finals are blank, and the test inputs are not on Hugging Face yet. Notes are in `Part3/sneha_singh/results.md`.

## Ritika Neema

### Part 1

To be filled by Ritika (`Part1/ritika_neema/`).

### Part 2

Rated the 25 blinded image pairs. No other Part 2 work is recorded yet (`Part2/ritika_neema/`).

### Part 3

To be filled by Ritika (`Part3/ritika_neema/`).
