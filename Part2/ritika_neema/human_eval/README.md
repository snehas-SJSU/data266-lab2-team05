# Part 2 blinded human evaluation (Ritika)

25 pairs, LoRA r8 against the base model, left/right shuffled, made by Cell 9 of
`notebooks/ritika_neema/p2_finetune.ipynb`. Each rater answers L, R or T (tie) for the image that better matches
the prompt in Pokemon style.

1. Send each rater the 25 `pair_XX.png` images and `ratings_template.csv`. Never send `key_hidden.csv`.
2. Save each filled sheet here as `ratings/ratings_<Name>.csv` (one answer column per sheet).
3. `python Part2/ritika_neema/human_eval/merge_ratings.py` writes `eval/ritika_neema/p2_human_ratings.csv`.
4. Re-run Cell 10 of the notebook for the win rate.
5. Commit `key_hidden.csv` here only after all ratings are in.

The first set of pairs was re-shuffled (pair-sampling seed 267) because its answer key was opened before any
rating. Only the re-shuffled set is rated.
