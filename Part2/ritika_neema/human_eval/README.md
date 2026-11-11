# Part 2 blinded human evaluation (Ritika)

25 pairs, LoRA r8 against the base model, left/right shuffled. Each rater answers L, R or T (tie) for the image
that better matches the prompt in Pokemon style.

The first set of pairs (Cell 9 of `notebooks/ritika_neema/p2_finetune.ipynb`, pair-sampling seed 266) was
not rated because its answer key was opened. `make_pairs.py` rebuilds the pairs with the same code and
pair-sampling seed 267, from the images in the Hugging Face dataset `neema30/data266-lab2-p2-ritika-images`.
With seed 266 it reproduces the original key exactly. Only the seed-267 set is rated.

1. `python Part2/ritika_neema/human_eval/make_pairs.py` writes `pair_01.png` ... `pair_25.png`,
   `ratings_template.csv` and `rate.html` to `~/Downloads/p2_ritika_rating/`, and the answer key to
   `key_hidden.csv` here. Nobody opens the key until all ratings are in.
2. Send each rater `rate.html` (one file, images included). They pick Left, Right or Tie for every pair and
   click "Download my ratings", which saves `ratings_<Name>.csv`.
3. Save each sheet here as `ratings/ratings_<Name>.csv`.
4. `python Part2/ritika_neema/human_eval/merge_ratings.py` writes `eval/ritika_neema/p2_human_ratings.csv`
   and prints the win rate with the code of Cell 10.
5. Commit `key_hidden.csv` only after all ratings are in.
