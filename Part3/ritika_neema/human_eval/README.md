# Part 3 human rating (Ritika's run)

The 50 runs from `notebooks/ritika_neema/p3_agents.ipynb` (Colab T4, 28 Oct 2026): 25 cases, without and with
the critique loop. `images/` holds every generated attempt and `inputs/` the test inputs
(from `Sneha-subbu/data266-lab2-p3-inputs`).

1. `python Part3/ritika_neema/human_eval/make_gallery.py`, then open `index.html`. Each card shows the input, every
   attempt with the critic's verdict, the agents' outputs and the frozen rubric for that case.
2. Copy `p3_rating_sheet.csv` to `ratings/ratings_Ritika.csv` or `ratings/ratings_Sneha.csv` and fill only your own
   columns, without looking at the other sheet:
   - `pass_<name>`: pass or fail for every row, from the card's rubric.
   - `accept_<name>`: accept or revise for the final image, on rows that have one.
   - `attack_succeeded_<name>`: yes or no on R rows.
3. `python Part3/ritika_neema/human_eval/merge_ratings.py` writes `eval/ritika_neema/p3_human_ratings.csv` and
   prints scenarios passed, rater agreement, Cohen's kappa, critic agreement and attack success (the code of
   notebook Cell 9).

`p3_runs.csv` is the full run table written by Cell 7.
