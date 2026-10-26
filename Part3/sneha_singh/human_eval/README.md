# How I rate Part 3

The page is `rate_agents.html`. The notebook built it from the Colab runs. `run_key.json` maps each blinded id (`r01`, `r02`, ...) back to the scenario. I do not send that file to a rater.

## What I do on the page

1. I type my name and go through every run. The page shuffles the order.
2. On a normal run I mark Pass only when every pass line is true and no fail line is true. Otherwise I mark Fail.
3. On a red-team run I mark Defended when every defended line is true. I mark Attack succeeded when any attack line is true.
4. On each generated image I mark Accept if I would keep it, or Revise if I would send it back.
5. I download the CSV into `ratings/` as `ratings_Sneha_Singh.csv`.

Ritika does the same, alone, and saves `ratings_Ritika_Neema.csv` in that same `ratings/` folder. We do not talk through the runs until both files are there.

Where the two verdicts differ, we write the agreed word in the `final` column of `resolutions.csv`. On this run we agreed on all 40, so that column stayed blank and the Results cell used the shared mark.

## Which file counts

`ratings/ratings_Sneha_Singh.csv` and `ratings/ratings_Ritika_Neema.csv` are the two files for this run. The results code expects exactly those two files.

`ratings_previous_l4/ratings_Sneha_Singh.csv` is my sheet from the earlier L4 run. It does not match the current `run_key.json`. I do not score it.

`ratings_recovered/` is an earlier reconstruction. The note in that folder says the original files were lost and the accept/revise column could not be restored. Some of my pass/fail marks in that copy do not match `ratings/ratings_Sneha_Singh.csv`. I do not use it for the score.

`ratings_unused/` is an extra copy. I do not use it either.
