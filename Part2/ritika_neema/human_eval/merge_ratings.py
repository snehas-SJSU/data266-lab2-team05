"""Merge each rater's sheet into the file the notebook's win-rate cell (Cell 10) reads.

Each rater fills a copy of ratings_template.csv (from notebooks/ritika_neema/p2_finetune.ipynb, Cell 9) with
L, R or T in one column, and saves it as human_eval/ratings/ratings_<Name>.csv.

Usage (repo root):
  python Part2/ritika_neema/human_eval/merge_ratings.py
Writes eval/ritika_neema/p2_human_ratings.csv with one column per rater (rater_<Name>).
"""
import glob, os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "eval/ritika_neema/p2_human_ratings.csv"

merged = None
for path in sorted(glob.glob(os.path.join(HERE, "ratings", "ratings_*.csv"))):
    name = os.path.basename(path)[len("ratings_"):-len(".csv")]
    df = pd.read_csv(path, dtype=str).fillna("")
    filled = [c for c in df.columns if c not in ("pair", "prompt") and df[c].str.strip().ne("").any()]
    if len(filled) != 1:
        raise SystemExit(f"{path}: expected exactly one filled answer column, found {filled}")
    answers = df[filled[0]].str.strip().str.upper()
    bad = sorted(set(answers) - {"L", "R", "T"})
    if bad or len(df) != 25:
        raise SystemExit(f"{path}: need 25 rows of L, R or T; found {len(df)} rows and values {bad}")
    col = df[["pair", "prompt"]].assign(**{f"rater_{name}": answers})
    merged = col if merged is None else merged.merge(col, on=["pair", "prompt"])

if merged is None:
    raise SystemExit("no ratings found in human_eval/ratings/")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
merged.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(merged)} pairs x {merged.shape[1] - 2} raters")
