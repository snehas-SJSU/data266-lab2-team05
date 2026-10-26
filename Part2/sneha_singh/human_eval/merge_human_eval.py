"""Merge rater CSVs (ratings_*.csv from rate_images.html) into the Part 2 human-evaluation result.

Usage (repo root):  .venv/bin/python Part2/sneha_singh/human_eval/merge_human_eval.py Part2/sneha_singh/human_eval/ratings lora_r16
  arg1: folder holding the rater CSV files   arg2: name of the final run (e.g. lora_r16)
Writes eval/p2_human_review.csv (rater columns filled) and prints win rate, per-rater rates, agreement.
"""
import glob, math, sys
import pandas as pd

folder, final = sys.argv[1], sys.argv[2]
key = pd.read_csv("Part2/sneha_singh/human_eval/answer_key.csv").set_index("pair")
files = sorted(glob.glob(f"{folder}/ratings_*.csv"))
assert len(files) >= 3, f"need 3+ rater files, found {len(files)}"
rows = pd.concat([pd.read_csv(f) for f in files])
rows = rows[rows.choice.isin(["L", "R"])]
rows["winner"] = [key.loc[p, "left" if c == "L" else "right"] for p, c in zip(rows.pair, rows.choice)]
rows["final_won"] = rows.winner == final
n, w = len(rows), int(rows.final_won.sum())
p = w / n; z = 1.96                                   # Wilson 95% interval
c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
print({"raters": rows.rater.nunique(), "votes": n, "final": final, "win_rate": round(p, 3), "ci95": [round(c-h, 3), round(c+h, 3)]})
print(rows.groupby("rater").final_won.mean().round(3).to_dict())
pv = rows.pivot(index="pair", columns="rater", values="final_won")
print({"unanimous_pairs": int((pv.nunique(axis=1) == 1).sum()), "of": len(pv)})
rev = pd.read_csv("eval/p2_human_review.csv")
import json
ptype = {x["prompt"]: x["type"] for x in json.load(open("eval/p2_prompts.json"))}
rows["type"] = rows.pair.map(rev.set_index("pair").prompt).map(ptype)
bt = rows.groupby("type").final_won.agg(["mean", "count"]); bt["mean"] = bt["mean"].round(3)
print("win rate by prompt type (votes):", bt.to_dict("index"))
for i, (r, g) in enumerate(rows.groupby("rater")):
    if i < 3: rev[f"rater{i+1}"] = rev.pair.map(g.set_index("pair").choice)
rev.to_csv("eval/p2_human_review.csv", index=False)
print("wrote eval/p2_human_review.csv")
