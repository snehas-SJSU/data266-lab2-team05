"""Merge the two rating sheets and print the human results (same code as Cell 9 of
notebooks/ritika_neema/p3_agents.ipynb, so Colab is not needed again).

Each rater fills their own copy of p3_rating_sheet.csv, only in their own columns (pass_<name>, accept_<name>,
attack_succeeded_<name>), and saves it as ratings/ratings_Ritika.csv or ratings/ratings_Sneha.csv.

Usage (repo root):
  python Part3/ritika_neema/human_eval/merge_ratings.py
Writes eval/ritika_neema/p3_human_ratings.csv and prints the table for the README.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "eval/ritika_neema/p3_human_ratings.csv"
KEYS = ["id", "mode"]

base = pd.read_csv(os.path.join(HERE, "p3_rating_sheet.csv"), dtype=str).fillna("")
for name in ("ritika", "sneha"):
    path = os.path.join(HERE, "ratings", f"ratings_{name.capitalize()}.csv")
    r = pd.read_csv(path, dtype=str).fillna("")
    cols = [c for c in r.columns if c.endswith("_" + name)]
    base = base.drop(columns=cols).merge(r[KEYS + cols], on=KEYS, how="left").fillna("")
    s_rows = base[base.id.str.startswith("S")]
    if (s_rows[f"pass_{name}"].str.strip() == "").any():
        raise SystemExit(f"{path}: pass_{name} is empty on some S rows")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
base.to_csv(OUT, index=False)
print("wrote", OUT)

# Cell 9 of the notebook, plus Cohen's kappa for the pass/fail marks.
h = base
yes = lambda s: s.str.strip().str.lower().isin(["pass", "yes", "y", "1", "true", "accept"])


def kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


res = []
for mode, d in h.groupby("mode"):
    s, rt = d[d.id.str.startswith("S")], d[d.id.str.startswith("R")]
    both = yes(s.pass_ritika) & yes(s.pass_sneha)
    agree = (yes(s.pass_ritika) == yes(s.pass_sneha)).mean()
    g = d[d.critic_final.isin(["accept", "revise"]) & (d.accept_ritika != "")]
    crit_agree = ((g.critic_final == "accept") == yes(g.accept_ritika)).mean() if len(g) else float("nan")
    attack = (yes(rt.attack_succeeded_ritika) | yes(rt.attack_succeeded_sneha)).sum()
    res.append({"system": mode, "scenarios passed (both raters) / 15": int(both.sum()),
                "rater agreement": round(agree, 3), "kappa": round(kappa(yes(s.pass_ritika), yes(s.pass_sneha)), 3),
                "critic agreement": round(crit_agree, 3), "attack success / 10": int(attack)})
print(pd.DataFrame(res).to_string(index=False))
