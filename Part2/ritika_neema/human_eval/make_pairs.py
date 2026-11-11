"""Rebuild the 25 blinded pairs with pair-sampling seed 267 (same code as Cell 9 of
notebooks/ritika_neema/p2_finetune.ipynb, which used seed 266 for the first set whose key was opened).

Reads the generated images from the Hugging Face dataset neema30/data266-lab2-p2-ritika-images, so no GPU is
needed. Writes, in the folder given by --out (default ~/Downloads/p2_ritika_rating):
  pair_01.png ... pair_25.png, ratings_template.csv and rate.html (one self-contained page to send to raters).
The answer key goes to Part2/ritika_neema/human_eval/key_hidden.csv and is never printed. Do not open it
until every rater is done.

Usage (repo root):
  python Part2/ritika_neema/human_eval/make_pairs.py
"""
import argparse, base64, html, io, json, os

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "neema30/data266-lab2-p2-ritika-images"
PAIR_SEED = 267
FINAL = "lora_r8"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/Downloads/p2_ritika_rating"))
out = ap.parse_args().out
os.makedirs(out, exist_ok=True)

PROMPTS = json.load(open("eval/p2_prompts.json"))
src = snapshot_download(REPO, repo_type="dataset", allow_patterns=["images/base/*", f"images/{FINAL}/*"])
GEN = {m: {(p["id"], s): Image.open(f"{src}/images/{m}/p{p['id']:02d}_s{s}.png").convert("RGB")
           for p in PROMPTS for s in (0, 1)} for m in ("base", FINAL)}

# Cell 9, with the pair-sampling seed changed to 267.
rng = np.random.default_rng(PAIR_SEED)
cands = [(p["id"], s) for p in PROMPTS for s in (0, 1)]
picked = [cands[i] for i in rng.choice(len(cands), 25, replace=False)]
key, cards = [], []
for n, (pid, s) in enumerate(picked, 1):
    left_is_final = bool(rng.integers(2))
    L, R = (GEN[FINAL], GEN["base"]) if left_is_final else (GEN["base"], GEN[FINAL])
    im = Image.new("RGB", (1024, 512), "white"); im.paste(L[(pid, s)], (0, 0)); im.paste(R[(pid, s)], (512, 0))
    im.save(f"{out}/pair_{n:02d}.png")
    prompt = PROMPTS[pid]["prompt"]
    key.append(dict(pair=n, prompt=prompt, left="final" if left_is_final else "base"))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    cards.append((n, prompt, base64.b64encode(buf.getvalue()).decode()))
pd.DataFrame(key).to_csv(f"{HERE}/key_hidden.csv", index=False)
pd.DataFrame({"pair": range(1, 26), "prompt": [k["prompt"] for k in key], "answer": ""}).to_csv(
    f"{out}/ratings_template.csv", index=False)

rows = "".join(f"""<section><h2>Pair {n}</h2><p class="pr">{html.escape(p)}</p>
<img src="data:image/jpeg;base64,{b}" alt="pair {n}"><div class="opts">
<label><input type="radio" name="p{n}" value="L"> Left</label><label><input type="radio" name="p{n}" value="T"> Tie</label>
<label><input type="radio" name="p{n}" value="R"> Right</label></div></section>""" for n, p, b in cards)
prompts = json.dumps([p for _, p, _ in cards])
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pokemon Image Rating</title>
<style>:root{{--bg:#fff;--fg:#1a1a1a;--line:#ddd;--accent:#2b5fd9}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151515;--fg:#eee;--line:#333;--accent:#7aa2ff}}}}
body{{background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif;margin:0 auto;max-width:1060px;padding:16px}}
section{{border-top:1px solid var(--line);padding:12px 0}}h2{{margin:0 0 4px;font-size:16px}}.pr{{margin:0 0 8px}}
img{{width:100%;height:auto;display:block}}.opts{{display:flex;gap:24px;padding:8px 0}}
button{{font:inherit;padding:8px 16px;background:var(--accent);color:#fff;border:0;border-radius:6px}}
input[type=text]{{font:inherit;padding:6px}}</style></head><body>
<h1>Which image fits the prompt better, in Pokemon style?</h1>
<p>For each of the 25 pairs, pick <b>Left</b>, <b>Right</b>, or <b>Tie</b>: the image that better matches the prompt
and looks more like a clean Pokemon illustration. The two models are shown in random order. Rate on your own.</p>
<p>Your name: <input type="text" id="name" placeholder="e.g. Ritika"></p>
{rows}
<p><button id="save">Download my ratings</button> <span id="msg"></span></p>
<script>
var PROMPTS = {prompts};
document.getElementById("save").onclick = function () {{
  var name = document.getElementById("name").value.trim().replace(/[^A-Za-z0-9_-]/g, "");
  if (!name) {{ document.getElementById("msg").textContent = "Enter your name first."; return; }}
  var lines = ["pair,prompt,answer"], missing = [];
  for (var i = 1; i <= 25; i++) {{
    var c = document.querySelector('input[name="p' + i + '"]:checked');
    if (!c) missing.push(i);
    lines.push(i + ',"' + PROMPTS[i - 1].replace(/"/g, '""') + '",' + (c ? c.value : ""));
  }}
  if (missing.length) {{ document.getElementById("msg").textContent = "Not rated yet: " + missing.join(", "); return; }}
  var a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv"}}));
  a.download = "ratings_" + name + ".csv"; a.click();
  document.getElementById("msg").textContent = "Saved ratings_" + name + ".csv";
}};
</script></body></html>"""
open(f"{out}/rate.html", "w").write(page)
print(f"wrote 25 pairs, ratings_template.csv and rate.html to {out}; key saved (not shown)")
