"""Build index.html, a page for rating my Part 3 run: one card per request with the input, every attempt,
the agents' outputs and the frozen rubric for that case.

Usage (repo root):
  python Part3/ritika_neema/human_eval/make_gallery.py
Then open Part3/ritika_neema/human_eval/index.html and fill p3_rating_sheet.csv (see README.md).
"""
import glob, html, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = {c["id"]: c for c in json.load(open("eval/p3_scenarios.json"))["scenarios"]}
RED = {c["id"]: c for c in json.load(open("eval/p3_redteam.json"))["cases"]}


def esc(x):
    return html.escape(str(x))


def strip_raw(d):
    return {k: v for k, v in d.items() if k != "raw"} if isinstance(d, dict) else d


def card(t):
    cid = t["request_id"].split("_")[0]
    mode = "with critique" if t["use_critique"] else "no critique"
    case = SCEN.get(cid) or RED.get(cid)
    good = case.get("pass_requires") or case.get("defended_if") or []
    bad = case.get("fail_if") or case.get("attack_succeeds_if") or []
    inp = os.path.basename(t["image_path"]) if t.get("image_path") else None
    imgs = ""
    if inp and os.path.exists(os.path.join(HERE, "inputs", inp)):
        imgs += f'<figure><img src="inputs/{esc(inp)}"><figcaption>input {esc(inp)}</figcaption></figure>'
    for i, p in enumerate(sorted(glob.glob(os.path.join(HERE, "images", f"{t['request_id']}_try*.png")))):
        c = t["critiques"][i] if i < len(t["critiques"]) else None
        note = ("critic: accept" if c["accept"] else "critic: revise") if c else ""
        imgs += f'<figure><img src="images/{esc(os.path.basename(p))}"><figcaption>attempt {i} {note}</figcaption></figure>'
    agents = {"planner": strip_raw(t["plan"]), "vision": strip_raw(t["vision"]), "prompt": strip_raw(t["prompt"]),
              "generation": strip_raw(t["generation"]), "critique": [strip_raw(c) for c in t["critiques"]]}
    return f"""<section><h2>{esc(cid)} &middot; {esc(mode)} &middot; <code>{esc(t['status'])}</code></h2>
<p><b>Request:</b> {esc(t['user_request'])}<br><b>Input:</b> {esc(case['input'])}<br>
<b>Final message:</b> {esc(t['final_message'])}</p>
<div class="imgs">{imgs or '<i>no image</i>'}</div>
<div class="rub"><div><b>Pass / defended if</b><ul>{''.join(f'<li>{esc(x)}</li>' for x in good)}</ul></div>
<div><b>Fail / attack succeeds if</b><ul>{''.join(f'<li>{esc(x)}</li>' for x in bad)}</ul></div></div>
<details><summary>agent outputs</summary><pre>{esc(json.dumps(agents, indent=1, default=str))}</pre></details></section>"""


traces = [json.load(open(p)) for p in sorted(glob.glob("traces/ritika_neema/[SR]*_*crit.json"))]
traces.sort(key=lambda t: (t["request_id"][0] != "S", t["request_id"].split("_")[0], t["use_critique"]))
page = f"""<!doctype html><meta charset="utf-8"><title>Part 3 rating - Ritika</title>
<style>body{{font:14px system-ui;margin:16px;max-width:1100px}}section{{border-top:2px solid #ccc;padding:8px 0}}
.imgs{{display:flex;gap:8px;flex-wrap:wrap}}img{{width:240px}}figure{{margin:0}}figcaption{{font-size:12px}}
.rub{{display:flex;gap:24px}}pre{{white-space:pre-wrap;font-size:12px}}</style>
<h1>Part 3 rating - Ritika's run ({len(traces)} requests)</h1>
<p>Fill <code>p3_rating_sheet.csv</code>: pass/fail per row from the rubric, accept/revise for the final image,
yes/no for "attack succeeded" on R rows. Rate on your own before comparing.</p>
{''.join(card(t) for t in traces)}"""
open(os.path.join(HERE, "index.html"), "w").write(page)
print("wrote", os.path.join(HERE, "index.html"), len(traces), "cards")
