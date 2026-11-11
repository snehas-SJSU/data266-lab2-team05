"""Build index.html, a page for rating my Part 3 run: one card per request with the input, every attempt,
the agents' outputs, the frozen rubric for that case and the rating buttons.

Usage (repo root):
  python Part3/ritika_neema/human_eval/make_gallery.py
Writes index.html here (images linked from images/ and inputs/) and, if Pillow is installed, a self-contained
copy with the images embedded at ~/Downloads/p3_ritika_rating/rate.html to send to the other rater.
Each rater picks their name, rates every card and clicks "Download my ratings", which saves
ratings_<Name>.csv in the format merge_ratings.py reads.
"""
import base64, csv, glob, html, io, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = {c["id"]: c for c in json.load(open("eval/p3_scenarios.json"))["scenarios"]}
RED = {c["id"]: c for c in json.load(open("eval/p3_redteam.json"))["cases"]}
SHEET = {(r["id"], r["mode"]): r for r in csv.DictReader(open(os.path.join(HERE, "p3_rating_sheet.csv")))}
EMBED_DIR = os.path.expanduser("~/Downloads/p3_ritika_rating")


def esc(x):
    return html.escape(str(x))


def strip_raw(d):
    return {k: v for k, v in d.items() if k != "raw"} if isinstance(d, dict) else d


def img_src(rel, embed):
    if not embed:
        return esc(rel)
    from PIL import Image, UnidentifiedImageError
    try:
        im = Image.open(os.path.join(HERE, rel)).convert("RGB")
    except UnidentifiedImageError:  # a deliberately broken test input
        return ""
    im.thumbnail((512, 512))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def radios(name, opts):
    return "".join(f'<label><input type="radio" name="{name}" value="{v}"> {lab}</label>' for v, lab in opts)


def card(t, embed):
    cid = t["request_id"].split("_")[0]
    mode = "with critique" if t["use_critique"] else "no critique"
    row = SHEET[(cid, mode)]
    case = SCEN.get(cid) or RED.get(cid)
    good = case.get("pass_requires") or case.get("defended_if") or []
    bad = case.get("fail_if") or case.get("attack_succeeds_if") or []
    inp = os.path.basename(t["image_path"]) if t.get("image_path") else None
    imgs = ""
    if inp and os.path.exists(os.path.join(HERE, "inputs", inp)):
        imgs += f'<figure><img src="{img_src("inputs/" + inp, embed)}"><figcaption>input {esc(inp)}</figcaption></figure>'
    for i, p in enumerate(sorted(glob.glob(os.path.join(HERE, "images", f"{t['request_id']}_try*.png")))):
        c = t["critiques"][i] if i < len(t["critiques"]) else None
        note = ("critic: accept" if c["accept"] else "critic: revise") if c else ""
        imgs += (f'<figure><img src="{img_src("images/" + os.path.basename(p), embed)}">'
                 f'<figcaption>attempt {i} {note}</figcaption></figure>')
    agents = {"planner": strip_raw(t["plan"]), "vision": strip_raw(t["vision"]), "prompt": strip_raw(t["prompt"]),
              "generation": strip_raw(t["generation"]), "critique": [strip_raw(c) for c in t["critiques"]]}
    key = f"{cid}|{mode}"
    ctl = f'<div class="q"><b>Pass?</b> {radios("pass|" + key, [("pass", "Pass"), ("fail", "Fail")])}</div>'
    if row["output"]:
        ctl += f'<div class="q"><b>Final image:</b> {radios("accept|" + key, [("accept", "Accept"), ("revise", "Revise")])}</div>'
    if cid.startswith("R"):
        ctl += f'<div class="q"><b>Attack succeeded?</b> {radios("attack|" + key, [("yes", "Yes"), ("no", "No")])}</div>'
    return f"""<section data-key="{esc(key)}"><h2>{esc(cid)} &middot; {esc(mode)} &middot; <code>{esc(t['status'])}</code></h2>
<p><b>Request:</b> {esc(t['user_request'])}<br><b>Input:</b> {esc(case['input'])}<br>
<b>Final message:</b> {esc(t['final_message'])}</p>
<div class="imgs">{imgs or '<i>no image</i>'}</div>
<div class="rub"><div><b>Pass / defended if</b><ul>{''.join(f'<li>{esc(x)}</li>' for x in good)}</ul></div>
<div><b>Fail / attack succeeds if</b><ul>{''.join(f'<li>{esc(x)}</li>' for x in bad)}</ul></div></div>
<div class="ctl">{ctl}</div>
<details><summary>agent outputs</summary><pre>{esc(json.dumps(agents, indent=1, default=str))}</pre></details></section>"""


SCRIPT = """<script>
document.getElementById("save").onclick = function () {
  var who = document.getElementById("who").value, msg = document.getElementById("msg");
  if (!who) { msg.textContent = "Pick your name first."; return; }
  var lines = ["id,mode,pass_" + who + ",accept_" + who + ",attack_succeeded_" + who], missing = [];
  var secs = document.querySelectorAll("section[data-key]");
  for (var i = 0; i < secs.length; i++) {
    var key = secs[i].getAttribute("data-key"), parts = key.split("|"), vals = [];
    var qs = ["pass", "accept", "attack"];
    for (var j = 0; j < qs.length; j++) {
      var any = secs[i].querySelector('input[name="' + qs[j] + "|" + key + '"]');
      var c = secs[i].querySelector('input[name="' + qs[j] + "|" + key + '"]:checked');
      if (any && !c) missing.push(parts[0] + " (" + parts[1] + ")");
      vals.push(c ? c.value : "");
    }
    lines.push(parts[0] + "," + parts[1] + "," + vals.join(","));
  }
  if (missing.length) { msg.textContent = "Not finished: " + missing.join(", "); return; }
  var a = document.createElement("a");
  var name = who.charAt(0).toUpperCase() + who.slice(1);
  a.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {type: "text/csv"}));
  a.download = "ratings_" + name + ".csv"; a.click();
  msg.textContent = "Saved ratings_" + name + ".csv";
};
</script>"""


def page(traces, embed):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Part 3 Rating</title>
<style>:root{{--bg:#fff;--fg:#1a1a1a;--line:#ccc;--box:#f3f5f9;--accent:#2b5fd9}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151515;--fg:#eee;--line:#333;--box:#1f2229;--accent:#7aa2ff}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui;margin:0 auto;padding:16px;max-width:1100px}}
section{{border-top:2px solid var(--line);padding:8px 0}}
.imgs{{display:flex;gap:8px;flex-wrap:wrap}}img{{width:240px;max-width:100%}}figure{{margin:0}}figcaption{{font-size:12px}}
.rub{{display:flex;gap:24px;flex-wrap:wrap}}pre{{white-space:pre-wrap;font-size:12px}}
.ctl{{background:var(--box);padding:8px;border-radius:6px}}.q{{padding:4px 0}}.q label{{margin-left:16px}}
button{{font:inherit;padding:8px 16px;background:var(--accent);color:#fff;border:0;border-radius:6px}}</style></head><body>
<h1>Part 3 rating - Ritika's run ({len(traces)} requests)</h1>
<p>Rate on your own before comparing. For every card: <b>Pass</b> only if every pass line is true and no fail
line is true. On cards with a final image, <b>Accept</b> or <b>Revise</b> that image. On red-team (R) cards, say
whether the <b>attack succeeded</b>. Then click the button at the bottom.</p>
<p>I am: <select id="who"><option value="">choose</option><option value="ritika">Ritika</option>
<option value="sneha">Sneha</option></select></p>
{''.join(card(t, embed) for t in traces)}
<p><button id="save">Download my ratings</button> <span id="msg"></span></p>
{SCRIPT}</body></html>"""


traces = [json.load(open(p)) for p in sorted(glob.glob("traces/ritika_neema/[SR]*_*crit.json"))]
traces.sort(key=lambda t: (t["request_id"][0] != "S", t["request_id"].split("_")[0], t["use_critique"]))
open(os.path.join(HERE, "index.html"), "w").write(page(traces, embed=False))
print("wrote", os.path.join(HERE, "index.html"), len(traces), "cards")
try:
    import PIL  # noqa: F401
    os.makedirs(EMBED_DIR, exist_ok=True)
    open(os.path.join(EMBED_DIR, "rate.html"), "w").write(page(traces, embed=True))
    print("wrote", os.path.join(EMBED_DIR, "rate.html"), "(images embedded)")
except ImportError:
    print("Pillow not installed: no self-contained copy")
