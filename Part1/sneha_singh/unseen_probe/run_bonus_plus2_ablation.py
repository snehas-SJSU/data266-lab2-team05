#!/usr/bin/env python3
"""Golden-set OFF vs ON ablation for the Part 1 +2 bonus notebook.

Loads implementation from notebooks/sneha_singh/p1_bonus_plus2_poc.ipynb only.
Does not modify notebooks/sneha_singh/p1_rag.ipynb.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["RUN_FULL_EVALUATION"] = "0"

ROOT = Path("/Users/snehasingh/Desktop/Fall 2026 Lab 2")
BONUS_NB = ROOT / "notebooks/sneha_singh/p1_bonus_plus2_poc.ipynb"
P1_NB = ROOT / "notebooks/sneha_singh/p1_rag.ipynb"
GOLDEN = ROOT / "eval/golden_set.json"
OUT = ROOT / "Part1/sneha_singh/outputs/bonus_plus2/evaluation/bonus_plus2_ablation"
LOG = OUT / "runner.log"


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p1_sha = hashlib.sha256(P1_NB.read_bytes()).hexdigest()
    bonus_sha = hashlib.sha256(BONUS_NB.read_bytes()).hexdigest()
    log(f"START bonus_nb={bonus_sha} p1_nb={p1_sha}")

    nb = json.loads(BONUS_NB.read_text())
    ns: dict = {"__name__": "__main__"}
    for i in [0, 4, 6, 8, 10, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 32]:
        src = "".join(nb["cells"][i].get("source", []))
        exec(compile(src, f"bonus_cell_{i}", "exec"), ns, ns)
        log(f"loaded bonus cell {i}")

    CONFIG = dict(ns["CONFIG"])
    CONFIG["outputs_dir"] = "Part1/sneha_singh/outputs/bonus_plus2"
    shortlist = int(CONFIG.get("rerank_shortlist", 40))
    Record = ns["Record"]
    root = ns["project_root"](ROOT)
    # Reuse Part 1 built index (read-only); answers/metrics write under bonus_plus2/.
    records = [
        Record(obj["record_id"], obj["text"], obj["metadata"])
        for obj in map(json.loads, (root / "Part1/sneha_singh/outputs/index" / "records.jsonl").read_text().splitlines())
    ]
    items = json.loads(GOLDEN.read_text())["items"]
    log(f"records={len(records)} golden={len(items)} shortlist={shortlist}")

    model = ns["OllamaReasoner"](
        CONFIG["answer_model"], CONFIG["ollama_endpoint"], ns["SEED"],
        expected_digest=CONFIG["answer_model_digest"],
    )
    encoder = ns["DenseEncoder"](
        CONFIG["embedding_model"], CONFIG["embedding_revision"], CONFIG["embedding_local_files_only"]
    )
    reranker = ns["PassageReranker"](
        CONFIG["reranker_model"], CONFIG["reranker_revision"], CONFIG["reranker_local_files_only"]
    )
    retriever = ns["HybridRetriever"](
        records, encoder,
        vector_dir=root / "Part1/sneha_singh/outputs/index" / "chroma",
        reranker=reranker, passage_reranking=True,
    )
    retriever.rerank_shortlist = shortlist

    results = {}
    for enabled, name in [(False, "off"), (True, "on")]:
        log(f"==== ARM {name} passage_reranking={enabled} shortlist={retriever.rerank_shortlist} ====")
        retriever.passage_reranking = enabled
        rag = ns["ConversationRAG"](retriever, model, CONFIG["top_k"], mode="hybrid")
        sig = hashlib.sha256(f"bonus-plus2-{name}-{shortlist}".encode()).hexdigest()
        rows = ns["evaluate_system"](
            items, rag, model, OUT / f"{name}.jsonl", sig,
            evaluation_signature=hashlib.sha256(f"bonus-plus2-eval-{name}".encode()).hexdigest(),
            retry_generation_errors=False,
        )
        results[name] = {row["id"]: row for row in rows}
        correct = sum(1 for row in rows if row.get("answer_correct"))
        log(f"ARM {name}: {correct}/{len(rows)}")
        for row in rows:
            log(
                f"  {name} {row['id']}: correct={row.get('answer_correct')} "
                f"status={row.get('status_correct')} num={row.get('numeric_correct')} "
                f"rubric={row.get('rubric_patterns_correct')} "
                f"judge={(row.get('judge') or {}).get('answer_correct')} "
                f"err={(row.get('error') or '')[:120]}"
            )

    off_n = sum(1 for r in results["off"].values() if r.get("answer_correct"))
    on_n = sum(1 for r in results["on"].values() if r.get("answer_correct"))
    gained = sorted(
        qid for qid, row in results["on"].items()
        if row.get("answer_correct") and not results["off"][qid].get("answer_correct")
    )
    lost = sorted(
        qid for qid, row in results["off"].items()
        if row.get("answer_correct") and not results["on"][qid].get("answer_correct")
    )
    summary = {
        "protocol": "bonus_plus2_deeper_shortlist_ce",
        "shortlist": shortlist,
        "off_correct": off_n,
        "on_correct": on_n,
        "n": len(items),
        "gained": gained,
        "lost": lost,
        "bonus_plus2_claimable": bool(on_n > off_n),
        "bonus_notebook_sha256": bonus_sha,
        "p1_rag_sha256_unchanged": p1_sha,
        "note": "Bonus notebook HybridRetriever OFF vs ON on golden set. Graded p1_rag.ipynb untouched.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "Part1/sneha_singh/outputs/bonus_plus2/evaluation/bonus_plus2_ablation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log("SUMMARY " + json.dumps(summary))
    # Report figure for the run
    try:
        import matplotlib.pyplot as plt
        fig_path = ROOT / "report/figures/part1/reranker_bonus_ablation.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        ax.axis("off")
        claim = "claimable (ON > OFF)" if summary["bonus_plus2_claimable"] else "not claimable"
        table = ax.table(
            cellText=[
                ["p1_rag.ipynb (graded Part 1)", "25/25", "25/25", "not claimable (tie)"],
                ["p1_bonus_plus2_poc.ipynb (merged +2)", f"{off_n}/25", f"{on_n}/25", claim],
            ],
            colLabels=["Notebook", "Reranker OFF", "Reranker ON", "+2 bonus"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.15, 1.7)
        ax.set_title("Part 1 passage-reranker golden ablation", pad=14)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        log(f"FIGURE {fig_path}")
    except Exception as exc:
        log(f"FIGURE_FAILED {exc}")
    log("BONUS_YES" if summary["bonus_plus2_claimable"] else "BONUS_NO")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["bonus_plus2_claimable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
