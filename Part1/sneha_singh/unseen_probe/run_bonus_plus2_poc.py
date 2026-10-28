#!/usr/bin/env python3
"""POC: passage-reranker OFF vs ON on a held-out harder set (not golden).

Goal: see whether ON answer-correct count can beat OFF for the lab +2 claim.
Does not change answering code or hard-code question answers.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["RUN_FULL_EVALUATION"] = "0"

ROOT = Path("/Users/snehasingh/Desktop/Fall 2026 Lab 2")
NB_PATH = ROOT / "notebooks/sneha_singh/p1_rag.ipynb"
PROBE = ROOT / "Part1/sneha_singh/unseen_probe/bonus_plus2_poc_questions.json"
OUT = ROOT / "Part1/sneha_singh/outputs/evaluation/bonus_plus2_poc_20261027"
LOG = OUT / "runner.log"


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NB_PATH.read_text())
    ns: dict = {"__name__": "__main__"}
    cells = [0, 4, 6, 8, 10, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 32]
    try:
        for i in cells:
            src = "".join(nb["cells"][i].get("source", []))
            exec(compile(src, f"nb_cell_{i}", "exec"), ns, ns)
            log(f"loaded cell {i}")
    except Exception:
        log(traceback.format_exc())
        return 1

    CONFIG = ns["CONFIG"]
    Record = ns["Record"]
    root = ns["project_root"](ROOT)
    records = [
        Record(obj["record_id"], obj["text"], obj["metadata"])
        for obj in map(json.loads, (root / CONFIG["artifact_dir"] / "records.jsonl").read_text().splitlines())
    ]
    probe = json.loads(PROBE.read_text())
    items = probe["items"]
    log(f"probe items={len(items)} ids={[i['id'] for i in items]}")

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
        records, encoder, vector_dir=root / CONFIG["artifact_dir"] / "chroma",
        reranker=reranker, passage_reranking=True,
    )

    results = {}
    for enabled, name in [(False, "off"), (True, "on")]:
        log(f"==== ARM {name} passage_reranking={enabled} ====")
        retriever.passage_reranking = enabled
        rag = ns["ConversationRAG"](retriever, model, CONFIG["top_k"], mode="hybrid")
        sig = hashlib.sha256(f"bonus-poc-{name}-{time.time()}".encode()).hexdigest()
        rows = ns["evaluate_system"](
            items, rag, model, OUT / f"{name}.jsonl", sig,
            evaluation_signature=hashlib.sha256(f"bonus-poc-eval-{name}".encode()).hexdigest(),
            retry_generation_errors=False,
        )
        by_id = {row["id"]: row for row in rows}
        results[name] = by_id
        correct = sum(1 for row in rows if row.get("answer_correct"))
        log(f"ARM {name}: correct={correct}/{len(rows)}")
        for row in rows:
            log(
                f"  {name} {row['id']}: correct={row.get('answer_correct')} "
                f"status={row.get('status_correct')} num={row.get('numeric_correct')} "
                f"rubric={row.get('rubric_patterns_correct')} "
                f"judge={(row.get('judge') or {}).get('answer_correct')} "
                f"err={(row.get('error') or '')[:100]}"
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
        "off_correct": off_n,
        "on_correct": on_n,
        "n": len(items),
        "gained": gained,
        "lost": lost,
        "bonus_plus2_claimable": bool(on_n > off_n),
        "note": "Held-out POC only; not golden-set and not a Kaggle score.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("SUMMARY " + json.dumps(summary))
    log("BONUS_YES" if summary["bonus_plus2_claimable"] else "BONUS_NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
