#!/usr/bin/env python3
"""Focused actual-model check for G05/G12/G15/G16 after 100%-fix patches."""
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
OUT = ROOT / "Part1/sneha_singh/outputs/evaluation/fix100_focused_20261027"
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
    golden = ns["load_golden"](root / CONFIG["golden_set"])
    by_id = {item["id"]: item for item in golden["items"]}
    # Preserve conversation order for GA (G15->G16->G17)
    focus_ids = ["G05", "G12", "G15", "G16"]
    items = [by_id[i] for i in focus_ids]

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
        log(f"==== ARM {name} ====")
        retriever.passage_reranking = enabled
        sig = hashlib.sha256(f"fix100-{name}-{time.time()}".encode()).hexdigest()
        # evaluate_system processes items in list order but ConversationRAG is fresh each call
        # For G15/G16 we need one ConversationRAG across the conversation.
        rag = ns["ConversationRAG"](retriever, model, CONFIG["top_k"], mode="hybrid")
        # Build ordered items: singles then conversation turns together
        ordered = [by_id["G05"], by_id["G12"]] + [
            item for item in golden["items"] if item["conversation_id"] == "GA" and item["id"] in {"G15", "G16"}
        ]
        rows = ns["evaluate_system"](
            ordered, rag, model, OUT / f"{name}.jsonl", sig,
            evaluation_signature=hashlib.sha256(f"fix100-eval-{name}".encode()).hexdigest(),
            retry_generation_errors=False,
        )
        results[name] = {row["id"]: row for row in rows if row["id"] in focus_ids}
        for qid in focus_ids:
            row = results[name][qid]
            log(
                f"{name} {qid}: correct={row.get('answer_correct')} "
                f"status={row.get('status_correct')} num={row.get('numeric_correct')} "
                f"rubric={row.get('rubric_patterns_correct')} "
                f"judge={(row.get('judge') or {}).get('answer_correct')} "
                f"ans={(row.get('prediction') or {}).get('answer', '')[:120]}"
            )

    summary = {
        arm: {qid: bool(results[arm][qid].get("answer_correct")) for qid in focus_ids}
        for arm in ["off", "on"]
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("SUMMARY " + json.dumps(summary))
    ok = all(summary["off"].values()) and all(summary["on"].values())
    log("ALL_PASS" if ok else "SOME_FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
