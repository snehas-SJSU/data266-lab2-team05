#!/usr/bin/env python3
"""Fresh OFF/ON passage-reranker ablation after the blend fix.

Writes to evaluation/reranker_ablation_postblend_20261027/ and does NOT
overwrite the historical reranker_ablation/ 20-vs-19 table.
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
OUT = ROOT / "Part1/sneha_singh/outputs/evaluation/reranker_ablation_fix100_20261027"
LOG = OUT / "runner.log"


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def exec_cells(ns: dict, indices: list[int]) -> None:
    nb = json.loads(NB_PATH.read_text())
    for i in indices:
        src = "".join(nb["cells"][i].get("source", []))
        if not src.strip():
            continue
        log(f"exec cell {i} ({len(src)} chars)")
        exec(compile(src, f"nb_cell_{i}", "exec"), ns, ns)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.json").write_text(
        json.dumps({"status": "starting", "started_at_epoch": time.time()}, indent=2)
    )
    ns: dict = {"__name__": "__main__", "__file__": str(NB_PATH)}
    # Implementation + eval helpers + CONFIG. Skip heavy Part-B rebuild cells.
    # 0 bootstrap, 4-32 definition cells used by ConversationRAG / evaluate_system.
    cells = [0, 4, 6, 8, 10, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 32]
    try:
        exec_cells(ns, cells)
    except Exception:
        log("FAILED while loading notebook cells:\n" + traceback.format_exc())
        (OUT / "status.json").write_text(
            json.dumps({"status": "failed_load", "error": traceback.format_exc()}, indent=2)
        )
        return 1

    # Paths / models / retrieval from existing artifacts (no PDF rebuild).
    CONFIG = ns["CONFIG"]
    Record = ns["Record"]
    DenseEncoder = ns["DenseEncoder"]
    PassageReranker = ns["PassageReranker"]
    HybridRetriever = ns["HybridRetriever"]
    OllamaReasoner = ns["OllamaReasoner"]
    ConversationRAG = ns["ConversationRAG"]
    load_golden = ns["load_golden"]
    evaluate_system = ns["evaluate_system"]
    summarize = ns["summarize"]
    sha256 = ns["sha256"]
    SEED = ns["SEED"]
    project_root = ns["project_root"]

    root = project_root(ROOT)
    records_path = root / CONFIG["artifact_dir"] / "records.jsonl"
    records = [
        Record(obj["record_id"], obj["text"], obj["metadata"])
        for obj in map(json.loads, records_path.read_text().splitlines())
    ]
    log(f"loaded {len(records)} records from {records_path}")

    golden = load_golden(root / CONFIG["golden_set"])
    items = golden["items"]
    assert len(items) == 25

    model = OllamaReasoner(
        CONFIG["answer_model"],
        CONFIG["ollama_endpoint"],
        SEED,
        expected_digest=CONFIG["answer_model_digest"],
    )
    encoder = DenseEncoder(
        CONFIG["embedding_model"],
        CONFIG["embedding_revision"],
        CONFIG["embedding_local_files_only"],
    )
    reranker = PassageReranker(
        CONFIG["reranker_model"],
        CONFIG["reranker_revision"],
        CONFIG["reranker_local_files_only"],
    )
    retriever = HybridRetriever(
        records,
        encoder,
        vector_dir=root / CONFIG["artifact_dir"] / "chroma",
        reranker=reranker,
        passage_reranking=True,
    )
    log(f"answerer={model.metadata.get('model')} encoder={encoder.model_name}")

    notebook = json.loads(NB_PATH.read_text())
    sources = [
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
    ]
    source_hash = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
    common = {
        "purpose": "post-blend passage-reranker OFF vs ON re-measure",
        "notebook_sources_sha256": source_hash,
        "golden_sha256": sha256(root / CONFIG["golden_set"]),
        "records_sha256": sha256(records_path),
        "evaluation_use": "automated/provisional post-blend development ablation",
        "historical_ablation_preserved": "Part1/sneha_singh/outputs/evaluation/reranker_ablation",
        "arm_order": ["off", "on"],
        "model": model.metadata,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(common, indent=2))

    rows_by_arm = {}
    old_toggle = retriever.passage_reranking
    try:
        for name, enabled in [("off", False), ("on", True)]:
            log(f"==== ARM {name.upper()} passage_reranking={enabled} ====")
            (OUT / "status.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "phase": name,
                        "passage_reranking": enabled,
                        "updated_at_epoch": time.time(),
                    },
                    indent=2,
                )
            )
            retriever.passage_reranking = enabled
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "arm": name,
                        "passage_reranking": enabled,
                        "source": source_hash,
                        "golden": common["golden_sha256"],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            evaluation_signature = hashlib.sha256(
                f"postblend-eval-{source_hash}".encode()
            ).hexdigest()
            rows = evaluate_system(
                items,
                ConversationRAG(retriever, model, CONFIG["top_k"], mode="hybrid"),
                model,
                OUT / f"{name}.jsonl",
                signature,
                evaluation_signature=evaluation_signature,
                retry_generation_errors=False,
            )
            if len(rows) != 25:
                raise RuntimeError(f"arm {name} incomplete: {len(rows)} rows")
            rows_by_arm[name] = rows
            arm_metrics = summarize(rows)
            log(
                f"ARM {name}: correct={arm_metrics.get('answer_correct_out_of_25')} "
                f"recall@5={arm_metrics.get('retrieval_recall_at_5')} "
                f"gen_err={arm_metrics.get('generation_errors')}"
            )
    finally:
        retriever.passage_reranking = old_toggle

    metrics = {
        "protocol": "postblend_20261027",
        "evaluation_use": common["evaluation_use"],
        "off": summarize(rows_by_arm["off"]),
        "on": summarize(rows_by_arm["on"]),
    }
    paired = {name: {row["id"]: row for row in rows} for name, rows in rows_by_arm.items()}
    metrics["paired_changes"] = {
        "gained": [
            item["id"]
            for item in items
            if paired["off"][item["id"]]["answer_correct"] is False
            and paired["on"][item["id"]]["answer_correct"] is True
        ],
        "lost": [
            item["id"]
            for item in items
            if paired["off"][item["id"]]["answer_correct"] is True
            and paired["on"][item["id"]]["answer_correct"] is False
        ],
    }
    off_n = metrics["off"]["answer_correct_out_of_25"]
    on_n = metrics["on"]["answer_correct_out_of_25"]
    metrics["bonus_plus2_claimable"] = bool(on_n > off_n)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_epoch": time.time(),
                "off": off_n,
                "on": on_n,
                "bonus_plus2_claimable": metrics["bonus_plus2_claimable"],
                "gained": metrics["paired_changes"]["gained"],
                "lost": metrics["paired_changes"]["lost"],
            },
            indent=2,
        )
    )
    log(
        f"DONE off={off_n}/25 on={on_n}/25 "
        f"gained={metrics['paired_changes']['gained']} "
        f"lost={metrics['paired_changes']['lost']} "
        f"bonus={metrics['bonus_plus2_claimable']}"
    )
    return 0 if metrics["bonus_plus2_claimable"] or on_n >= off_n else 0


if __name__ == "__main__":
    sys.exit(main())
