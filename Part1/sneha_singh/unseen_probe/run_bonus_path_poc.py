#!/usr/bin/env python3
"""Run deeper-shortlist CE path POC for Part 1 +2 bonus.

Does not modify notebooks/sneha_singh/p1_rag.ipynb.
Writes under Part1/sneha_singh/outputs/bonus_plus2/path_poc/.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["RUN_FULL_EVALUATION"] = "0"

ROOT = Path("/Users/snehasingh/Desktop/Fall 2026 Lab 2")
P1_NB = ROOT / "notebooks/sneha_singh/p1_rag.ipynb"
GOLDEN = ROOT / "eval/golden_set.json"
OUT = ROOT / "Part1/sneha_singh/outputs/bonus_plus2/path_poc"
LOG = OUT / "runner.log"
SHORTLIST = int(os.environ.get("BONUS_SHORTLIST", "40"))
RUN_ANSWERS = os.environ.get("RUN_BONUS_ANSWER_ABLATION", "1") == "1"


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p1_sha = hashlib.sha256(P1_NB.read_bytes()).hexdigest()
    log(f"START p1_rag_sha={p1_sha} shortlist={SHORTLIST} answers={RUN_ANSWERS}")

    nb = json.loads(P1_NB.read_text())
    ns: dict = {"__name__": "__main__"}
    for i in [0, 4, 6, 8, 10, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 32]:
        src = "".join(nb["cells"][i].get("source", []))
        exec(compile(src, f"p1_rag_cell_{i}", "exec"), ns, ns)
        log(f"loaded p1 cell {i}")

    CONFIG = dict(ns["CONFIG"])
    CONFIG["outputs_dir"] = "Part1/sneha_singh/outputs/bonus_plus2"
    Record = ns["Record"]
    root = ns["project_root"](ROOT)
    records = [
        Record(obj["record_id"], obj["text"], obj["metadata"])
        for obj in map(json.loads, (root / CONFIG["artifact_dir"] / "records.jsonl").read_text().splitlines())
    ]
    items = json.loads(GOLDEN.read_text())["items"]
    log(f"records={len(records)} golden={len(items)}")

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
    base = ns["HybridRetriever"](
        records, encoder, vector_dir=root / CONFIG["artifact_dir"] / "chroma",
        reranker=reranker, passage_reranking=True,
    )
    top_k = int(CONFIG["top_k"])

    def hybrid_fused_indices(retriever, query: str, pool: int):
        query_embedding = retriever.encoder.encode([query])[0]
        dense_scores = np.einsum("ij,j->i", retriever.embeddings, query_embedding)
        dense_order = list(reversed(dense_scores.argsort()))
        if retriever.collection is not None:
            response = retriever.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(max(pool, 20), len(retriever.records)),
            )
            dense_order = [retriever.by_id[record_id] for record_id in response["ids"][0]]
        bm25_scores = retriever.bm25.get_scores(ns["_tokenise"](query))
        bm25_order = sorted(range(len(retriever.records)), key=lambda i: bm25_scores[i], reverse=True)
        fused: dict[int, float] = {}
        for rank, index in enumerate(dense_order[:pool], start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        for rank, index in enumerate(bm25_order[:pool], start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        visual_labels = ns["visual_labels"]
        _normalise_label = ns["_normalise_label"]
        target_labels = {_normalise_label(x).lower() for x in visual_labels(query)}
        exact = [
            i for i, record in enumerate(retriever.records)
            if (record.metadata.get("figure_id") or "").lower() in target_labels
        ]
        related = []
        for i, record in enumerate(retriever.records):
            label = (record.metadata.get("figure_id") or "").lower()
            if label.endswith(" (corrected)") and label.removesuffix(" (corrected)") in target_labels:
                fused.setdefault(i, 0.0)
                related.append(i)
        for i in exact:
            fused.setdefault(i, 0.0)
        return list(fused), fused, exact, related, dense_scores, bm25_scores

    def hits_from_order(retriever, order, fused, exact, related, k, dense_scores, bm25_scores):
        hits = []
        pages_seen = set()
        RetrievalHit = ns["RetrievalHit"]
        for index in order:
            key = (retriever.records[index].metadata["doc_id"], retriever.records[index].metadata["page"])
            if key in pages_seen and index not in exact and index not in related:
                continue
            pages_seen.add(key)
            hits.append(RetrievalHit(
                record=retriever.records[index],
                score=fused[index],
                dense_score=float(dense_scores[index]),
                bm25_score=float(bm25_scores[index]),
            ))
            if len(hits) >= k:
                break
        return hits

    def retrieve_off(retriever, query, k=top_k):
        candidates, fused, exact, related, dense_scores, bm25_scores = hybrid_fused_indices(
            retriever, query, max(4 * k, 20)
        )
        order = sorted(candidates, key=lambda i: (i in exact, i in related, fused[i]), reverse=True)
        return hits_from_order(retriever, order, fused, exact, related, k, dense_scores, bm25_scores)

    def retrieve_on_deep(retriever, query, k=top_k, shortlist=SHORTLIST):
        candidates, fused, exact, related, dense_scores, bm25_scores = hybrid_fused_indices(
            retriever, query, shortlist
        )
        scores = retriever.reranker.score(query, [retriever.records[i].text for i in candidates])
        ce = dict(zip(candidates, scores))
        order = sorted(candidates, key=lambda i: (i in exact, i in related, ce[i], fused[i]), reverse=True)
        return hits_from_order(retriever, order, fused, exact, related, k, dense_scores, bm25_scores)

    def hit_keys(hits):
        return [
            (h.record.metadata["doc_id"], h.record.metadata["page"], h.record.metadata.get("figure_id"))
            for h in hits
        ]

    log("==== RANKING DIAGNOSTIC ====")
    rank_rows = []
    for item in items:
        q = item["question"]
        off = retrieve_off(base, q)
        on = retrieve_on_deep(base, q)
        off_k, on_k = hit_keys(off), hit_keys(on)
        needed = {(s["doc_id"], s.get("page")) for s in item.get("sources", []) if s.get("page") is not None}

        def recall(keys, needed=needed):
            pages = {(d, p) for d, p, _ in keys}
            return sum(1 for n in needed if n in pages) / max(len(needed), 1)

        rank_rows.append({
            "id": item["id"],
            "order_changed": off_k != on_k,
            "off_recall_sources": recall(off_k),
            "on_recall_sources": recall(on_k),
            "off_top": off_k,
            "on_top": on_k,
        })

    summary_rank = {
        "shortlist": SHORTLIST,
        "top_k": top_k,
        "n": len(rank_rows),
        "order_changed": sum(1 for r in rank_rows if r["order_changed"]),
        "on_better_source_recall": sum(1 for r in rank_rows if r["on_recall_sources"] > r["off_recall_sources"]),
        "off_better_source_recall": sum(1 for r in rank_rows if r["off_recall_sources"] > r["on_recall_sources"]),
        "mean_off_source_recall": sum(r["off_recall_sources"] for r in rank_rows) / len(rank_rows),
        "mean_on_source_recall": sum(r["on_recall_sources"] for r in rank_rows) / len(rank_rows),
    }
    (OUT / "ranking_diagnostic.json").write_text(
        json.dumps({"summary": summary_rank, "rows": rank_rows}, indent=2, default=str)
    )
    log("RANK " + json.dumps(summary_rank))

    if not RUN_ANSWERS:
        log("Answer ablation skipped (RUN_BONUS_ANSWER_ABLATION!=1)")
        return 0

    class ArmRetriever:
        def __init__(self, inner, mode: str):
            self.inner = inner
            self.mode = mode
            self.records = inner.records
            self.encoder = inner.encoder
            self.reranker = inner.reranker
            self.passage_reranking = mode == "on"
            self.bm25 = inner.bm25

        def retrieve(self, query, top_k=5, preferred_visuals=None, source_question=None):
            if self.mode == "on":
                return retrieve_on_deep(self.inner, query, k=top_k)
            return retrieve_off(self.inner, query, k=top_k)

        def retrieve_dense(self, query, top_k=5):
            return self.inner.retrieve_dense(query, top_k=top_k)

    ConversationRAG = ns["ConversationRAG"]
    results = {}
    for mode in ("off", "on"):
        log(f"==== ANSWER ARM {mode} ====")
        retriever = ArmRetriever(base, mode)
        rag = ConversationRAG(retriever, model, CONFIG["top_k"], mode="hybrid")
        sig = hashlib.sha256(f"bonus-path-{mode}-{SHORTLIST}".encode()).hexdigest()
        rows = ns["evaluate_system"](
            items, rag, model, OUT / f"answers_{mode}.jsonl", sig,
            evaluation_signature=hashlib.sha256(f"bonus-path-eval-{mode}".encode()).hexdigest(),
            retry_generation_errors=False,
        )
        results[mode] = {row["id"]: row for row in rows}
        correct = sum(1 for row in rows if row.get("answer_correct"))
        log(f"ARM {mode}: {correct}/{len(rows)}")
        for row in rows:
            log(
                f"  {mode} {row['id']}: correct={row.get('answer_correct')} "
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
        "path": "deeper_shortlist_ce",
        "shortlist": SHORTLIST,
        "off_correct": off_n,
        "on_correct": on_n,
        "n": len(items),
        "gained": gained,
        "lost": lost,
        "bonus_plus2_claimable": bool(on_n > off_n),
        "ranking": summary_rank,
        "p1_rag_sha256": p1_sha,
        "note": "POC only. Fold winning ON design into graded notebook before claiming lab +2.",
    }
    (OUT / "answer_summary.json").write_text(json.dumps(summary, indent=2))
    log("SUMMARY " + json.dumps(summary))
    log("BONUS_YES" if summary["bonus_plus2_claimable"] else "BONUS_NO")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
