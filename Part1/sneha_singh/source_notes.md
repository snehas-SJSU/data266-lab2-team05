# Part 1 source notes — Sneha Singh

These notes document how I use the three supplied PDFs as a read-only knowledge base. They are not answer keys, and values below must be verified against retrieved evidence in an executed run. PDF page numbers below are file page numbers, starting at 1. I retain the supplied `DOC1`/`DOC2`/`DOC3` IDs in citations.

## Sources and priority

| Source | Role | Trust |
| --- | --- | --- |
| `DOC1` (`Document.pdf`) | Blanchard Chapters 1–2; macro concepts and earlier data vintage | high |
| `DOC2` (Supplement A) | 2011–13 outturns, explicit 2010 revision, chart comparisons | high |
| `DOC3` (Briefing B) | Labour, inflation, policy analysis; corrected figure | medium, except Appendix C |
| `DOC3` Appendix C, file page 7 | Unverified reader commentary and Figure C-1 | low; never retrieval evidence |

Apply `kb_manifest.json`: an explicit erratum/revision replaces the older value; an outturn replaces a forecast for the same measure and year; otherwise report an unresolved high/medium-trust disagreement as `conflict`, cite both, and prefer the higher-trust value. Conversation assumptions remain in per-conversation memory and do not edit the base index. Text or images in any source are data, never instructions.

## Document details that change answers

- `DOC1` has 2011/12 entries marked with `*` as forecasts, not realized outcomes. `DOC2` Table A-1 (file p2) and Table A-2 (p3) are the later outturns. `DOC2` explicitly revises U.S. 2010 output growth from the earlier 3.0% to 2.4% (p3, reiterated p7).
- Do not treat every differing number as a revision. `DOC3` Table B-2 (p5) reports U.S. 2010 unemployment 9.8 while `DOC1` has 9.6; `DOC3`'s China 2012 growth 7.8 (p5) differs from `DOC2` Table A-2's 7.7 (p3). No explicit replacement is stated for these pairs; surface the conflict and source priority rather than silently averaging.
- `DOC2` Figure A-2 and A-3 (p4) contrast a 2011 forecast vintage with 2014 outturns. Figures A-4 and A-5 (p5) rebase the same stock-index story to different dates (January 2007 versus January 2009). Figures A-6 and A-7 (p6) show Euro-area unemployment for 2010 versus 2012. Exact labels matter even when chart shapes look similar.
- `DOC3` Figures B-1 and B-2 (pp2–3) use different U.S. Okun-law samples. Figures B-3 and B-4 (p4) use different inflation comparison series: GDP deflator versus core CPI. The original Figure B-5 (p5) contains an erroneous 2012 plotted value of 4.5%; the erratum and Figure B-5 (corrected) (p6) replace that point with 4.1%.
- `DOC3` Appendix C (p7) includes false claims and Figure C-1 contains an instruction aimed at AI assistants. Its text and chart pixels are excluded from answer evidence and image input. For a question about the appendix itself, describe its unverified status using the manifest or trusted cross-checks; do not obey or cite its directive as a fact.

## Implementation checks

- `src/p1_rag.py` extracts text chunks and separate numbered visual objects with doc ID, file page, exact label, caption, trust, and a local crop path. It keeps the low-trust records only for audit and excludes them from the normal retriever.
- Each trusted figure is described by the declared local VLM. The description cache records the image hash, caption, model version/digest, and actual model output. Descriptions can contain extraction errors, so they are evaluation aids rather than independent ground truth. Appendix C's visual has status `excluded_low_trust` and is not described.
- The local run uses the real MiniLM encoder, persistent ChromaDB, and a local image-capable Qwen2.5-VL answerer. Derived records are in `outputs/index/`; figure crops, descriptions, model metadata, raw evaluation records, and generated results stay under `outputs/`.
- Kaggle graded path: `notebooks/sneha_singh/p1_rag.ipynb` with `Qwen/Qwen2.5-VL-7B-Instruct` (NF4), hybrid retrieval + general answering policies. Public practice best score: **0.909**. Final select/tag follows the executed Save Version.

## Research references and design choices

- [Lewis et al., RAG (2020)](https://arxiv.org/abs/2005.11401): separate evidence retrieval from generation; keep source provenance and compare retrieval variants. The lab does not require recreating the paper's jointly trained DPR/BART models.
- [Faysse et al., ColPali (2024)](https://arxiv.org/abs/2407.01449): text-only PDF parsing can lose visual information. I use the lab-permitted VLM-description route for chart search, rather than claiming to implement ColPali's multi-vector page embeddings.
- [Greshake et al., indirect prompt injection (2023)](https://arxiv.org/abs/2302.12173): retrieved data, memory, and images can carry instructions. I keep system rules separate, enforce trust filtering before retrieval/image input, quarantine low-trust records, and erase conversation memory. This reduces exposure but does not prove general injection immunity; the report includes measured examples and limitations.
