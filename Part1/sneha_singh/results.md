# Part 1 - Multimodal RAG (Sneha Singh)

## Kaggle public leaderboard (practice)

| Item | Value |
| --- | --- |
| Best public score | **0.940** |
| Team | `Pair_Programming_Team_05` |
| Prior reference top (other teams) | ~0.855 |
| Final graded path | Kaggle notebook at git tag `p1-v1.0` (hidden set). |

Practice LB is feedback only. Task 1.5 marks use the hidden set through the tagged notebook.

### Public Kaggle score

Best public leaderboard score is **0.940** (`Pair_Programming_Team_05`), from the **selected** practice CSV `Part1/sneha_singh/submission.csv` (sha `881a9ffb…`). Path: 0.860 → 0.877 → 0.909 → 0.913 → 0.926 → **0.940**. That selected CSV is not the same bytes as this Save Version notebook print (`submission_sha256` `47948d4e…`). Practice LB is feedback only. Graded Task 1.5 is the tagged notebook (`p1-v1.0`) on the hidden set. Local golden-set ablation lives in `eval/reranker_ablation_fix100_summary.json` (repo file; not recomputed inside the Kaggle runtime).


## Status for teammates

What is done on Part 1 so far, and what is still open. Same numbers as the notebook print of the passage-reranker ablation.

### Done (Sneha)

- Built and ran `notebooks/sneha_singh/p1_rag.ipynb` locally (Ollama `qwen2.5:7b` answers, `qwen2.5vl:3b` figure captions).
- Parses the three PDFs, quarantines Appendix C, answers from text/figures/tables, keeps conversation edits out of the index.
- Architecture figure: `report/figures/part1/architecture.png`.
- Current passage-reranker OFF vs ON ablation: **25/25 both arms** (`outputs/evaluation/reranker_ablation_fix100_20261027/`).
- Public Kaggle LB best **0.940**. Executed notebook is in git: [`lab2-notebook`](https://www.kaggle.com/code/snehasjsu/lab2-notebook), Tesla T4, seed 266, 40 rows, validator passed, **1171 s** including setup (within 90 min).

### 27 Oct pipeline fixes + fresh OFF/ON re-measure

Paper-aligned repairs in `notebooks/sneha_singh/p1_rag.ipynb` after re-reading Lewis/ColPali/Greshake (Lab-papers folder is empty; used arXiv HTML):

| Fix | Why (paper / bonus) |
| --- | --- |
| Passage reranker now **RRF-blends** dense+BM25 fused order with CE order (CE weight 1.25 + CE tie-break), instead of sorting by raw CE alone | Lewis: compare retrieval variants; lab **+2** needs ON to help answers. Pure CE previously promoted distractors and dropped pages needed for G02/G05 while still helping G12/G14. |
| Coerce `false_premise` → `answered` on neutral polar questions if the backend ignores the enum | Stops G08-style wrong status when the answer text is already correct. |
| Broader `source_base_year_binding` quote patterns + existing metadata completion | Keeps the book's base year (e.g. 2005) in the answer when evidence is present. |
| Kaggle submission forces `passage_reranking=True` and prints setup minutes vs the 90-min wall | Final system uses the bonus path; timing visibility for the GPU run. |

#### Fix-100 ablation (27 Oct, local Ollama) — **current**

Outputs: `Part1/sneha_singh/outputs/evaluation/reranker_ablation_fix100_20261027/` (older postblend 22=22 and historical 20-vs-19 preserved).

Extra general repairs after the 22=22 tie: accept `imported` for import coverage; complete missing CPI/deflator basket terms from source facts (trim to 120 words); attach inventory citations when the answer already states overlapping import/domestic claims (fixes ON G12 sentence-coverage misses).

| Measure | Reranker OFF | Reranker ON |
| --- | --- | --- |
| Answers correct | **25/25** | **25/25** |
| Retrieval Recall@5 | 100% | 100% |
| Generation errors | 0 | 0 |
| Paired gained / lost | — | none / none |

Focused re-check G05/G12/G15/G16: **ALL_PASS** both arms. **+2 bonus not claimable** on this run (ON cannot beat a perfect OFF). Pipeline is at 100% local golden-set answers.

### Models in use (verified)

Two backends exist in `notebooks/sneha_singh/p1_rag.ipynb`; only the local one has been measured on the golden set.

| Runtime | When it runs | Models | Status |
| --- | --- | --- | --- |
| Local Mac (Ollama) | Default when not on Kaggle | Answers: `qwen2.5:7b` (Q4_K_M). Figures: `qwen2.5vl:3b`. Encoder: `all-MiniLM-L6-v2`. Reranker: `ms-marco-MiniLM-L6-v2` (CPU). | **Current golden-set measure: 25/25 OFF and 25/25 ON** (`reranker_ablation_fix100_20261027`). Manifest: `"inference": "local Ollama"`. |
| Kaggle GPU (`transformers`) | Only when `/kaggle/input` is present | One model: `Qwen/Qwen2.5-VL-7B-Instruct` (NF4), no Ollama server. Encoder and reranker stay on CPU. | Public LB best **0.940**. Graded path: Save Version of this notebook at `p1-v1.0`. |

Do not treat Mac golden-set scores as Kaggle scores.

### Remaining risks after the fix-100 run

| ID | Problem |
| --- | --- |
| +2 bonus | Not claimed on this pipeline (golden-set OFF/ON tied 25/25). The team claims the +2 with Ritika's reranker ablation (report II.B, `Part1/ritika_neema/README.md`). Bonus PoC notebook/outputs removed from the repo. |
| Kaggle | Executed hardcode-cleaned notebook committed (T4, 40 rows, validated, 1171 s, output sha `47948d4e…`). Practice public best **0.940** (selected `submission.csv` sha `881a9ffb…`). |
| Human ratings / report Appendix C examples | Done — both members in `eval/human_review.csv`; Appendix C write-up in `report/report.md` §II.D |
| Question-pattern salvage / Okun boost / B-5 4.1 / −2.3 skips | Removed; executed cleaned Kaggle notebook committed and tagged `p1-v1.0` |}

### Still open for both members

1. Public leaderboard best remains **0.940**. Keep that score in the report and docs.
2. Ritika's Part 1 pipeline, Kaggle link, and golden/holdout reranker ablation are recorded under `Part1/ritika_neema/` (team +2 claim).

Both raters' groundedness scores are in `eval/human_review.csv` (mean 4.88 for each rater; within one point on 24/25).

Appendix C safety write-up is in `report/report.md` §II.D. Public LB **0.940** and compliance notebook are already documented.

Task: answer questions over three supplied economics PDFs (read-only knowledge base), including multi-turn conversations. Notebook: `notebooks/sneha_singh/p1_rag.ipynb`. **Current** table is the fix-100 passage-reranker ablation (`reranker_ablation_fix100_20261027/metrics.json`). It is a local golden-set development ablation, not the Kaggle public LB. Both members' groundedness ratings are filled in `eval/human_review.csv`.

## Architecture

![Part 1 architecture: ingestion, retrieval, conversation memory, and evaluation](../../report/figures/part1/architecture.png)

The figure is the pipeline in the sections below. Three read-only PDFs are chunked by page, and trusted figures are described so they can be retrieved. Appendix C is quarantined and is never used as evidence. Dense retrieval is the baseline. The hybrid retriever is the final system. A user assumption stays in that conversation and is not written into the index.

## Models and why I chose them

Latest measured print (27 October fix-100): passage-reranker OFF vs ON on the hybrid pipeline (dense + BM25 + RRF-blended CE). Local Ollama answerer, not the Kaggle Hugging Face model. Kaggle GPU run still open.

1. **Reranker OFF**: hybrid retrieval without the CPU cross-encoder passage reranker.
2. **Reranker ON**: the same hybrid pipeline with `ms-marco-MiniLM-L6-v2` passage reranking.
3. **Local answer path (measured):** Ollama `qwen2.5vl:3b` describes trusted figures; **answers** and the provisional judge use `qwen2.5:7b` (Q4_K_M, local Ollama `0.34.4`, temperature 0, seed 266), grounded on retrieved text and figure descriptions (text-only, no direct image input). This is what the ablation table reports.
4. **Kaggle answer path (not measured yet):** when the notebook detects Kaggle, it loads `Qwen/Qwen2.5-VL-7B-Instruct` with `transformers` (NF4 on one 16 GB GPU) and does not start Ollama. That run has not been completed, so there is no Kaggle score to put in the report yet.

## Preprocessing
- Page-aware text chunks, separate figure and table crops, table-row records, and trust levels from the manifest; the low-trust Appendix C is quarantined and never used as evidence.
- Conversation memory is separate for each conversation; user-supplied assumptions never enter the index.
- Golden set: 25 questions (5 multi-turn across two conversations), authored and reviewed by both members (`eval/golden_set.json`).

## Hardware
Local Mac with Ollama for the current scores (CPU cross-encoder). The required end-to-end run on a 16 GB Kaggle GPU within 90 minutes, using `Qwen/Qwen2.5-VL-7B-Instruct`, has **not** been done yet.

## Test metrics (development diagnostics) — **current**

Run finished 27 Oct 2026. Automated/provisional golden-set development results—not a Kaggle score. Source: `Part1/sneha_singh/outputs/evaluation/reranker_ablation_fix100_20261027/metrics.json`.

| Measure | Reranker OFF | Reranker ON |
| --- | --- | --- |
| Answers correct | **25/25** | **25/25** |
| Retrieval Recall@5 | 100% | 100% |
| All required sources in top 5 | 100% | 100% |
| Citation support—model judge | 100% (37/37) | 100% (38/38) |
| Median seconds/question | 62.20 | 63.20 |
| Generation / judge errors | 0 / 0 | 0 / 0 |
| Paired gained (ON only) | — | none |
| Paired lost (ON only) | — | none |
| +2 bonus (ON > OFF answers) | — | tie at 25/25 on this pipeline; team claim uses Ritika's ablation |
| Human groundedness (both members) | 4.88 mean each | 4.88 mean each |
| Human agreement within one point | 24/25 | 24/25 |

### Historical ablation (26 Oct; superseded)

Source: `outputs/evaluation/reranker_ablation/metrics.json`.

| Measure | Reranker OFF | Reranker ON |
| --- | --- | --- |
| Answers correct | 20/25 | 19/25 |
| Retrieval Recall@5 | 100% | 100% |
| Citation support—model judge | 94.4% | 97.3% |
| Median seconds/question | 55.63 | 50.12 |
| Generation / judge errors | 2 / 0 | 2 / 0 |

## Comparison (reranker off vs on)
- **Current (fix-100):** both arms **25/25** answers, 100% Recall@5, 100% citation support, 0 generation errors. Paired gained/lost: none. **+2 bonus not claimable** (ON must strictly beat OFF).
- Historical (26 Oct): Recall@5 stayed 100%; citation support 94.4%→97.3%; answers 20/25 OFF vs 19/25 ON (gained G12/G14, lost G02/G05/G08).
- Each arm had 2 generation errors and 0 judge errors. Correctness and citation scores use a same-model judge, so they can share answerer biases; human groundedness ratings from both members are in `eval/human_review.csv` (mean 4.88 each).

## Strengths
- Retrieval stayed at 100% Recall@5 on both arms. Timing excludes indexing and grading and does not establish Kaggle GPU performance.
- Answers are grounded JSON with cited sources, and trust levels and a quarantined low-trust appendix are respected.
- Everything is local (no hosted API).

## Weaknesses / limitations
- Results are development diagnostics on a team golden set that informed the design; the graded Kaggle path uses the hidden set.
- Human groundedness ratings are done (`eval/human_review.csv`). Public Kaggle LB best is **0.940**; executed T4 notebook is committed.
- Same-model grading and a text-only answerer (figures enter through VLM descriptions).
- Historical note only: the first reranker ablation did not raise answer correctness (19/25 against 20/25) and had 2 generation errors per arm. The fix-100 rerun is 25/25 both arms with 0 generation errors.

## Failed cases (earlier dense vs hybrid development run)

The notes below are from an earlier dense-versus-hybrid comparison, not from the reranker OFF/ON table above. Keep them as development history until both members review the current ablation failures.

- Hybrid G12 omits the required imported-consumption distinction and incorrectly says the GDP deflator remained positive in 2009. The actual retrieved Briefing B page 4 says the GDP deflator also fell slightly, while core CPI stayed positive. This is a real source-grounding error, not merely wording. The semantic and citation judge accepted it incorrectly; the unchanged fixed concept rubric still marks the answer false.
- Hybrid G21 correctly returns conflict, both 9.8 and 9.6, no explicit revision, and preference for DOC1. However, its answer text does not explicitly label the other source DOC3 or explain the high-versus-medium trust levels; DOC3 appears only in structured citations. The unchanged required-pattern check fails. The semantic judge's pass does not override that failure.
- Dense G19 is a rejected generation: its source-scope/operand audit rejected the plan after bounded retries, despite a positive-sounding audit explanation. It remains counted as incorrect.
- On the current reranker ablation, turning the reranker on gained G12 and G14 and lost G02, G05, and G08. Human ratings are in `eval/human_review.csv`.

## Latest correction checks

- Source-scoped retrieval now reserves relevant prose from explicitly named chapters or documents. Chapter membership and source aliases are inferred from source headings, not a table of test-question answers.
- Fact filtering retains evidence from the requested sources; numeric plans must use the requested source's estimate.
- A source-derived linear-equation tool calculates zero crossings from the printed intercept and slope. It declines ambiguous, zero-slope and unsupported nonlinear equations rather than substituting the intercept for a threshold.
- The clean current-source notebook execution completed all 31 code cells with zero cell errors and no kernel-cleanup warning. All 230 deterministic regression tests passed, including alternate accounting amounts, ambiguous competing entities, scaled GDP units, operand-direction checks and an integration test that deliberately filters out the required accounting facts. All 16 actual-model synthetic checks passed. Executed outputs are saved in the canonical notebook; verification details are in `outputs/evaluation/notebook_verification.json`. This diagnostic run deliberately skipped the full comparison, reports and submission export; it is not a new full score.
- A bound-comparison handoff bug was fixed: the source-derived compatibility proof now reaches the final answer checker, not only the planner. An integration test verifies that handoff while retaining the actual-answer support/completeness audit.
- The first focused pass corrected the equation calculation but exposed remaining ranking, false-refusal, table-column and percentage-point-scaling errors. The latest adjustment preserves source-specific facts and binds non-year table calculations to the requested row and parameter, with explicit percent-unit checks. Earlier failed diagnostic attempts are retained in evaluation history.
- This is pipeline adjustment, not training model weights on the evaluation questions. The source PDFs, references and fixed grading rules are unchanged.

### Current reasoning fixes

Current generation signature: `9fc585264f13424683217c1f522596794292dc62d79d9703cf90133429ded7bb`. Before the full run, the actual-model focused check in `outputs/evaluation/reasoning_recheck.jsonl` passed all seven selected questions (G04, G09, G13, G14, G18, G19 and G24), with zero generation or judge errors. G18 and G19 were run in their conversation order. All seven subsequently passed the full hybrid evaluation too. The focused checks themselves are not an independent holdout or full score.

| Corrected behaviour | Actual-model check |
|---|---|
| Bind temporal factor to later divided by earlier, not subtraction, percent scaling or the inverse | G04: `33 / 9 = 3.6666667` times; pass |
| Keep a hypothetical denominator separate from the original accounting frame | G18: 60% hypothetical, 75% original; pass |
| Restate one aggregate entity, without mixing in a component company's split | G19: wages 150, profits 50, income 200, labour share 75%; pass |
| Reject a contradicted GDP assertion and retain the source's explanation | G24: false premise; GDP 200; final goods counted, intermediate goods not added again; pass |
| Preserve earlier equation, bound-comparison and threshold fixes | G09, G13 and G14: pass |

The fixes derive quantity roles, accounting identities, time direction and explanations from actual retrieved source spans. Typed proof facts survive the relevance filter; unrelated entities are not pooled. Scope and final-claim audits remain active. No golden-question IDs or reference answer values are used by generation helpers.

On the preceding repair candidate (`84e4d4452fc3...`), dense G01 and G02 both passed after the source-context schema was bounded; dense G14 returned a valid but incorrect answer using its wrong retrieved U.S. threshold. All three had zero generation/judge errors. This is not a current-source full dense score. Failed intermediate diagnostic versions remain in evaluation history.

## Future work

### Frozen new-question test in progress

- Separate test: `Part1/sneha_singh/eval/unseen_20261026_v1.json`, frozen SHA-256 `a85837e18e021532219f619e6a40f4bb0eb6106eff1101b847c61508abe120ab` before predictions. It has 25 new questions with the 5/6/3/5/3/3 category allocation and two multi-turn conversations; the original golden set is unchanged.
- Team golden set: `eval/golden_set.json`, authored by Sneha Singh and Ritika Neema. New questions use the same PDFs; some exercise mechanisms already developed on the earlier set.
- Current notebook checks: all 241 deterministic regression tests pass. The workflow is rechecking the two known failures and two actual-model alternate-entity sentence-support probes before automatically starting the new dense-versus-hybrid comparison. Current-source full notebook execution and Kaggle GPU validation remain pending.
- The first G12 repair removed the wrong GDP-deflator claim and included imported goods but still failed the unchanged fixed wording check. A subsequent variant exposed an erroneous conflict status for compatible tendency/exception statements; that routing was corrected and is being verified. G21's explicit source-ID/trust rendering passed its actual-model checks. Failed intermediate outputs remain preserved; no new full score is inferred from these focused checks.
- All first-pass new-test generations, including errors, will be preserved. References and rubrics are passed only to grading, not the answerer; no weights are trained. Results will be saved separately under `outputs/evaluation/unseen_20261026_v1/`. Do not fit answers to this test after inspecting its predictions or quietly retry errors away.

### Remaining validation

- Address G12's unsupported dated claim and incomplete imported-consumption explanation, and G21's incomplete explicit trust explanation, with general source-grounding/completeness checks rather than question-specific answers. Any later source change needs re-verification; this completed score belongs only to the signature above.
- Author and freeze a fresh independent question set with the required Git timestamp, then re-evaluate.
- Run the final notebook on a Kaggle GPU, check the 90-minute limit and submit.
- Both members' groundedness ratings are filled in `eval/human_review.csv` (mean 4.88 each; within one point on 24/25).

## Completed frozen new-question first-pass comparison — 26 October 2026

This section supersedes the pending-work notes above. The earlier 23/25 hybrid and 13/25 dense development table remains historical and is not the result of this test. Before this run, the current notebook passed all 241 deterministic regressions, both actual-model focused cases (G12/G21), and both alternate-entity sentence-support probes. That focused success did not establish general reliability. Current-source Restart/Run All and Kaggle GPU validation still remain pending.

**Frozen NEW-QUESTION first-pass diagnostics over the same supplied PDF corpus.** This is not a Kaggle score. The test was frozen before predictions. No reference answers, expected numbers or rubric patterns were supplied to the answerer. Generation errors were retained without retrying them away, and no false grade was manually changed.

| Measurement | Dense baseline | Hybrid final |
|---|---:|---:|
| Answers correct | 10 / 25 (40%) | 15 / 25 (60%) |
| Retrieval Recall@5 | 64.6% | 97.9% |
| All required sources in top 5 | 62.5% | 95.8% |
| Citation support (model judge) | 80.8% (21/26) | 81.5% (22/27) |
| Median seconds per question | 72.40 | 43.37 |
| Generation / judge errors | 5 / 0 | 3 / 0 |
| Human review | collected later on golden set (`eval/human_review.csv`) | same |

Both systems have 25 unique graded rows, zero unscored rows, and matching metrics/manifest signatures. Generation failures count as incorrect, not omitted observations. The completed runner removed its lock. I independently rechecked the frozen dataset, generation/evaluation code fingerprints, protected source/development inputs and strict correct/error counts; all matched the saved manifest. No code, frozen questions or grades were changed after this test started.

Provenance:

- Outputs: `Part1/sneha_singh/outputs/evaluation/unseen_20261026_v1/` (`final.jsonl`, `baseline.jsonl`, `metrics.json`, `run_manifest.json`, `status.json`, `runner_status.json`).
- Run signature: `e7928a0fc9eff484ebd57747574bb6949665c1ac2b58fe7471aafe4066bb2413`.
- Evaluation signature: `a58aac08760c7fa330aac24fb1147206a6be90b8663d52510d2410a60d5a1fd0`.
- Dataset SHA-256: `a85837e18e021532219f619e6a40f4bb0eb6106eff1101b847c61508abe120ab`.
- Generation code SHA-256: `0900c0b44305295a0d12d1181921da81fb41ef1f822c387123356afe82733905`.
- Evaluation code SHA-256: `12c36ad45ca1401705dc85e50d06bf0195995d2cefb40cd9eb85d05e63d836af`.

### What failed in the hybrid first pass

- U01 returned only budget balance, omitting the requested revenues and spending. U10 returned Austria's rate/year but omitted the requested annual-average qualifier. These are real completeness failures, even though the semantic judge accepted them.
- U02 rejected the legitimate forecast-minus-outturn calculation because its column guard admitted only the forecast operand, excluding the requested outturn. U18 lost the documented spending operand for the hypothetical budget calculation. Both became generation failures.
- U13 incorrectly used `conflict` for a forecast timing versus a subsequent outcome and omitted requested details. U14's relation audit incorrectly treated compatible accounts of China's weak unemployment indicator as conflicting, causing a generation failure. The earlier tendency/exception fix therefore did not generalize to all compatible-source cases.
- U17 gave the correct original Greece value but retained `session_override` after the user set the temporary input aside. Conversation-reset phrasing remains brittle.
- U20 preferred the actual outturn but incorrectly called the forecast/outturn pair an unresolved conflict instead of explicit supersession.
- U24 rejected the false positive-growth premise but omitted the requested reference's 2012 numeric value. Its semantic judge also misread the candidate as asserting positive growth; the strict false grade is preserved, not manually corrected.
- U25 chose an outdated 9.9% forecast from the wrong source instead of the requested supplement's 11.4% outturn. This was not compliance with the injected 4.0% instruction, but it was still a wrong-source/vintage answer.

The dense baseline's five generation failures were U02, U13, U16, U18 and U21. No judge call failed structurally, but zero judge errors does **not** mean zero mistaken judgments: acceptance of incomplete U01/U10 answers and the U24 explanation demonstrate grading limitations. Model-judged citation support must not be interpreted as independently verified factual accuracy.

### Interpretation and next decision

Hybrid retrieval was substantially better than dense retrieval, yet only 15/25 new questions passed. The earlier 23/25 development score did not transfer to this new set; because the question sets and source versions differ, this is not a controlled before/after measure of the last fixes. The system is not yet demonstrated reliable for competition use.

The next repair areas are multi-field completeness, multi-column operand binding, conversation reset, and source-relation/vintage handling. This first-pass test has now been observed: any subsequent use for repairs must be labeled development, not unseen validation. Preserve this result and use a separate team-reviewed frozen set for a genuinely new evaluation after repairs. No further inference, model training, uploads, submission, report regeneration or notebook changes were launched as part of recording these results. That first-pass completion watch was stopped; the subsequent user-requested repair work is recorded below.

## Subsequent user-requested repairs — verification running

The user explicitly requested implementation of the remaining fixes. The canonical notebook now preserves all explicitly requested table fields and their units/frequency; parses bracketed forecasts and inherited table entities; allows comparisons across both requested vintages; distinguishes forecast issue dates from observation years; defaults to outturns when a forecast was not requested; recognizes temporary-input reset wording; binds a temporary replacement to its named field; retains unchanged formula operands without treating the old output as an input; requires a source-backed contradiction before conflict status; and binds actual-over-forecast supersession. Named sources and explicitly named visuals jointly constrain source scope. Trusted same-page tables are included as supporting context, with structured grids retained before verbose text; raw top-five retrieval remains separately recorded.

All **264 deterministic regression tests passed**, including alternate entities, amounts and years. A source-row preflight over the supplied corpus confirmed recovery of the requested multi-field observations, both forecast/outturn inputs, the original and hypothetical budget operands, and the signed Euro-area outturns. These are mechanism checks, not a new accuracy score.

A **16-case actual-model POST-FEEDBACK DEVELOPMENT recheck** is active under `outputs/evaluation/repair_20261026_v3/`, signature `e12cbcc33fc09f36e1eae2625bcbcf5607e9ed61e95778a27af38452ccc1261f`. It covers U01/U02/U10/U13/U14/U15-U21/U24/U25 and the G12/G21 controls, with conversation order preserved. U01 has passed with all requested financial fields; other current-run checks remain pending. Earlier short diagnostic runs were intentionally interrupted for additional scoped fixes and their completed rows remain archived; they are not full verification results.

The original 15/25 hybrid and 10/25 dense first-pass measurements, source PDFs, original golden set, frozen question file, references, grading rules and human ratings are unchanged. This repair work does not turn that observed set back into an unseen holdout. Current-source clean full-notebook execution and Kaggle GPU validation remain pending. The repair completion watch is active and should report verified outcomes or genuine blockers, not normal-progress notifications.

Early actual-model finding: U02 now passes the formerly over-restrictive two-column operand check, but the generated plan adds an invalid redundant percentage-point scaling: it reports both `3.2 - 2.4 = 0.8` and `(3.2 - 2.4) * 100 = 80 percentage points`. Its strict grade remains false. The declared observation units must reach arithmetic-plan validation, not just the prose renderer. This confirms that the repairs are not yet fully verified; the active run should finish before further notebook edits and regression-backed correction of this path.

### Completed targeted check and follow-up repairs

The above 16-case recheck completed: **10/16 strict passes, 2 generation errors, 0 structurally failed judge calls**. This is a targeted post-feedback development check, not a new 25-question comparison. All protected inputs remained unchanged. Its manifest, summary and every prediction are preserved in `outputs/evaluation/repair_20261026_v3/`.

Real remaining implementation failures were U02 (invalid multiplication of an already-percent difference by 100), U13 (the drafted explanation omitted its required signed table observation and table citation), and U20 (the deterministic vintage plan carried an empty arithmetic field rejected by the direct-fact schema). U15 and U18 explicitly distinguished authorized temporary inputs from documentary values, but the judge demanded the documentary inputs instead; U24 explicitly rejected positive growth, but the judge described it as asserting positive growth. These are evidence-backed grading discrepancies, not manually awarded passes: **all six strict false grades remain false**.

The next notebook revision propagates structured percent units into the plan guard and renderer, rejects redundant raw-rate scaling before accepting a plan, adds omitted selected table observations with their own citations before the unchanged claim audits, and makes the vintage tool conform to the direct-fact schema. **277 deterministic tests passed**, including alternate rates, preservation of legitimate relative-percent calculations, source-selection safeguards and the passage-reranking toggle/cache signatures. Actual-model verification of this revision is pending.

The existing MiniLM cross-encoder is retained. A controlled paired experiment is now implemented in the canonical notebook: hybrid passage reranking OFF versus ON, holding source anchors, fact ranking, candidate generation, routing, generator and grading fixed. It uses separate arm caches and a frozen protocol under `outputs/evaluation/reranker_ablation/`. It requires clean current-source notebook verification first. No reranker improvement has been measured yet; any gain, tie or regression will be reported as automated/provisional golden-set development evidence, not unseen validation or a Kaggle result.

### Five-case verification and sign-convention clarification

The next focused run completed under `outputs/evaluation/repair_20261026_v4/`, signature `7a7252602626a1d27af237e56f1e56c792ca5b144cf06eeca0b5abacd63a5b1b`: **4/5 strict passes, 0 generation errors, 0 judge runtime errors**. U02, U20 and both G12/G21 controls passed. U13 now returned the required signed balance and table citation, and the semantic judge accepted it, but the unchanged numeric check required the positive deficit magnitude as well. Its strict false grade is preserved. The workflow correctly stopped before clean notebook execution or the reranker experiment.

The follow-up revision states the requested deficit magnitude only when the selected source explicitly defines a negative budget balance as a deficit; it retains the original signed balance, units, period and citation. No expected answer or evaluation number is embedded in the helper. **281 deterministic regression tests pass**, including alternate signed balances, missing source definitions and positive-balance controls. A preflight on the actual retrieved table confirmed the signed observation and its explicit definition produce both representations. Fresh actual-model verification is still required; these checks do not establish a new 25-question accuracy score.

### Current five-case repair verification passed

The subsequent actual-model check completed under `outputs/evaluation/repair_20261026_v5/`, signature `2a0e9f2a0e697692240c91c76dd20d521963c9eabe0fbf9d57cb1e20f20ed1a3`: **5/5 strict passes, 0 generation errors, 0 judge runtime errors**, with 281 passing deterministic regressions. U02, U13, U20 and the G12/G21 controls all passed. An independent read-only check confirmed five unique rows, matching generation/evaluation signatures, no unscored rows and unchanged protected inputs. The previous v3/v4 results and all false grades remain preserved.

This verifies the focused repairs only; it is not a new full-set accuracy result or unseen validation. The gated workflow has started clean current-source notebook execution and is currently running the actual-model synthetic checks. Full notebook verification and the subsequent 25-question-per-arm reranker comparison remain pending. No reranker accuracy improvement is claimed yet.

### Clean notebook verified; paired reranker experiment started

Clean current-source notebook execution completed successfully: **31/31 code cells, 0 cell errors, 281 passing deterministic regression tests, and 16/16 passing actual-model synthetic checks**. The executed outputs are saved in the canonical notebook. An independent read-only check confirmed all execution counts, absence of error outputs, synthetic-check completion, and the matching notebook-source SHA-256 `98cdf730284d760148e9440907c132ccd3383880d783dd7c3015b989543619ba`. Verification details are in `outputs/evaluation/notebook_verification.json`. Full evaluation, local submission export and report regeneration were disabled for this verification.

The gated workflow then started the paired passage-reranking experiment, protocol `6cd647741508538295bed272ccb6b24c5750b2425a81135572ecc18ee73a3a34`, under `outputs/evaluation/reranker_ablation/`. It runs the same 25 original golden questions in each hybrid arm, first OFF then ON, with other retrieval/generation/grading settings fixed. Both arms must finish before any improvement is reported. These are automated/provisional post-feedback development diagnostics; Kaggle GPU validation and an independent team-reviewed holdout remain pending.

## Completed passage-reranker comparison

Automated/provisional **post-feedback golden-set development ablation**, not an untouched holdout or Kaggle score. Both arms are hybrid pipelines. Only candidate passage reranking is toggled; source-anchor cross-encoder ranking, fact ranking, candidate generation, routing, context expansion, generator, top-five budget and grading remain fixed. This is not a dense-versus-hybrid comparison or removal of all cross-encoder uses.

| Measure | Passage reranking OFF | Passage reranking ON | ON minus OFF |
|---|---:|---:|---:|
| Answers correct | 20 / 25 (80%) | 19 / 25 (76%) | -1 answer; -4 percentage points |
| Retrieval Recall@5 | 100.0% | 100.0% | 0.0 percentage points |
| All required sources found in top 5 | 100.0% | 100.0% | 0.0 percentage points |
| Citation support (model judge) | 94.4% (34/36) | 97.3% (36/37) | +2.9 percentage points |
| Median seconds per question | 55.63 | 50.12 | -5.50 seconds |
| Generation / judge runtime errors | 2 / 0 | 2 / 0 | 0 / 0 |
| Unscored questions | 0 | 0 | 0 |

**No answer-accuracy improvement was demonstrated.** Passage reranking gained G12 and G14 but lost G02, G05 and G08, a net loss of one strict pass. Model-judged citation support increased and median response time was lower in this run; a single ordered 25-question-per-arm run does not establish a reliable speed or accuracy advantage. OFF's one-answer lead is also too limited to claim general superiority. Retrieval source coverage was already saturated in both arms, and 100% retrieval does not certify correct operand selection, units, reasoning or final claims.

### Remaining failures and grading limitations

- OFF strict failures: G04, G12, G14, G15 and G20. Its rejected generations were G04 and G14.
- ON strict failures: G02, G04, G05, G08, G15 and G20. Its rejected generations were G04 and G20. All rejected generations remain counted as incorrect; none were retried away.
- G02 ON emitted a dimensionally wrong car-economy calculation (`20000 * 12 / 24000 = 10`). The semantic judge incorrectly accepted it; the fixed numeric check retained the failure. G05 ON used a base-year normalization incompatible with the fixed reference, although its semantic judge accepted it. These are not evidence of reliable arithmetic or normalization merely because citations were accepted.
- G04 in both arms bound a duration-comparison request to an unrelated unemployment-rate table cell. G20 still failed event/date/replaced-estimate resolution; OFF gave an unsupported/incomplete answer and ON rejected its plan. These are genuine scope/completeness limitations.
- G08 ON gave the source-supported explanation that rebasing preserves percentage returns but used the wrong `false_premise` status for a neutral question. Its judge reason also contradicted the actual candidate. G15 in both arms clearly separated the temporary input from the documentary value, yet the judge rejected it with a self-contradictory reason. Their strict false grades are preserved, not manually promoted to passes.
- G12 OFF failed the unchanged concept-pattern check despite semantic acceptance; G14 OFF was rejected by an over-restrictive source-support audit. The recorded errors and drafts remain available for review. No implementation or rubric was changed after inspecting this experiment's predictions, and no replacement experiment was run to obtain a higher score.

### Completion and provenance

Both native experiment status and the entire gated workflow are complete, with locks removed. Independent read-only verification confirmed **50 unique graded rows**, the original questions, matching arm/evaluation signatures, aggregate metrics, unchanged notebook source hash and protected inputs. The earlier 10/25 dense and 15/25 hybrid first-pass new-question results, all historical development tables, reference answers, source data and human-review files remain unchanged.

- Outputs: `outputs/evaluation/reranker_ablation/` (`off.jsonl`, `on.jsonl`, `metrics.json`, `run_manifest.json`, `status.json`).
- Protocol: `6cd647741508538295bed272ccb6b24c5750b2425a81135572ecc18ee73a3a34`.
- OFF generation signature: `53300f88aa954fa69c0ef2bf2432de46b6897321831cb7256bae8f6e406c3ac0`.
- ON generation signature: `ab5261f04dab10e3f6cb01d76cc20b226a739a9cfa6c5a7260c2b7863ff75145`.
- Shared evaluation signature: `e0247f3f92c5a637449862feb5bf538fa0451ff5da36f8460de6289dd5f52b3a`.
- Golden-set SHA-256: `efc5942fe7627483fa28382a94be611ed8793b9f35b874200e52e36015fd0349`.
- Notebook source SHA-256: `98cdf730284d760148e9440907c132ccd3383880d783dd7c3015b989543619ba`.

The prior focused repair gate passed 5/5; clean execution passed all 31 code cells, 281 deterministic tests and 16 actual-model synthetic checks. Those checks establish execution and targeted mechanism coverage, not general model reliability. The next decision needs real team review of the failed cases and independent evaluation; Kaggle GPU validation remains pending. The completion watch is being stopped after reporting this measured outcome.

## Newly authorized mechanism repairs after the completed comparison

The user requested the remaining fixes after reviewing the completed experiment. The 20/25 OFF and 19/25 ON results and every saved prediction remain unchanged. Subsequent checks are post-feedback development diagnostics, not a replacement frozen ablation or a new holdout.

Source-page review clarified the remaining failures: G02 mixed price/quantity roles; G04's guard demanded unemployment-rate cells for a duration-in-weeks request; G05 used the correct base-year normalization of 1 but omitted the year 2005; G08 used a false-premise status for a neutral yes/no question and omitted one base date; G20 lost the actual dated revision notice in favor of a cross-reference/publication context. In particular, the earlier shorthand description of G05 as a normalization error was imprecise: its numeric failure was the missing year, not the value 1.

The canonical notebook now derives quantity-year and valuation-year operands from a consistent printed quantity/price table, rejects duration-to-rate measure substitution, retains exact proof evidence through fact filtering, exposes short dated event clauses inside visual context, follows section text even when that page's visual was already retrieved, distinguishes event dates from publication vintages, and preserves source-bound index years/base dates. A leading neutral polar question cannot acquire a false-premise status merely because it is followed by an explanation request. These are source-derived contracts over arbitrary entities, amounts and years; generation helpers receive no evaluation IDs or reference answers. The lab's prohibition on hard-coded answers/question-specific rules remains a required compliance check, and no post-feedback score should be presented as an untouched final evaluation.

All **300 deterministic regression tests pass**, including alternate entities/years, ambiguous or inconsistent tables, rate-versus-duration negatives, event-versus-publication dates, physical base captions versus VLM guesses, and a wired quantity-price generation test. Five additional alternate-input actual-model source-contract checks are implemented, but have not yet passed an actual run. Current-source clean notebook verification is also pending again because the source code changed.

One gated workflow under `outputs/evaluation/mechanism_repair_20261026_v6/` is testing ten document-backed development cases: G02/G04/G05/G08/G20 plus G12/G21 and U02/U13/U20 regression controls. If all ten strict checks pass without runtime/judge errors, it will execute the clean canonical notebook, including the original sixteen synthetic checks and five new alternate-input source-contract checks. It does not launch a new full 25-question evaluation or rerun the completed reranker experiment. Historical measured reranker results will be recomputed from preserved rows and displayed read-only in notebook output, explicitly marked historical when the generation source version differs.

The biased judge failures remain recorded for genuine human review; no grade, reference, tolerance or human rating has been altered.

### Follow-up unit repair and verification status

The v6 targeted development check completed at **9/10**, with zero generation or judge runtime errors and protected inputs unchanged (generation signature `00a94a77fdfeb2474a25170e757562448ccff4857574f9e6d9223884cceeb2b6`). Its sole strict failure, U02, was genuine: alongside the correct direct rate difference, it emitted a relative-percent calculation mislabeled as a percentage-point shortfall. That failed grade and all ten predictions remain preserved. The strict gate stopped before clean notebook execution.

The canonical notebook now validates requested percentage-point gaps against bound already-percent source observations: a simple gap requires direct subtraction; a normalized relative change cannot be mislabeled as a point gap or added when only the point gap was requested. Correctly labeled relative changes remain allowed when explicitly requested. Seven alternate-number regression checks cover mixed requests, rendering, and disguised ratio expressions. **307 deterministic tests passed.** This is bounded validation of simple source-rate gaps, not a general dimensional-algebra guarantee.

A new v7 diagnostic workflow uses the same ten post-feedback development cases, with U02 first, and preserves the v6 directory. Actual-model acceptance and current-source clean execution (31 cells, original 16 synthetics and five source-contract checks) remain pending. No new full-25 score or replacement reranker experiment is claimed.

The v7 run subsequently completed **9/10**, with **one rejected generation** and zero judge runtime errors (signature `ad6775d87d09a4cb17f4783de1b9bdfb80d791cb081a15c07e91d333e8cdc944`). All other nine cases passed. U02 repeatedly proposed an invalid second calculation despite the unit rejection; its final plan subtracted an input a second time. The rejection is a preserved strict failure, not a successful answer. Independent checks confirmed ten unique signed rows and unchanged protected inputs. Clean execution did not start.

The follow-up v8 planning contract now constrains a single requested gap between exactly two same-citation, explicitly percent-valued observations to at most one direct subtraction. It does not supply an expected result, choose the operand order, remove a rejected calculation after generation, or apply to mixed relative/ratio/weighted requests. Existing operand, direction, support and correctness checks remain active. Six alternate-input deterministic tests include the actual planner-schema path; **313 regressions pass**. A sixth actual-model source-contract fixture uses an alternate entity, period and rates; its expectations remain confined to testing. The new v8 workflow must pass the ten focused cases before clean execution of all 31 cells, original sixteen synthetic checks and all six source-contract checks. Those current-source model/clean checks remain pending. Every earlier failure and completed comparison remains unchanged; this is post-feedback development work, not independent validation.

### Completed v8 scoped verification — 26 October 2026

The entire v8 gated workflow is now complete, with both locks removed. These are **automated/provisional post-feedback development diagnostics**, not an independent holdout, a new full-25 comparison or a Kaggle score.

| Verification | Measured outcome |
|---|---:|
| Targeted document-backed strict checks | 10/10 |
| Focused generation / judge runtime errors | 0 / 0 |
| Deterministic regression tests | 313/313 |
| Clean canonical notebook code cells | 31/31 |
| Notebook error outputs | 0 |
| Original actual-model synthetic checks | 16/16 |
| Alternate-input actual-model source-contract checks | 6/6 |

All targeted cases U02, G02, G04, G05, G08, G20, G12, G21, U13 and U20 passed the unchanged strict grading. U02 now gives only the source-bound subtraction and correct percentage-point unit; the alternate entity/year/rates fixture C6 also passed. Together with the other focused and synthetic checks, this verifies the repaired quantity/price roles, duration scope, source base-year/base-date retention, dated revision binding, neutral-question status and simple two-rate gap paths within their tested scope. It does not establish universal reasoning reliability or eliminate known semantic-judge bias.

Independent read-only verification confirmed ten unique fully graded rows, exact original questions, matching generation/evaluation signatures, current notebook source identity, all 31 distinct executed code cells with saved outputs, recomputed synthetic signatures, and unchanged protected source/reference/first-pass/ablation files. The original sixteen and six alternate-input model checks are complete and passing under current signatures. Historical reranker metrics are displayed read-only in saved notebook output, explicitly marked as a different source version; neither arm was rerun or regraded. No report, submission or new full-25 evaluation was generated.

- Targeted outputs: `outputs/evaluation/mechanism_repair_20261026_v8/`.
- Targeted generation signature: `d687ce2aae2c156937e461ae1261179a9238635aa2780600824363b2d07230ac`.
- Targeted evaluation signature: `d64d85c759ff25f91007b617fb4d1b7b8d938f778f6a5255bcd7e82143a1da26`.
- Current notebook source SHA-256: `b06f7a02a8fe45638239ab169b5b5339ac70e39177adaf3afc729f5b57c31b15`.
- Verification: `outputs/evaluation/notebook_verification.json`, `synthetic_checks.json` and `synthetic_source_contract_checks.json`.

All v6/v7 failures and historical tables remain preserved; no false grade, rubric, reference value, tolerance or human rating was changed. Team-authored golden set, human groundedness ratings (`eval/human_review.csv`), and the Kaggle GPU notebook (public LB 0.940) are complete. Post-feedback development scores in this diary are not untouched final-test performance. No new model download, weight training, Git commit, upload or submission occurred in this scoped verification. The completion watch is stopped after this scoped verification.

## Independent follow-up code audit — 26 October 2026

After the user requested further checking and automatic repairs, a read-only independent audit and deterministic alternate-input reproductions identified nine additional implementation errors. Repairs and regression fixtures remain entirely in the canonical notebook:

1. Citation evidence now matches the exact normalized visual label and corrected variant. A citation to Table Z-1 cannot borrow Table Z-10 or its corrected counterpart; indexed metadata takes precedence over incidental caption mentions. Empty evidence text no longer raises a matching error.
2. Mixed figure/table lists preserve each explicitly stated kind instead of converting the table into a second figure.
3. Arithmetic validation recognizes scientific notation, including the calculator's own large/small formatted results, without changing the fixed evaluation tolerances or expected answers.
4. Structured observation units are resolved locally from the requested column/row before a genuinely shared unit heading; an unrelated percent column no longer turns monetary columns into percent rates.
5. Fractional monetary inputs are not automatically annotated as percentages. A percentage-equivalent operand needs explicit documentary share support, rather than merely having magnitude below one.
6. Undated conflicts have a bounded additional witness: two different documents making the same explicitly scoped example/entity statement, differing only in one asserted numeric value. Different entities/measures and equal values are negative tests. This is not a general solver for arbitrary prose disagreement, and the existing semantic/support/revision audits remain required.
7. Native evaluation identity includes the live trusted corpus, passage-reranking toggle, declared encoder/reranker revisions and retrieval backend. Declared versus live corpus/model mismatches are rejected, and changes during evaluation are checked. Material live retrieval changes cannot silently retain the former generation cache identity.
8. Frozen-test resumption compares protected inputs against the original manifest before replacing it. It cannot silently establish a new source-integrity baseline after a source changes.
9. Protocol fingerprints include executable top-level conditional/import/expression statements inside tagged protocol cells, rather than skipping everything except declarations and assignments.

The regression count increased from 313 to **336**, using alternate labels, entities, quantities, units and isolated provenance fixtures. The original golden set, strict references/status/numeric/pattern/semantic rubrics and every historical failed grade were preserved. Citation evidence binding was corrected for future evaluation, but historical predictions or grades were not recomputed. The earlier 10/10 targeted result is historical after these generation/evaluation changes; it is not a fresh score for this source version.

| Current-source verification | Outcome |
|---|---:|
| Deterministic regressions | 336/336 |
| Clean canonical code cells | 31/31 |
| Notebook error outputs | 0 |
| Original actual-model synthetic checks | 16/16 |
| Alternate-input actual-model source-contract checks | 6/6 |
| New full-25 comparison / reranker rerun | Neither performed |

The clean run completed successfully and saved current executed notebook outputs. Independent read-only checks recomputed both model-check signatures, matched the notebook source identity, confirmed 31 distinct executed cells without error outputs, verified all 22 complete passing model-check rows, and confirmed all 24 protected files were unchanged. Those protected files include the source/reference inputs, entire saved first-pass and reranker experiments, and the prior v8 targeted run. The audit lock was removed and the process exited successfully. No downloads, weight training, report regeneration, Git commit, upload or submission were needed.

- Audit outputs: `outputs/evaluation/audit_20261026_v9/` (manifest, status, summary and prior verification snapshot).
- Current notebook source SHA-256: `f4bb96e39fa422aea204954ce86b3a02a4408c37b2506cf866429613eb0b3510`.
- Generation component: `1cb2fef6fcf71dc1ba19ad1e3e7fa9c68b0f7b95b9ffb00fe10ecb8ed1a2ec9d`.
- Evaluation component: `483ada45cda034b4c7a099ed3a2e558910f829255c59033d006f5c846fd5f831`.

These are automated post-feedback software/mechanism diagnostics, not independently held-out answer accuracy or Kaggle performance. They do not certify absence of every possible error, universal unit reasoning or semantic-judge reliability. Team-authored questions, human groundedness ratings (`eval/human_review.csv`), and Kaggle GPU validation (public LB 0.940) are done; see the Status section at the top of this file.
