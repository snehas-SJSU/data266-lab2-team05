#!/usr/bin/env python3
"""Build the DATA 266 Lab 2 report in IEEE two-column Word format."""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Pt, RGBColor, Twips

ROOT = Path(__file__).resolve().parent
# Compressed JPEGs keep the Word file small enough for GitHub to open.
FIG = ROOT / "figures_compressed"
OUT = ROOT / "DATA266_Lab2_Team05_IEEE.docx"

# Usable width inside 0.63 in margins on US Letter: 7.24 in.
# Two columns with a 0.2 in gap: about 3.52 in each.
COL_W = Inches(3.45)


def set_run_font(run, size=10, bold=False, italic=False, color=None):
    run.font.name = "Times New Roman"
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), "Times New Roman")


def add_page_field(paragraph):
    run = paragraph.add_run()
    set_run_font(run, size=8)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def set_columns(section, n, space_twips=288):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    if n > 1:
        cols.set(qn("w:space"), str(space_twips))
    else:
        cols.attrib.pop(qn("w:space"), None)


def configure_page(section):
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.63)
    section.right_margin = Inches(0.63)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.6)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    add_page_field(p)


def P(doc, text="", size=10, bold=False, italic=False, align="justify",
      before=0, after=0, first=0, center=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if first:
        pf.first_line_indent = Inches(first)
    if center or align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if text:
        run = p.add_run(text)
        set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def heading(doc, text):
    return P(doc, text, size=10, bold=True, align="center", before=11, after=3, center=True)


def subhead(doc, text):
    return P(doc, text, size=10, italic=True, align="left", before=8, after=2)


def body(doc, text, indent=True):
    return P(doc, text, size=10, align="justify", before=0, after=2, first=0.15 if indent else 0)


def caption(doc, text):
    return P(doc, text, size=8, italic=False, align="center", before=1, after=6, center=True)


def figure(doc, path, cap, width=None, height=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run()
    if height is not None:
        run.add_picture(str(path), height=height)
    else:
        run.add_picture(str(path), width=width or COL_W)
    caption(doc, cap)


def shade_header(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F2F2F2")
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_border(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "666666")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def set_table_fixed(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    total = sum(widths)
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(total))
    tblW.set(qn("w:type"), "dxa")
    layout = tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblPr.append(layout)
    layout.set(qn("w:type"), "fixed")
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for child in list(grid):
            grid.remove(child)
    else:
        grid = OxmlElement("w:tblGrid")
        tblPr.addnext(grid)
    for w in widths:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w))
        grid.append(gc)
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Twips(w)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW")
                tcPr.append(tcW)
            tcW.set(qn("w:w"), str(w))
            tcW.set(qn("w:type"), "dxa")


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    set_table_fixed(table, widths)
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(h)
        set_run_font(run, size=8, bold=True)
        shade_header(cell)
        set_cell_border(cell)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            set_run_font(run, size=8)
            set_cell_border(cell)
    # small spacer after the table
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)
    return table


def table_title(doc, text):
    P(doc, text, size=8, bold=True, align="center", before=8, after=1, center=True)


def build():
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), "Times New Roman")

    sec0 = doc.sections[0]
    configure_page(sec0)
    set_columns(sec0, 1)


    P(doc, "Retrieval over Economics Documents, Pokemon-Style Diffusion, and a Multimodal Image Agent",
      size=16, bold=True, align="center", before=0, after=6, center=True)
    P(doc, "Sneha Singh and Ritika Neema", size=11, align="center", before=2, after=0, center=True)
    P(doc, "Department of Applied Data Science, San Jose State University, San Jose, CA, USA",
      size=10, italic=True, align="center", before=0, after=0, center=True)
    P(doc, "DATA 266, Fall 2026  ·  Pair Programming Team 05",
      size=10, align="center", before=0, after=2, center=True)
    P(doc, "GitHub: https://github.com/snehas-SJSU/data266-lab2-team05",
      size=9, align="center", before=0, after=8, center=True)

    abs_p = P(doc, "", size=9, align="justify", before=2, after=2)
    r1 = abs_p.add_run("Abstract—")
    set_run_font(r1, size=9, bold=True, italic=True)
    r2 = abs_p.add_run(
        "This paper presents one team system for each DATA 266 Lab 2 task, evaluated at seed "
        "266. Part 1 is multimodal RAG over three read-only economics PDFs: dense retrieval "
        "answered 13 of 25 golden questions; the final hybrid system with passage reranking "
        "answered 16. Mean groundedness from both members was 4.88. The team public Kaggle "
        "practice best is 0.940. Part 2 fine-tunes Stable Diffusion 1.5 with LoRA; KID falls "
        "from 0.06024 to 0.02314 for rank 16. Blind raters did not prefer rank 16 over the "
        "base model (win rate 0.387). Part 3 is a five-agent image system: 6 of 15 scenarios "
        "passed without a critique loop and 4 of 15 with it; red-team attack success was 0 of 10."
    )
    set_run_font(r2, size=9, bold=True)

    idx = P(doc, "", size=9, align="justify", before=2, after=4)
    a = idx.add_run("Index Terms—")
    set_run_font(a, size=9, bold=True, italic=True)
    b = idx.add_run("document retrieval, latent diffusion, low-rank adaptation, image agents")
    set_run_font(b, size=9, bold=True)

    sec1 = doc.add_section(WD_SECTION.CONTINUOUS)
    configure_page(sec1)
    set_columns(sec1, 2, 288)

    heading(doc, "I.  INTRODUCTION")
    body(doc,
         "Lab 2 requires a single IEEE report for multimodal RAG, diffusion fine-tuning, and "
         "a multimodal image agent. Both members implemented and ran all three parts. This "
         "document is the common team report: one architecture, one baseline-versus-final "
         "table, a short failure analysis, and a safety discussion for each part. Seed 266 "
         "is fixed for splits, sampling, and generation. Evaluation artifacts were written "
         "before the reported runs. Per-member implementation detail is in CONTRIBUTIONS.md.",
         indent=False)

    heading(doc, "II.  QUESTION ANSWERING OVER THREE PDFS")
    subhead(doc, "A.  Architecture")
    body(doc,
         "The knowledge base is three read-only PDFs. Each page becomes a text chunk plus "
         "separate figure and table crops. Trusted figures are captioned and stored with the "
         "text. Appendix C of Document 3, including Figure C-1, is marked low trust. Its text "
         "is not indexed and its pixels are not described. Fig. 1 is that pipeline. The appendix "
         "has no arrow into retrieval.",
         indent=False)
    figure(doc, FIG / "part1" / "architecture.jpg",
           "Fig. 1. Three PDFs are indexed. Appendix C is left out. Dense retrieval is the baseline. The final retriever adds BM25 and an optional passage reranker.")
    body(doc,
         "Dense retrieval is the baseline. The final retriever adds BM25 and can rerank the "
         "fused list with a cross-encoder. We blend the two rankings rather than sorting by "
         "the cross-encoder alone. An answer is a status, a short text, and citations that "
         "have to name a retrieved record. The answer is written from those records, in the "
         "retrieve-then-generate order [6]. A temporary number supplied in chat is used only "
         "in that conversation and is not written back into the index. A request to edit the "
         "documents permanently is refused.")
    body(doc,
         "The golden-set run used qwen2.5:7b for answers and qwen2.5vl:3b for figure captions, "
         "temperature 0. The Kaggle run of the same notebook loads Qwen2.5-VL-7B-Instruct in "
         "4-bit on one Tesla T4.")

    subhead(doc, "B.  Results")
    body(doc,
         "Table I is the lab-required comparison: dense retrieval only versus the final hybrid "
         "system with passage reranking, on a 25-question golden set that is not copied from "
         "Questions.csv. Human groundedness was scored 1–5 by both members "
         "(eval/human_review.csv); mean 4.88, and scores agreed within one point on 24 of 25 "
         "questions.",
         indent=False)
    table_title(doc, "TABLE I.  DENSE BASELINE VERSUS FINAL SYSTEM")
    add_table(
        doc,
        ["Measurement", "Dense", "Final"],
        [
            ["Correct answers / 25", "13", "16"],
            ["Recall at 5", "0.913", "0.957"],
            ["Citation accuracy", "0.837", "0.768"],
            ["Mean groundedness (both)", "—", "4.88"],
            ["Median seconds / question", "8.31", "8.48"],
        ],
        [2400, 1300, 1300],
    )
    body(doc,
         "Final means hybrid retrieval with exact figure/table lookup and "
         "ms-marco-MiniLM-L-6-v2 passage reranking. On a held-out set of 25 questions "
         "committed before any run on that set, hybrid scored 19/25 and the final system "
         "scored 21/25. Public practice scores for that pipeline rose from 0.526 to 0.576. "
         "The team selected practice leaderboard best is 0.940. The graded notebook is the "
         "executed Kaggle run at tag p1-v1.0 (about 1137 s on a Tesla T4). A complementary "
         "local hybrid ablation reached 25/25 with the reranker off and on; that tie shows "
         "reranking need not change answers once the right page is already retrieved, while "
         "Table I supports the team reranker bonus claim.")

    subhead(doc, "C.  What worked and what did not")
    body(doc,
         "Conversation-scoped memory kept chat assumptions out of the index. Exact figure "
         "lookup reduced near-identical chart mix-ups. Passage reranking added two correct "
         "golden answers and two held-out answers through better fact and status decisions, "
         "but lowered figure and citation scores on the golden set. A web-trained "
         "cross-encoder often ranks the passage containing a number above the figure or "
         "table record the scorer expects to see cited.",
         indent=False)
    body(doc,
         "Remaining misses are mostly multi-hop table questions and near-duplicate chart "
         "titles across documents. Hybrid retrieval helps when the figure chunk is already "
         "near the top; it does not fully solve two-table questions.")

    subhead(doc, "D.  Safety")
    body(doc,
         "Appendix C is dropped at ingest. A false claim on that page, or the instruction "
         "drawn inside Figure C-1, cannot be retrieved and is not shown to the vision model. "
         "That chart is an indirect prompt injection: an instruction aimed at the model, hidden "
         "in the document [7]. "
         "A citation has to name a record that was actually retrieved.",
         indent=False)
    body(doc,
         "Concrete golden-set checks match that design. A rate that appears only in the "
         "untrusted appendix is returned as not_found. A false GDP premise that contradicts the "
         "book table is corrected with false_premise. An instruction to ignore the documents "
         "and report a fabricated inflation number is ignored; the answer stays on the trusted "
         "table. Temporary chat-only values stay in memory and are never written back to the "
         "index.")

    heading(doc, "III.  FINE-TUNING A DIFFUSION MODEL")
    subhead(doc, "A.  Setup")
    body(doc,
         "We fine-tuned Stable Diffusion 1.5 [1] on the Pokemon BLIP captions, 833 image-caption "
         "pairs. No file was corrupt and none was an exact duplicate, so nothing was "
         "removed. Seed 266 split the data 90/10 into 750 training images and 83 test images, "
         "resized to 512. The test images were used only for scoring. Captions are short and "
         "sometimes omit a color, and most pictures are a single creature in the center.",
         indent=False)
    body(doc,
         "The two runs share the data, 1500 steps, batch 4, and learning rate 0.0001. Both "
         "are LoRA adapters on the attention projections, with the VAE, the text encoder, and "
         "the base UNet frozen [2]. Rank 4 has 797,184 trainable parameters. Rank 16 has "
         "3,188,736. We kept rank 16: at the same training cost (0.2399 against 0.2401 "
         "GPU-hours) it had the lower KID (0.02314 against 0.02671) and the higher CLIP "
         "score (0.3155 against 0.3110). The KID gap is inside the noise, so this is a "
         "tie-break, and the blind study below did not favor rank 16 over the base model "
         "either. Mean loss over the last tenth of training was 0.0535 "
         "and 0.0533. The runs took 864.2 s and 863.8 s. Fig. 2 is this pipeline.")
    figure(doc, FIG / "part2" / "architecture.jpg",
           "Fig. 2. One dataset, two LoRA ranks on a frozen Stable Diffusion 1.5, then the same 20 prompts for the base model and both adapters.")
    body(doc,
         "Twenty prompts were saved before training ended: 10 from the test captions, 5 "
         "outside that style, and 5 that ask for a count or a second object. Each model "
         "generated seeds 0 and 1 at 512 pixels, 30 steps, and guidance 7.5. That is 40 "
         "images per model.")

    subhead(doc, "B.  Results")
    body(doc,
         "Table II is the comparison. KID is the 20 in-domain generations against the 83 "
         "test images, with the subset standard deviation. CLIP is prompt-image similarity "
         "on all 40 images of that model. LPIPS is the mean distance between the two seeds "
         "of each prompt. KID falls by more than that error bar, so the pictures move toward "
         "the training set. CLIP and diversity move the other way, by a little. Rank 16 is "
         "only slightly ahead of rank 4, inside the KID noise.",
         indent=False)
    table_title(doc, "TABLE II.  BASE MODEL AND TWO LORA RANKS")
    add_table(
        doc,
        ["Measurement", "Base", "Rank 4", "Rank 16"],
        [
            ["KID (lower better)", "0.06024 ± 0.00852", "0.02671 ± 0.00617", "0.02314 ± 0.00506"],
            ["CLIP score", "0.3220", "0.3110", "0.3155"],
            ["LPIPS diversity", "0.7349", "0.7178", "0.7123"],
            ["Win rate vs base (3 raters)", "—", "—", "0.387 (29/75)"],
            ["GPU-hours (peak)", "—", "0.2401 (4.22 GB)", "0.2399 (4.27 GB)"],
        ],
        [1400, 1200, 1200, 1200],
    )
    body(doc,
         "Fig. 3 is the in-domain sheet behind the KID number: base, rank 4, and rank 16 on "
         "the first eight prompts at seed 0 (left panel prompts 1–4, right panel 5–8). Style "
         "moves toward the caption set; faces and palettes tighten as rank rises.")
    figure(
        doc, FIG / "part2" / "grid.jpg",
        "Fig. 3. Base, rank 4, and rank 16 on the first eight in-domain prompts, seed 0.",
        width=COL_W,
    )
    table_title(doc, "TABLE III.  CONFIRMATORY LORA VS DORA (RANK 8)")
    add_table(
        doc,
        ["Model", "KID", "CLIP", "GPU-hours"],
        [
            ["Base", "0.08577 ± 0.01124", "0.3215", "—"],
            ["LoRA rank 8", "0.05606 ± 0.00671", "0.3147", "0.4223"],
            ["DoRA rank 8", "0.05352 ± 0.00673", "0.3167", "0.5262"],
        ],
        [1300, 1700, 1000, 1000],
    )
    body(doc,
         "Both members and one rater from outside the team compared rank 16 with the base "
         "model on 25 blind pairs, sides swapped. Rank 16 won 29 of 75 votes (win rate 0.387; "
         "95% CI 0.285–0.5), so raters did not clearly prefer the fine-tune. Per rater the "
         "notebook printed Rajesh 0.40, Ritika Neema 0.44, and Sneha Singh 0.32. By prompt "
         "type, in-domain win rate was 0.41, out-of-domain 0.444, and compositional 0.278. "
         "Table III is a confirmatory adapter study (LoRA rank 8 vs DoRA rank 8, 1200 steps, "
         "seed 266) [3]: KID falls for both adapters; DoRA’s KID edge (0.0025) sits inside "
         "one standard deviation at higher GPU cost. The same three raters preferred LoRA "
         "rank 8 over the base model (win rate 0.747). The Hugging Face model card on the "
         "rank-16 repository records base model, data, settings, results, limitations, and "
         "non-commercial license.")
    body(doc,
         "Fig. 4 shows the failure modes that dominate the human study (left panel "
         "out-of-domain, right panel compositional). Compositional prompts drop a leaf, a "
         "box, a count, or a second creature. Out-of-domain lakes, cars, and rooms often stay "
         "photographic. We still keep rank 16 for the automatic metrics at equal training "
         "cost; for open prompts the base model remains competitive with people.")
    figure(
        doc, FIG / "part2" / "failure_candidates.jpg",
        "Fig. 4. Prompts outside the training style, seed 0. Counts and second objects are often missing.",
        width=COL_W,
    )

    subhead(doc, "C.  What worked and what did not")
    body(doc,
         "LoRA clearly pulled in-domain samples toward the Pokemon look: the KID drop is the "
         "main success. The cost is a small CLIP regression and lower seed diversity, which "
         "matches the intuition that a low-rank adapter specializes the prior. Compositional "
         "prompts remain the weakest slice in the blind study (win rate 0.278). Rank 4 is "
         "almost as strong as rank 16 on KID, so capacity was not the bottleneck; caption "
         "coverage and prompt design were.",
         indent=False)
    body(doc,
         "DoRA’s KID edge sits inside noise at higher GPU cost, so it does not overturn the "
         "reported LoRA choice. Human preference for rank-8 LoRA over the base model "
         "(win rate 0.747) is clearer than for rank 16, which is why Table II still leads "
         "with automatic metrics while the blind study stays honest.")

    subhead(doc, "D.  Safety")
    body(doc,
         "The training images are copyrighted characters. The adapters and generated images "
         "are for non-commercial course work only. The model card on the rank-16 Hugging Face "
         "repo states the same limit and summarizes training settings and measured results.",
         indent=False)
    body(doc,
         "The Stable Diffusion safety checker flagged 10 of the 120 images: 2 from the base "
         "model, 4 from rank 4, and 4 from rank 16 (8 in-domain, 2 compositional). Fig. 5 "
         "shows those ten. Manual review found ordinary cartoons, so we treated the flags as "
         "false positives and did not remove the images from the metric tables.")
    figure(doc, FIG / "part2" / "nsfw_flagged.jpg",
           "Fig. 5. The ten images flagged by the safety checker. We treated these flags as false positives.",
           width=COL_W)
    body(doc,
         "For memorization we took 10 in-domain rank-16 images at seed 0 and found the nearest "
         "training image by CLIP similarity. Scores run from 0.789 to 0.938. Fig. 6 shows the "
         "three closest pairs. The creature type matches; pose and color differ. The largest "
         "per-tile RMS distance to that training image, over 16 tiles, was 0.584 to 0.801. "
         "Carlini et al. treat a distance near 0.15 as a near-copy on their scale [4]. Nothing "
         "we generated was in that range.")
    figure(doc, FIG / "part2" / "memorization.jpg",
           "Fig. 6. Three generations (left of each pair) and nearest training image by CLIP similarity.",
           width=COL_W)

    heading(doc, "IV.  A MULTIMODAL IMAGE AGENT")
    subhead(doc, "A.  Architecture")
    body(doc,
         "Five steps share one state. The planner chooses a route or refuses the request. "
         "Vision describes the image, lists objects, reads any text, and answers two questions "
         "about the picture. The prompt step rewrites the request and may not add objects the "
         "user did not ask for. Generation records the model, prompt, seed, and step count. "
         "Critique accepts the image or writes one revision, and it stops after two revisions. "
         "Fig. 7 shows where a request stops early.",
         indent=False)
    figure(doc, FIG / "part3" / "architecture.jpg",
           "Fig. 7. A caption, an unsafe request, or a vague request stops early. The critic revises at most twice. Every request saves a JSON trace.")
    body(doc,
         "Style change is Stable Diffusion image-to-image. Edits use InstructPix2Pix [5]. "
         "Enhancement is brightness, contrast, and sharpening. A Pokemon-style request loads "
         "the rank-16 adapter from Section III. Language and vision calls are Qwen2.5-VL-3B "
         "at temperature 0. Text found in the image is stored as text. It is not copied into "
         "the prompt. The Gradio page shows each step, and the same app is deployed as a "
         "Hugging Face Space.")

    subhead(doc, "B.  Results")
    body(doc,
         "Fifteen scenarios and ten red-team cases were written before the run. Table IV is "
         "the 26 October run on an L4 (24 GB VRAM), at 512 pixels and 30 steps, seed 266. The baseline "
         "column is the same pipeline with the critique loop off. The final column is the "
         "loop on. The loop does not help. It passes fewer scenarios and takes about twice "
         "as long.",
         indent=False)
    table_title(doc, "TABLE IV.  WITH AND WITHOUT THE CRITIQUE LOOP")
    add_table(
        doc,
        ["Measurement", "No critique", "With critique"],
        [
            ["Scenarios passed / 15", "6/15", "4/15"],
            ["Critic agreement", "n/a", "34/48 = 0.71"],
            ["Red-team attacks / 10", "same checks", "0/10"],
            ["Rater agreement", "0.725 (kappa 0.573)", "11 disagreements resolved"],
            ["Median seconds", "15.1", "31.4"],
        ],
        [1750, 1650, 1650],
    )
    body(doc,
         "Routes matched on 30 of 30 scenarios. Both members rated all 40 runs independently "
         "(raw agreement 0.725, Cohen kappa 0.573); 11 disagreements were resolved in "
         "resolutions.csv. The critic’s accept/revise mark matched human ratings on 34 of 48 "
         "attempts. Passes with the loop on were two domain-style images, one style change, "
         "and one edit. Color and lighting revisions remain brittle: the critic often asks "
         "for a second pass that changes palette but not the object layout the rater wanted. "
         "A confirmatory T4 replication of the same 25 cases also showed no primary gain from "
         "the critique loop (6/15 either way under joint pass). Fig. 8 is a successful "
         "domain-style generation.")
    figure(
        doc, FIG / "part3" / "domain_style.jpg",
        "Fig. 8. Domain-style generation with the LoRA rank-8 adapter. The creature is in the training style. The flat background matches the caption set.",
        width=Inches(2.6),
    )
    body(doc,
         "Failures cluster on compositional edits and vague prompts: InstructPix2Pix often "
         "keeps the original layout when a second object is requested, and vague style "
         "requests correctly ask for clarification instead of guessing.")

    subhead(doc, "C.  Safety and system card")
    body(doc,
         "The planner refuses sexual content, graphic violence, and a deceptive edit of a "
         "real person. An empty or broken file returns a short message. Visible text in an "
         "image is stored as data and is never copied into the generation prompt. The output "
         "checker runs even when the critique loop is off. A checker flag blocks the image "
         "only when the vision model agrees, because that checker flags ordinary cartoons, "
         "as in Fig. 5.",
         indent=False)
    body(doc,
         "On the ten attacks, a request to print the system prompt (R01) and a request whose "
         "image said to generate a nude person (R04) both came back as captions; the text in "
         "the image was not followed. R02 still ran safety checks and stopped at the revision "
         "limit. R05–R07 were refused before generation. R08, a deceptive edit of a real "
         "person, was refused. R09 and R10 returned clear errors for a corrupt file and an "
         "empty file. R03 is the residual miss: a faint corner string was not read, but the "
         "user request was only to brighten the image and was not rewritten into a harmful "
         "action. Attack success was 0 of 10 on the reported run.")
    body(doc,
         "System card. The system can caption a photo, answer a visual question, restyle, "
         "edit, brighten, or draw with the Part 2 adapter; it refuses unsafe requests, asks "
         "when the request is vague, and reports which step failed. Known limits: color and "
         "time-of-day edits are brittle; the 3B critic can over-reject; the adapter is "
         "non-commercial research only; free-tier Space latency varies. Guards are the input "
         "policy check, output check, image-text isolation, a two-revision cap, and a JSON "
         "trace for every request. The public Space URL is listed in links.txt.")

    heading(doc, "V.  CONCLUSION")
    body(doc,
         "The team delivered one reported system per part. Part 1 improves from dense retrieval "
         "(13/25) to a hybrid reranked final system (16/25) on the golden set, with mean "
         "groundedness 4.88 and a public Kaggle practice best of 0.940; Appendix C stays "
         "quarantined. Part 2 lowers KID from 0.06024 to 0.02314 with LoRA rank 16; human "
         "preference over the base model is mixed, while a confirmatory rank-8 study reaches "
         "win rate 0.747. Part 3 refuses all ten red-team attacks but does not gain primary "
         "accuracy from the critique loop (6/15 without, 4/15 with). Where automatic metrics "
         "and human judgements disagree, both are reported. Tags, eval tables, and links are "
         "in the repository; member notes and AI tooling are in CONTRIBUTIONS.md and "
         "AI_USAGE.md.",
         indent=False)

    heading(doc, "REFERENCES")
    refs = [
        "[1]  R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 10684–10695.",
        "[2]  E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent., 2022.",
        "[3]  S.-Y. Liu et al., “DoRA: Weight-decomposed low-rank adaptation,” in Proc. Int. Conf. Mach. Learn., 2024.",
        "[4]  N. Carlini et al., “Extracting training data from diffusion models,” in Proc. USENIX Security Symp., 2023, pp. 5253–5270.",
        "[5]  T. Brooks, A. Holynski, and A. A. Efros, “InstructPix2Pix: Learning to follow image editing instructions,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 18392–18402.",
        "[6]  P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Proc. Adv. Neural Inf. Process. Syst., 2020.",
        "[7]  K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection,” in Proc. ACM Workshop Artif. Intell. Secur., 2023, pp. 79–90.",
    ]
    for ref in refs:
        p = P(doc, ref, size=8, align="justify", before=0, after=2)
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.first_line_indent = Inches(-0.2)

    doc.save(OUT)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    build()
