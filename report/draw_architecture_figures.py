#!/usr/bin/env python3
"""Redraw Part 1–3 architecture diagrams with print-sized type."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "figures_arch"
OUT.mkdir(exist_ok=True)

W, H = 2400, 1400


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(54, True)
F_SEC = font(36, True)
F_BOX = font(32, True)
F_LINE = font(28, False)
F_FOOT = font(24, False)


def rounded(draw, xy, fill, outline="#333333", r=22, width=3):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def arrow(draw, x1, y1, x2, y2, color="#333333"):
    draw.line((x1, y1, x2, y2), fill=color, width=5)
    if x2 > x1:
        draw.polygon([(x2, y2), (x2 - 16, y2 - 10), (x2 - 16, y2 + 10)], fill=color)
    elif y2 > y1:
        draw.polygon([(x2, y2), (x2 - 10, y2 - 16), (x2 + 10, y2 - 16)], fill=color)


def text_block(draw, x, y, lines, fonts, fill="#111111", gap=8):
    cy = y
    for line, f in zip(lines, fonts):
        draw.text((x, cy), line, font=f, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=f)
        cy += (bbox[3] - bbox[1]) + gap
    return cy


def box_with_text(draw, xy, fill, lines, fonts, pad=22):
    rounded(draw, xy, fill=fill)
    x0, y0, _, _ = xy
    text_block(draw, x0 + pad, y0 + pad, lines, fonts)


def draw_part1():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "Part 1 architecture", font=F_TITLE, fill="#111")

    d.text((40, 110), "1.1  Ingestion and indexing", font=F_SEC, fill="#111")
    y, h = 165, 170
    boxes = [
        (40, y, 560, y + h, "#FFE0B2",
         ["PDFs + trust manifest", "DOC1 / DOC2: high trust", "DOC3: medium trust"]),
        (620, y, 1140, y + h, "#BBDEFB",
         ["Page-aware chunks", "doc id + page", "figure / table crops"]),
        (1200, y, 1720, y + h, "#B3E5FC",
         ["Trusted figures", "VLM captions cached", "kept with text index"]),
        (1780, y, 2360, y + h, "#FFCDD2",
         ["Appendix C", "LOW TRUST", "never retrieved"]),
    ]
    for x0, y0, x1, y1, fill, lines in boxes:
        box_with_text(d, (x0, y0, x1, y1), fill, lines, [F_BOX, F_LINE, F_LINE])
    arrow(d, 560, y + h // 2, 615, y + h // 2)
    arrow(d, 1140, y + h // 2, 1195, y + h // 2)
    d.line((1720, y + h // 2, 1775, y + h // 2), fill="#999", width=4)
    d.text((1685, y + h + 8), "no path", font=F_FOOT, fill="#C62828")

    d.text((40, 380), "1.2  Retrieval and grounded answers", font=F_SEC, fill="#111")
    y, h = 435, 180
    box_with_text(d, (40, y, 520, y + h), "#BBDEFB",
                  ["ChromaDB", "MiniLM + BM25", "hybrid fusion"], [F_BOX, F_LINE, F_LINE])
    box_with_text(d, (600, y, 1100, y + h), "#E0E0E0",
                  ["Baseline", "dense only", "no BM25 / rerank"], [F_BOX, F_LINE, F_LINE])
    box_with_text(d, (1180, y, 1720, y + h), "#90CAF9",
                  ["Final system", "hybrid + optional", "passage reranker"], [F_BOX, F_LINE, F_LINE])
    box_with_text(d, (1800, y, 2360, y + h), "#FFF59D",
                  ["Answer model", "JSON: status, answer,", "citations, figure labels"],
                  [F_BOX, F_LINE, F_LINE])
    arrow(d, 520, y + h // 2, 595, y + h // 2)
    arrow(d, 520, y + h // 2, 520, y + h + 30)
    arrow(d, 520, y + h + 30, 1180, y + h + 30)
    arrow(d, 1180, y + h + 30, 1180, y + h)
    arrow(d, 1100, y + h // 2, 1175, y + h // 2)
    arrow(d, 1720, y + h // 2, 1795, y + h // 2)
    arrow(d, 1720, y + 40, 1795, y + 40)

    d.text((40, 700), "1.3  Conversation", font=F_SEC, fill="#111")
    d.text((1240, 700), "1.4  Evaluation", font=F_SEC, fill="#111")
    y, h = 755, 220
    box_with_text(d, (40, y, 1160, y + h), "#FFF59D",
                  ["One memory per conversation",
                   "Chat assumptions stay out of the index",
                   "No permanent edit of the PDFs",
                   "Seed 266 · temperature 0"],
                  [F_BOX, F_LINE, F_LINE, F_LINE])
    box_with_text(d, (1240, y, 2360, y + h), "#C8E6C9",
                  ["Golden set: 25 questions",
                   "Correct / 25 · Recall@5 · citations",
                   "Groundedness 1–5 (both members)",
                   "Median seconds · Kaggle 90 min"],
                  [F_BOX, F_LINE, F_LINE, F_LINE])

    d.text((40, 1080),
           "Text and pixels are data. Instructions inside a document or chart are not followed.",
           font=F_FOOT, fill="#444")
    im.save(OUT / "part1_architecture.jpg", quality=92, optimize=True)


def draw_part2():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "Part 2 architecture", font=F_TITLE, fill="#111")
    d.text((40, 110), "Fine-tune pipeline", font=F_SEC, fill="#111")
    y, h, bw, gap = 170, 260, 720, 40
    specs = [
        ("#BBDEFB", ["2.1  Data", "Pokemon BLIP captions", "833 pairs · 0 corrupt",
                     "Seed 266 → 750 / 83", "Resize 512×512"]),
        ("#E0E0E0", ["2.2  Fine-tuning", "Frozen: VAE, CLIP, UNet", "LoRA on attention",
                     "Rank 4 and rank 16", "1500 steps · batch 4 · lr 1e-4"]),
        ("#90CAF9", ["2.3  Generation", "20 prompts × seeds 0, 1", "Base + rank 4 + rank 16",
                     "120 images at 512 px", "Same prompts for all"]),
    ]
    for i, (fill, lines) in enumerate(specs):
        x0 = 40 + i * (bw + gap)
        box_with_text(d, (x0, y, x0 + bw, y + h), fill, lines, [F_BOX] + [F_LINE] * 4, pad=28)
        if i < 2:
            arrow(d, x0 + bw, y + h // 2, x0 + bw + gap - 5, y + h // 2)

    d.text((40, 480), "Score and safety", font=F_SEC, fill="#111")
    y, h = 540, 280
    box_with_text(d, (40, y, 1180, y + h), "#C8E6C9",
                  ["2.4  Metrics",
                   "KID vs test images (lower better)",
                   "CLIP score · LPIPS diversity",
                   "GPU-hours and peak memory",
                   "Blind human win rate (3 raters)"],
                  [F_BOX] + [F_LINE] * 4, pad=28)
    box_with_text(d, (1260, y, 2360, y + h), "#FFCDD2",
                  ["2.5  Safety",
                   "Copyrighted characters → research only",
                   "SD safety checker on all 120 images",
                   "Memorization: nearest train by CLIP",
                   "Weights on Hugging Face (not GitHub)"],
                  [F_BOX] + [F_LINE] * 4, pad=28)
    arrow(d, 1180, y + h // 2, 1255, y + h // 2)
    d.text((40, 880),
           "Final adapter kept: LoRA rank 16 (blind study). "
           "Second pair LoRA r8 vs DoRA r8 favors LoRA r8 on cost.",
           font=F_FOOT, fill="#444")
    im.crop((0, 0, W, 1000)).save(OUT / "part2_architecture.jpg", quality=92, optimize=True)


def draw_part3():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "Part 3 architecture", font=F_TITLE, fill="#111")
    d.text((40, 110), "Agents (shared typed state)", font=F_SEC, fill="#111")
    y, h, bw, gap = 170, 200, 420, 30
    labels = [
        ("Planner", "route / refuse"),
        ("Vision", "caption + text"),
        ("Prompt", "rewrite request"),
        ("Generate", "edit / style / LoRA"),
        ("Critique", "accept / revise ×2"),
    ]
    for i, (title, sub) in enumerate(labels):
        x0 = 40 + i * (bw + gap)
        box_with_text(d, (x0, y, x0 + bw, y + h), "#90CAF9",
                      [title, sub], [F_BOX, F_LINE], pad=28)
        if i < 4:
            arrow(d, x0 + bw, y + h // 2, x0 + bw + gap - 5, y + h // 2)

    d.text((40, 420), "Guards and delivery", font=F_SEC, fill="#111")
    y, h = 480, 260
    box_with_text(d, (40, y, 600, y + h), "#FFF59D",
                  ["Early stops", "Caption / VQA → stop", "  after Vision",
                   "Unsafe or vague →", "  stop at Planner"],
                  [F_BOX] + [F_LINE] * 4, pad=26)
    box_with_text(d, (640, y, 1200, y + h), "#FFCDD2",
                  ["Safety", "Image text is data,", "  not an instruction",
                   "JSON trace for", "  every request"],
                  [F_BOX] + [F_LINE] * 4, pad=26)
    box_with_text(d, (1240, y, 1760, y + h), "#C8E6C9",
                  ["Demo", "Gradio UI", "HF Space (boxai)", "Seed 266 · 512 px", "30 steps"],
                  [F_BOX] + [F_LINE] * 4, pad=26)
    box_with_text(d, (1800, y, 2360, y + h), "#E1BEE7",
                  ["Evaluation", "15 scenarios", "10 red-team cases",
                   "Critique on vs off", "Rater agreement"],
                  [F_BOX] + [F_LINE] * 4, pad=26)
    d.text((40, 800),
           "Critique revises at most twice. "
           "Unsafe or injection-style image text never becomes a prompt.",
           font=F_FOOT, fill="#444")
    im.crop((0, 0, W, 920)).save(OUT / "part3_architecture.jpg", quality=92, optimize=True)


def main():
    draw_part1()
    draw_part2()
    draw_part3()
    for name in ("part1_architecture.jpg", "part2_architecture.jpg", "part3_architecture.jpg"):
        path = OUT / name
        print(f"{name}: {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
