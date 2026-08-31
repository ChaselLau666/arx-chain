#!/usr/bin/env python3
"""Render the canonical pipeline Markdown into a polished operator PDF."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, XPreformatted

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/ARX_LIFT2S_CUSTOM_PIPELINE.md"
OUTPUT = ROOT / "docs/ARX_LIFT2S_CUSTOM_PIPELINE.pdf"


def register_font() -> str:
    candidates = [
        os.environ.get("ARX_MANUAL_FONT"),
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                pdfmetrics.registerFont(TTFont("ARXCJK", candidate, subfontIndex=0))
                return "ARXCJK"
            except Exception:
                continue
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def inline(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)


def render() -> Path:
    font = register_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=15, spaceAfter=5)
    bullet = ParagraphStyle("BulletCN", parent=body, leftIndent=15, firstLineIndent=-8)
    note = ParagraphStyle(
        "NoteCN", parent=body, leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#4C78A8"),
        borderWidth=1, borderPadding=7, backColor=colors.HexColor("#EEF5FB"), spaceBefore=5, spaceAfter=8
    )
    headings = {
        1: ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font, fontSize=22, leading=29, textColor=colors.HexColor("#17324D"), alignment=TA_CENTER, spaceAfter=16),
        2: ParagraphStyle("H1CN", parent=styles["Heading1"], fontName=font, fontSize=15, leading=21, textColor=colors.HexColor("#17324D"), spaceBefore=12, spaceAfter=7),
        3: ParagraphStyle("H2CN", parent=styles["Heading2"], fontName=font, fontSize=12, leading=17, textColor=colors.HexColor("#2C5F8A"), spaceBefore=8, spaceAfter=5),
    }
    code_style = ParagraphStyle(
        "Code", fontName="Courier", fontSize=7.5, leading=10, leftIndent=7, rightIndent=7,
        borderColor=colors.HexColor("#D6DEE5"), borderWidth=0.6, borderPadding=7,
        backColor=colors.HexColor("#F5F7F9"), spaceBefore=4, spaceAfter=7,
    )
    code_cn_style = ParagraphStyle("CodeCN", parent=code_style, fontName=font)

    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), body))
            paragraph.clear()

    for line in lines:
        if line.startswith("```"):
            if in_code:
                code_text = "\n".join(code)
                selected_style = code_cn_style if any(ord(char) > 127 for char in code_text) else code_style
                story.append(XPreformatted(html.escape(code_text), selected_style))
                code.clear()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            story.append(Paragraph(inline(heading.group(2)), headings[level]))
            continue
        if line.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(inline(line[2:]), note))
            continue
        if re.match(r"^[-*]\s+", line):
            flush_paragraph()
            story.append(Paragraph("• " + inline(line[2:]), bullet))
            continue
        if re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            story.append(Paragraph(inline(line), bullet))
            continue
        paragraph.append(line.rstrip("  "))
    flush_paragraph()

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(18 * mm, 12 * mm, "ARX LIFT2s 自有采集、训练与远程推理手册")
        canvas.drawRightString(192 * mm, 12 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm, title="ARX LIFT2s 自有采集、训练与远程推理手册",
        author="ARX LIFT2s Project",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(render())
