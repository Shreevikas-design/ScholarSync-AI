from __future__ import annotations

import re
from datetime import date
from html import escape as html_escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from schemas.models import LiteratureReport


def safe_filename(title: str, extension: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()[:80] or "scholarsync-report"
    return f"{slug}.{extension.lstrip('.')}"


def export_markdown(report: LiteratureReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.markdown, encoding="utf-8")
    return path


def _escape(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("**", "").replace("\n", "<br/>")


def _parse_body(body: str, styles) -> list:
    lines = body.splitlines()
    output = []
    table_lines: list[str] = []
    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        raw_rows = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in table_lines]
        raw_rows = [r for r in raw_rows if not all(set(cell) <= {"-", ":"} for cell in r)]
        cell_style = ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=6.5, leading=8)
        head_style = ParagraphStyle(name="TableHead", parent=cell_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#16324F"))
        rows = [[Paragraph(_escape(cell), head_style if row_i == 0 else cell_style) for cell in row] for row_i, row in enumerate(raw_rows)]
        if rows:
            table = Table(rows, colWidths=[1.2 * inch, 1.1 * inch, 1.1 * inch, 0.75 * inch, 0.8 * inch, 0.75 * inch, 1.0 * inch, 0.8 * inch, 0.8 * inch, 1.3 * inch], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324F")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            output.append(table)
            output.append(Spacer(1, 8))
        table_lines = []
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        content = line.strip()
        if not content:
            output.append(Spacer(1, 5))
        elif content.startswith("- "):
            output.append(Paragraph("&#8226; " + _escape(content[2:]), styles["BodyText"]))
        else:
            output.append(Paragraph(_escape(content), styles["BodyText"]))
    flush_table()
    return output


def export_pdf(report: LiteratureReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ScholarTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#18324B"), spaceAfter=12))
    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(letter), rightMargin=0.7 * inch, leftMargin=0.7 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch, title=report.title, author="ScholarSync AI",
    )
    story = [
        Paragraph(_escape(report.title), styles["ScholarTitle"]),
        Paragraph(f"Generated {date.today().isoformat()} | ScholarSync AI", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Abstract", styles["Heading1"]),
        Paragraph(_escape(report.abstract), styles["BodyText"]),
        Spacer(1, 10),
    ]
    for section in report.sections:
        story.append(Paragraph(_escape(section.heading), styles["Heading1"]))
        story.extend(_parse_body(section.body, styles))
        story.append(Spacer(1, 8))
    story.append(Paragraph("References", styles["Heading1"]))
    if report.references:
        for item in report.references:
            rendered = _escape(item)
            for url in re.findall(r"https?://[^\s]+", item):
                clean_url = url.rstrip(".,")
                display = html_escape(clean_url)
                href = html_escape(clean_url, quote=True)
                rendered = rendered.replace(display, f'<link href="{href}" color="#185ABC">{display}</link>')
            story.append(Paragraph(rendered, styles["BodyText"]))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No records were returned by selected sources.", styles["BodyText"]))
    doc.build(story)
    return path
