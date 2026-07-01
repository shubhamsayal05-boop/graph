"""Executive-summary report export (PDF via reportlab, plain-text fallback)."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO


def build_summary_report(meta: dict, mode_rows: list):
    """Build an AVL-style executive summary.

    ``mode_rows`` is a list of ``[mode, dr, events, weight]`` rows. Returns
    ``(bytes, mime, extension)`` — a PDF when reportlab is available, else text.
    """
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)
        styles = getSampleStyleSheet()
        head = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]
        story = [Paragraph("AVL-DRIVE-Style Executive Drivability Summary", styles["Title"]),
                 Spacer(1, 6), Paragraph(f"Generated: {generated}", styles["Normal"]),
                 Spacer(1, 12), Paragraph("Configuration &amp; File Metadata", styles["Heading2"])]
        mt = Table([["Field", "Value"]] + [[k, str(v)] for k, v in meta.items()], colWidths=[6 * cm, 9 * cm])
        mt.setStyle(TableStyle(head))
        story += [mt, Spacer(1, 16), Paragraph("Operation-Mode DRIVE Ratings (1-10)", styles["Heading2"])]
        st_tbl = Table([["Operation Mode", "DR", "Events", "Weight"]] + mode_rows,
                       colWidths=[7 * cm, 3 * cm, 3 * cm, 2 * cm])
        st_tbl.setStyle(TableStyle(head))
        story.append(st_tbl)
        doc.build(story)
        return buf.getvalue(), "application/pdf", "pdf"
    except Exception:
        lines = ["AVL-DRIVE-STYLE EXECUTIVE DRIVABILITY SUMMARY", "=" * 46,
                 f"Generated: {generated}", "", "CONFIGURATION & FILE METADATA", "-" * 46]
        lines += [f"{k:<26}: {v}" for k, v in meta.items()]
        lines += ["", "OPERATION-MODE DRIVE RATINGS (1-10)", "-" * 46]
        lines += [f"{r[0]:<26}: DR {r[1]:>5}  events {r[2]:>3}  weight {r[3]}" for r in mode_rows]
        return ("\n".join(lines) + "\n").encode("utf-8"), "text/plain", "txt"
