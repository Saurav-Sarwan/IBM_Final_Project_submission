"""
Embeds the 3 dashboard screenshots into DMart_Report.docx
using python-docx. Run after main.py has been executed.
"""

import sys
import subprocess

# Install python-docx if not present
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx", "-q"])
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

import os

SCREENSHOTS = [
    ("outputs/screenshot_executive.png",
     "Dashboard Page 1 — Executive Summary",
     "6 KPI cards, 5 Key Findings strip, Products-by-Category bar chart and Discount Depth Distribution doughnut chart."),
    ("outputs/screenshot_insights.png",
     "Dashboard Page 2 — Risks & Opportunities",
     "3 Risk cards with supporting charts (extreme discount by category, top promo subcategories), 3 Opportunity cards with assortment gap and price gap charts."),
    ("outputs/screenshot_actions.png",
     "Dashboard Page 3 — Recommended Actions",
     "5 numbered action cards with horizon labels, Action Impact vs Effort bubble matrix, and Category Discount % comparison chart."),
]

REPORT_PATH = "DMart_Report.docx"
OUT_PATH = "DMart_Report.docx"

def add_screenshot_section(doc, img_path, title, description, section_num):
    """Append a screenshot section to the document."""
    # Section heading
    h = doc.add_heading(f"{section_num}. Screenshot — {title}", level=2)
    h.runs[0].font.size = Pt(13)
    h.runs[0].font.color.rgb = RGBColor(0x1a, 0x3a, 0x5c)

    # Description paragraph
    p = doc.add_paragraph(description)
    p.runs[0].font.size = Pt(11)
    p.runs[0].font.color.rgb = RGBColor(0x57, 0x60, 0x6a)
    p.paragraph_format.space_after = Pt(8)

    # Insert image (scaled to ~6.5 inches wide to fit A4)
    if os.path.exists(img_path):
        pic_para = doc.add_paragraph()
        pic_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = pic_para.runs[0] if pic_para.runs else pic_para.add_run()
        run.add_picture(img_path, width=Inches(6.3))
        pic_para.paragraph_format.space_after = Pt(14)
        print(f"  [embedded] {img_path}")
    else:
        p2 = doc.add_paragraph(f"[Image not found: {img_path}]")
        p2.runs[0].font.color.rgb = RGBColor(0xe7, 0x4c, 0x3c)
        print(f"  [missing] {img_path}")

    # Caption
    cap = doc.add_paragraph(f"Figure {section_num - 6}: {title}")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(10)
    cap.runs[0].font.color.rgb = RGBColor(0x57, 0x60, 0x6a)
    cap.paragraph_format.space_after = Pt(16)


def main():
    if not os.path.exists(REPORT_PATH):
        print(f"ERROR: {REPORT_PATH} not found. Run main.py first.")
        sys.exit(1)

    doc = Document(REPORT_PATH)

    # Append screenshot sections
    for i, (img, title, desc) in enumerate(SCREENSHOTS, start=7):
        add_screenshot_section(doc, img, title, desc, i)

    doc.save(OUT_PATH)
    print(f"\n[DONE] Screenshots embedded → {OUT_PATH}")


if __name__ == "__main__":
    main()
