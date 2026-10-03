"""Render reports/report.md to reports/report.pdf.

Usage: python src/make_report_pdf.py

Requires `markdown`, `beautifulsoup4` and `xhtml2pdf` (see requirements.txt)
and a DejaVu Sans install (ships with most Linux distros; needed so Greek
letters / arrows in the report render instead of falling back to tofu
boxes). xhtml2pdf's CSS support is limited (no word-break, @font-face must
read from a path under the repo root, inline-code backgrounds can overflow
a narrow table cell) -- the workarounds below exist because of those
constraints, not by choice.
"""
import re
import shutil
import sys
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from xhtml2pdf import pisa

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "reports" / "report.md"
OUT = REPO / "reports" / "report.pdf"
BASE_DIR = SRC.parent  # for resolving relative image paths like figures/x.png
FONT_DIR = REPO / ".fonts_tmp"  # must be under REPO: xhtml2pdf sandboxes @font-face reads

FONT_CANDIDATES = {
    "DejaVuSans.ttf": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "DejaVuSans-Bold.ttf": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "DejaVuSans-Oblique.ttf": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ],
    "DejaVuSansMono.ttf": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
}


def stage_fonts() -> None:
    FONT_DIR.mkdir(exist_ok=True)
    missing = []
    for name, candidates in FONT_CANDIDATES.items():
        dest = FONT_DIR / name
        if dest.exists():
            continue
        found = next((c for c in candidates if Path(c).exists()), None)
        if found is None:
            missing.append(name)
        else:
            shutil.copy(found, dest)
    if missing:
        sys.exit(
            "Missing DejaVu font file(s): "
            + ", ".join(missing)
            + ". Install the `fonts-dejavu-core` package (or point "
            "FONT_CANDIDATES in this script at wherever they live)."
        )


def build_html() -> str:
    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "toc", "nl2br"],
    )
    soup = BeautifulSoup(html_body, "html.parser")

    # xhtml2pdf's CSS support is limited; force image width via an HTML
    # attribute (reportlab's image flowable reads this reliably) instead of
    # relying on CSS max-width, which it silently ignores.
    for img in soup.find_all("img"):
        img["width"] = "215"

    # A table with many columns (the 13-column run-configuration table)
    # needs a smaller font/padding and more aggressive text wrapping than
    # the rest of the document, or cells overlap illegibly. Tag wide
    # tables, then handle their <code> spans specially below.
    wide_cols_threshold = 9
    wide_tables = set()
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        ncols = len(header_row.find_all(["th", "td"])) if header_row else 0
        if ncols >= wide_cols_threshold:
            wide_tables.add(id(table))
            table["style"] = "font-size: 6pt;"
            for cell in table.find_all(["th", "td"]):
                cell["style"] = "padding: 1.5pt 2pt;"

    # xhtml2pdf does not implement word-break/word-wrap, and only wraps on
    # actual whitespace (a zero-width space is invisible but has no glyph
    # in the embedded font and renders as a tofu box, so that's not a
    # usable substitute either). Insert real spaces at natural break
    # characters inside long <code> spans (repo ids, nested file paths,
    # "flag=value" settings) so they wrap instead of overflowing/
    # overlapping the next cell.
    def break_code_text(text: str, in_wide_table: bool) -> str:
        threshold = 10 if in_wide_table else 23
        if len(text) <= threshold:
            return text
        return re.sub(r"([/\-=_])", "\\1 ", text)

    for code in soup.find_all("code"):
        parent_table = code.find_parent("table")
        in_wide = parent_table is not None and id(parent_table) in wide_tables
        if code.string is not None:
            code.string.replace_with(break_code_text(code.string, in_wide))
        if in_wide:
            # The <code> span's shaded background box appears to be sized
            # from the pre-wrap text extent, which bleeds into the next
            # column on a cell's first line even after the font-size and
            # break-point fixes above. Dropping the code styling
            # (monospace font + background) in this one densely-packed
            # table and rendering it as plain text avoids that; plain text
            # wraps correctly elsewhere already.
            code.unwrap()

    return str(soup)


CSS = """
@font-face { font-family: "DejaVu"; src: url("../.fonts_tmp/DejaVuSans.ttf"); }
@font-face { font-family: "DejaVu"; src: url("../.fonts_tmp/DejaVuSans-Bold.ttf"); font-weight: bold; }
@font-face { font-family: "DejaVu"; src: url("../.fonts_tmp/DejaVuSans-Oblique.ttf"); font-style: italic; }
@font-face { font-family: "DejaVuMono"; src: url("../.fonts_tmp/DejaVuSansMono.ttf"); }

@page {
    size: letter;
    margin: 2cm 1.8cm 2.2cm 1.8cm;
    @frame footer_frame {
        -pdf-frame-content: footer_content;
        bottom: 1cm; margin-left: 1.8cm; margin-right: 1.8cm; height: 1cm;
    }
}

body { font-family: "DejaVu"; font-size: 9.5pt; line-height: 1.4; color: #1a1a1a; }

h1 { font-size: 19pt; margin-top: 0; margin-bottom: 10pt; border-bottom: 1.5pt solid #333; padding-bottom: 6pt; }
h2 { font-size: 14pt; margin-top: 18pt; margin-bottom: 6pt; color: #16324f; border-bottom: 0.75pt solid #aaa; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin-top: 14pt; margin-bottom: 5pt; color: #16324f; }
h4 { font-size: 10.5pt; margin-top: 10pt; margin-bottom: 4pt; color: #333; }

p { margin: 5pt 0; text-align: justify; }
a { color: #1a5276; }

table { width: 100%; border-collapse: collapse; margin: 8pt 0; font-size: 8.3pt; }
th, td { border: 0.5pt solid #999; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background-color: #e8eef3; font-weight: bold; }
tr:nth-child(even) td { background-color: #f7f9fb; }

code { font-family: "DejaVuMono"; background-color: #f0f0f0; padding: 1pt 2pt; font-size: 8pt; }
pre { font-family: "DejaVuMono"; background-color: #f0f0f0; padding: 6pt; font-size: 7.6pt;
      border: 0.5pt solid #ccc; white-space: pre-wrap; }
pre code { background-color: transparent; padding: 0; }

blockquote { margin: 6pt 0 6pt 10pt; padding-left: 8pt; border-left: 2pt solid #999; color: #444; }

img { width: 215pt; height: auto; display: block; margin-bottom: 3pt; }

hr { border: none; border-top: 0.5pt solid #ccc; margin: 10pt 0; }

ul, ol { margin: 4pt 0 4pt 14pt; padding: 0; }
li { margin: 2pt 0; }
"""


def link_callback(uri: str, rel: str) -> str:
    # Resolve @font-face/img relative paths to absolute filesystem paths,
    # bypassing xhtml2pdf's default same-dir-as-doc sandbox.
    if uri.startswith("/"):
        return uri
    return str((BASE_DIR / uri).resolve())


def main() -> None:
    stage_fonts()
    html_body = build_html()
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div id="footer_content" style="text-align:center; font-size:7.5pt; color:#777;">
SimCSE on an SNLI subset &mdash; U2T02 &mdash; page <pdf:pagenumber/> of <pdf:pagecount/>
</div>
{html_body}
</body></html>"""

    with open(OUT, "wb") as f:
        result = pisa.CreatePDF(html_doc, dest=f, link_callback=link_callback, encoding="utf-8")

    shutil.rmtree(FONT_DIR, ignore_errors=True)

    if result.err:
        print(f"xhtml2pdf reported {result.err} error(s)", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
