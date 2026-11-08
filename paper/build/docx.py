"""Builds the REEPE paper (.docx) from the English Markdown draft and the REEPE IEEE template.
usage: docx.py TEMPLATE.docx PAPER.md REFS.md OUT.docx [en|ru]
The language sets the fixed labels (Abstract, Keywords, References) and the caption numbering
(Fig./TABLE or Рис./ТАБЛИЦА); the text itself comes from PAPER.md.

TEMPLATE must be a transitional .docx: the REEPE template ships as Strict OOXML, which python-docx
can not open, so the Makefile converts it with LibreOffice first.

Markdown subset of PAPER.md (everything before the first '---' line is a draft note and skipped):
  ## Title / ## Index Terms / ## Abstract   front matter (or ## Название / ## Ключевые слова /
  ## Authors                                 ## Аннотация / ## Авторы), then its paragraphs or items
  ## I. Introduction, ### A. Related Work    Heading 1 / Heading 2, the template numbers them
  - item                                     bullet list
  `x_{i} = y^{2}`  (1)                       display equation with its number
  Table: caption (Таблица: ...), pipe table  table head + table
  ![caption](path)                           figure + figure caption
  `code`, **bold**, *italic*, $x_{i}$       inline (code in Courier, $..$ as math)
  [DATA ...], [CHECK ...]                    kept and highlighted, so nothing unfinished slips through
"""
import copy
import pathlib
import re
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

FRONT = {"title": "title", "название": "title", "index terms": "keywords",
         "ключевые слова": "keywords", "abstract": "abstract", "аннотация": "abstract",
         "authors": "authors", "авторы": "authors"}
LABELS = {"en": {"abstract": "Abstract—", "keywords": "Keywords—", "refs": "References",
                 "fig": "Fig. %1.", "table": "TABLE %1. "},
          "ru": {"abstract": "Аннотация—", "keywords": "Ключевые слова—", "refs": "Список литературы",
                 "fig": "Рис. %1.", "table": "ТАБЛИЦА %1. "}}
TABLE_PREFIX = re.compile(r"^(Table|Таблица): ")
COLUMN_MM = 85  # one column of the A4 template: (210 - 2*15.8 - 6.35) / 2

# ==========================
# ===  Markdown parsing  ===
# ==========================


def blocks(md):
    """Yields (kind, payload) blocks in document order."""
    lines = md.split("\n")
    if "---" in lines:
        lines = lines[lines.index("---") + 1:]
    i, caption = 0, None
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line == "---":
            i += 1
            continue
        if m := re.match(r"^(#{2,3}) (.*)$", line):
            yield ("h" + str(len(m.group(1)) - 1), m.group(2).strip())
        elif m := re.match(r"^`([^`]+)`\s+\((\d+)\)$", line):
            yield ("eq", (m.group(1), m.group(2)))
        elif m := re.match(r"^!\[(.*)\]\((.*)\)$", line):
            yield ("fig", (m.group(1), m.group(2)))
        elif TABLE_PREFIX.match(line):
            caption = TABLE_PREFIX.sub("", line)
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            yield ("table", (caption, rows))
            caption = None
            continue
        elif line.startswith("- "):
            yield ("li", line[2:])
        elif m := re.match(r"^(\d+)\. (.*)$", line):
            yield ("ol", (m.group(1), m.group(2)))
        else:
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(
                    r"^(#|- |\d+\. |\||`[^`]+`\s+\(\d+\)$|!\[|Table: |Таблица: )", lines[i + 1]):
                i += 1
                para.append(lines[i].rstrip())
            yield ("p", " ".join(para))
        i += 1


INLINE = re.compile(r"(\$[^$]+\$|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[(?:DATA|CHECK)[^\]]*\])")


def add_inline(par, text, bold=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("$"):
            add_math(par, part[1:-1])
            continue
        if part.startswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Courier New"
            run.font.size = Pt(9)
        elif part.startswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*"):
            run = par.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("[DATA") or part.startswith("[CHECK"):
            run = par.add_run(part)
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            run = par.add_run(part)
        if bold:
            run.bold = True


EQ_TOKEN = re.compile(r"([_^])(\{[^}]*\}|[A-Za-z0-9]+)")


def times(run):
    """The equation style is set in Symbol, which turns Latin letters into Greek ones."""
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:cs"), "Times New Roman")
    return run


def add_equation(par, expr, number):
    """Center tab, the expression, right tab, (n)."""
    times(par.add_run("\t"))
    add_math(par, expr)
    times(par.add_run(f"\t({number})"))


def add_math(par, expr):
    """Sub/superscripts from _x, ^x, _{..}, ^{..}; single letters in italic, the rest upright."""
    pos = 0
    for m in EQ_TOKEN.finditer(expr):
        add_eq_text(par, expr[pos:m.start()])
        body = m.group(2).strip("{}")
        run = times(par.add_run(body))
        run.font.subscript = m.group(1) == "_"
        run.font.superscript = m.group(1) == "^"
        run.italic = body.isalpha() and len(body) == 1
        pos = m.end()
    add_eq_text(par, expr[pos:])


def add_eq_text(par, text):
    for part in re.split(r"([A-Za-zλμ][A-Za-zλμ0-9]*)", text):
        if part:
            run = times(par.add_run(part))
            run.italic = part.isalpha() and len(part) == 1


# ==========================
# ===  Template handling ===
# ==========================


def body_anchor(doc):
    """Removes the sample body (Abstract .. References) and returns the section-break paragraph
    that closes the two-column section; new content goes right before it."""
    els = list(doc.element.body.iterchildren())
    start = next(i for i, el in enumerate(els) if style_of(el) == "Abstract")
    end = next(i for i in range(start, len(els)) if has_sect(els[i]))
    for el in els[start:end]:
        doc.element.body.remove(el)
    return els[end]


def style_of(el):
    ps = el.find(qn("w:pPr") + "/" + qn("w:pStyle")) if el.tag == qn("w:p") else None
    return ps.get(qn("w:val")) if ps is not None else None


def has_sect(el):
    return el.tag == qn("w:p") and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None


def set_text(par, text):
    for run in par.runs[1:]:
        run._element.getparent().remove(run._element)
    if par.runs:
        par.runs[0].text = text
    else:
        par.add_run(text)


def set_widths(tbl, rows):
    """Column widths follow the text: a column gets at least its longest word, so Russian words
    do not break letter by letter, and long cells get room to wrap into two or three lines."""
    plain = [[re.sub(r"[`*$]|_\{|\}", "", c) for c in r] for r in rows]
    weights = []
    for c in range(len(plain[0])):
        cells = [r[c] for r in plain if c < len(r)]
        longest_word = max((len(w) for cell in cells for w in cell.split()), default=1)
        longest_cell = max((len(cell) for cell in cells), default=1)
        # +2 characters cover the cell margins of the template table.
        weights.append(max(longest_word, min(longest_cell, 30) / 2.2, 3) + 2)
    total = sum(weights)
    for c, wgt in enumerate(weights):
        width = Mm(COLUMN_MM * wgt / total)
        tbl.columns[c].width = width
        for cell in tbl.columns[c].cells:
            cell.width = width


class Writer:
    def __init__(self, doc, anchor, base, table_pr):
        self.doc, self.anchor, self.base, self.table_pr = doc, anchor, base, table_pr

    def para(self, style, text="", bold=False):
        par = self.doc.add_paragraph(style=style)
        self.anchor.addprevious(par._p)
        if text:
            add_inline(par, text, bold)
        return par

    def table(self, caption, rows):
        if caption:
            self.para("table head", caption).paragraph_format.keep_with_next = True
        ncol = max(len(r) for r in rows)
        tbl = self.doc.add_table(rows=len(rows), cols=ncol)
        if self.table_pr is not None:
            tbl._tbl.remove(tbl._tbl.tblPr)
            tbl._tbl.insert(0, copy.deepcopy(self.table_pr))
        for r, cells in enumerate(rows):
            for c, text in enumerate(cells):
                par = tbl.cell(r, c).paragraphs[0]
                par.style = self.doc.styles["table col head" if r == 0 else "table copy"]
                # Every row but the last keeps with the next one, so the table is not split.
                par.paragraph_format.keep_with_next = r < len(rows) - 1
                add_inline(par, text)
        set_widths(tbl, rows)
        self.anchor.addprevious(tbl._tbl)
        self.para("Body Text").paragraph_format.space_after = Pt(0)

    def figure(self, caption, path):
        par = self.para("Normal")
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True  # the caption stays with its figure
        par.add_run().add_picture(str(self.base / path), width=Mm(COLUMN_MM))
        self.para("figure caption", caption)


# ==========================
# ===  Build             ===
# ==========================


def caption_numbering(doc, labels):
    """The template numbers captions through numbering.xml ('Fig. %1.', 'TABLE %1. ')."""
    for t in doc.part.numbering_part.element.iter(qn("w:lvlText")):
        val = t.get(qn("w:val"), "")
        if val.startswith("Fig."):
            t.set(qn("w:val"), labels["fig"])
        elif val.startswith("TABLE"):
            t.set(qn("w:val"), labels["table"])


def build(template, paper_md, refs_md, out, lang="en"):
    labels = LABELS[lang]
    doc = docx.Document(template)
    caption_numbering(doc, labels)
    base = pathlib.Path(paper_md).parent
    front, body = {}, []
    current = None
    for kind, payload in blocks(pathlib.Path(paper_md).read_text()):
        if kind == "h1" and payload.lower() in FRONT:
            current = FRONT[payload.lower()]
            front[current] = []
            continue
        if kind == "h1":
            current = None
        if current:
            front[current].append(payload)
        else:
            body.append((kind, payload))

    paras = list(doc.paragraphs)
    title = next(p for p in paras if p.style.name == "paper title")
    set_text(title, front["title"][0].strip("*"))
    fill_authors(doc, front.get("authors", []))

    # Borders of the sample table, taken before the sample body goes away.
    table_pr = copy.deepcopy(doc.tables[0]._tbl.tblPr) if doc.tables else None
    anchor = body_anchor(doc)
    w = Writer(doc, anchor, base, table_pr)
    abstract = w.para("Abstract")
    run = abstract.add_run(labels["abstract"])
    run.italic = True
    add_inline(abstract, " ".join(front.get("abstract", ["[DATA abstract]"])))
    kw = w.para("Keywords")
    run = kw.add_run(labels["keywords"])
    run.italic = True
    add_inline(kw, " ".join(front.get("keywords", [])))

    for kind, payload in body:
        if kind == "h1":
            w.para("Heading 1", re.sub(r"^[IVX]+\.\s+", "", payload))
        elif kind == "h2":
            w.para("Heading 2", re.sub(r"^[A-ZА-ЯЁ]\.\s+", "", payload))
        elif kind == "p":
            w.para("Body Text", payload)
        elif kind == "li":
            w.para("bullet list", payload)
        elif kind == "ol":
            w.para("Body Text", f"{payload[0]}) {payload[1]}")
        elif kind == "eq":
            add_equation(w.para("equation"), *payload)
        elif kind == "table":
            w.table(*payload)
        elif kind == "fig":
            w.figure(*payload)

    w.para("Heading 5", labels["refs"])
    for kind, payload in blocks(pathlib.Path(refs_md).read_text().replace("# References", "---", 1)):
        if kind == "ol":
            w.para("references", payload[1])
    doc.save(out)


def fill_authors(doc, lines):
    """The template's author block is the run of Normal paragraphs between the title section and
    the abstract; it takes the lines of '## Authors' one per paragraph."""
    block = [p for p in doc.paragraphs if p.style.name == "Normal" and p.text.strip()]
    block = [p for p in block if p._p.getparent() is doc.element.body][:6]
    lines = lines or ["[Author name]", "[Department]", "Bauman Moscow State Technical University",
                      "Moscow, Russian Federation", "[e-mail]"]
    for par, text in zip(block, lines + [""] * len(block)):
        if text:
            # The sample lines start with superscript affiliation marks; drop every run, so the
            # new text does not inherit them.
            for run in par.runs:
                run._element.getparent().remove(run._element)
            add_inline(par, text)
        else:
            par._p.getparent().remove(par._p)


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    build(*sys.argv[1:])
