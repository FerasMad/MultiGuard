#!/usr/bin/env python3
"""Build a self-contained HTML Technical Design Review.

- Converts the TDR Markdown to HTML (headings, tables, lists, blockquotes,
  inline bold/code, fenced code, and ```mermaid``` -> live diagrams).
- Appends an Appendix with the FULL source of the 5 pipeline classes.
- Mermaid + highlight.js are loaded from CDN (needed only to *view* diagrams /
  colour code; the document is readable offline regardless).
"""
from __future__ import annotations

import base64
import html
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "MultiGuardPipeline")
MD_PATH = os.path.join(PKG, "TECHNICAL_DESIGN_REVIEW.md")
OUT_HTML = os.path.join(ROOT, "reports", "Technical_Design_Review.html")

SOURCE_FILES = [
    ("semantic.py", "SemanticEncoder"),
    ("forensic_text.py", "ForensicTextEncoder"),
    ("image_forensic.py", "ImageForensicEncoder"),
    ("fusion.py", "FusionModule"),
    ("main_pipeline.py", "MainPipeline"),
]

_toc: list[tuple[int, str, str]] = []  # (level, text, slug)


def slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s]+", "-", s) or "sec"


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return text


def cell(text: str) -> str:
    return inline(text.replace("\\|", "|").strip())


def img_data_uri(src: str) -> str:
    """Resolve a local image path and return a base64 data URI (self-contained)."""
    candidates = [
        os.path.join(os.path.dirname(MD_PATH), src),
        os.path.join(ROOT, "reports", os.path.basename(src)),
        os.path.join(ROOT, src),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            ext = os.path.splitext(path)[1].lstrip(".") or "png"
            return f"data:image/{ext};base64,{b64}"
    return src  # fall back to the raw path if not found


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        # fenced code / mermaid
        if line.lstrip().startswith("```"):
            lang = line.lstrip()[3:].strip()
            i += 1
            buf: list[str] = []
            while i < n and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            code = "\n".join(buf)
            if lang == "mermaid":
                out.append(f'<div class="mermaid">{html.escape(code)}</div>')
            else:
                cls = f' class="language-{lang}"' if lang else ""
                out.append(f"<pre><code{cls}>{html.escape(code)}</code></pre>")
            continue

        # table
        if line.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = [cell(c) for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([cell(c) for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = ["<table><thead><tr>"] + [f"<th>{h}</th>" for h in header] + ["</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{(r[j] if j < len(r) else '')}</td>" for j in range(len(header))) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            txt = m.group(2)
            sg = slug(txt)
            if 2 <= lvl <= 3:
                _toc.append((lvl, txt, sg))
            out.append(f'<h{lvl} id="{sg}">{inline(txt)}</h{lvl}>')
            i += 1
            continue

        if re.match(r"^\s*---+\s*$", line):
            out.append("<hr/>")
            i += 1
            continue

        # standalone image  ![alt](src)
        im = re.match(r"^\s*!\[(.*?)\]\((.*?)\)\s*$", line)
        if im:
            alt, src = im.group(1), im.group(2)
            uri = img_data_uri(src)
            out.append(
                f'<figure><img src="{uri}" alt="{html.escape(alt)}"/>'
                f"<figcaption>{html.escape(alt)}</figcaption></figure>"
            )
            i += 1
            continue

        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append(f"<blockquote>{inline(' '.join(buf))}</blockquote>")
            continue

        if re.match(r"^\s*[-*]\s+", line):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*[-*]\s+", "", lines[i])))
                i += 1
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in buf) + "</ul>")
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*\d+\.\s+", "", lines[i])))
                i += 1
            out.append("<ol>" + "".join(f"<li>{b}</li>" for b in buf) + "</ol>")
            continue

        if not line.strip():
            i += 1
            continue

        out.append(f"<p>{inline(line)}</p>")
        i += 1

    return "\n".join(out)


CSS = """
:root{--fg:#1f2933;--muted:#52606d;--accent:#5b21b6;--line:#e4e7eb;--bg:#fff;--code:#f5f3ff;}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
  line-height:1.6;max-width:980px;margin:0 auto;padding:32px 24px 80px;background:var(--bg);}
h1{font-size:1.9rem;border-bottom:3px solid var(--accent);padding-bottom:.3em;margin-top:0}
h2{font-size:1.4rem;margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.25em}
h3{font-size:1.13rem;margin-top:1.5em;color:#3730a3}
h4{font-size:1rem;margin-top:1.2em;color:var(--muted)}
p,li{font-size:.96rem}
code{background:var(--code);color:#5b21b6;padding:.12em .35em;border-radius:4px;
  font-family:Consolas,Menlo,monospace;font-size:.86em}
pre{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;overflow:auto;font-size:.82rem;line-height:1.5}
pre code{background:none;color:inherit;padding:0;font-size:inherit}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.9rem}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f7f5ff}
tr:nth-child(even) td{background:#fafafa}
blockquote{border-left:4px solid var(--accent);margin:1em 0;padding:.4em 1em;background:#faf8ff;color:var(--muted)}
hr{border:none;border-top:1px solid var(--line);margin:2em 0}
.mermaid{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;margin:1em 0;text-align:center}
.toc{background:#faf8ff;border:1px solid var(--line);border-radius:8px;padding:14px 22px;margin:1.5em 0}
.toc a{color:var(--accent);text-decoration:none}
.toc a:hover{text-decoration:underline}
.toc .lvl3{margin-left:18px;font-size:.9em}
.meta{color:var(--muted);font-size:.86rem}
.filehdr{margin-top:1.6em;font-family:Consolas,monospace;font-weight:700;color:#3730a3}
figure{margin:1.2em 0;text-align:center}
figure img{max-width:560px;width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{color:var(--muted);font-size:.85rem;margin-top:.4em}
"""

CDN = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script>document.addEventListener('DOMContentLoaded',()=>{if(window.hljs)hljs.highlightAll();});</script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({startOnLoad:true, theme:'neutral'});
</script>
"""


def build() -> None:
    with open(MD_PATH, encoding="utf-8") as f:
        md = f.read()
    body = md_to_html(md)

    toc_items = []
    for lvl, txt, sg in _toc:
        cls = "lvl3" if lvl == 3 else "lvl2"
        toc_items.append(f'<div class="{cls}"><a href="#{sg}">{html.escape(txt)}</a></div>')
    toc_html = '<div class="toc"><strong>Contents</strong>' + "".join(toc_items) + "</div>"

    # appendix: full source of the 5 classes
    app = ['<hr/><h2 id="appendix-source">Appendix A — Source Code (the 5 classes)</h2>',
           "<p>Verbatim source of the five pipeline files under "
           "<code>phases/v4/src/v4/pipeline/</code>.</p>"]
    for fname, cls in SOURCE_FILES:
        with open(os.path.join(PKG, fname), encoding="utf-8") as f:
            src = f.read()
        app.append(f'<div class="filehdr">{fname} &mdash; class {cls}</div>')
        app.append(f'<pre><code class="language-python">{html.escape(src)}</code></pre>')

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MultiGuard Pluggable Pipeline — Technical Design Review</title>
<style>{CSS}</style>
{CDN}
</head>
<body>
<p class="meta">MultiGuard · Technical Design Review · pluggable pipeline (phases/v4/src/v4/pipeline)</p>
{toc_html}
{body}
{''.join(app)}
</body>
</html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(doc)
    print("WROTE:", OUT_HTML, f"({len(doc):,} bytes)")


if __name__ == "__main__":
    build()
