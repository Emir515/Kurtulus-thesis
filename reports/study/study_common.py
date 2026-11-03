"""Shared helpers for the notebook study guides (reports/study/build_nb0X_study.py).

A guide is a list of blocks. Each block shows one notebook output (its printed text and figures)
with a plain-language explanation and, where the reader asked about it, highlighted "Your question" boxes.
render() writes a self-contained HTML file; make_pdf() prints a light-theme copy to PDF with Microsoft Edge.
"""
import html, json, os, re, subprocess, sys

CSS = """
:root{--bg:#fbfbf9;--fg:#1c1c1a;--muted:#5d5c57;--card:#ffffff;--line:#e3e2dd;--accent:#2a6fc4;--warn:#9a5b00;--warnbg:#fff6e6;--code:#f2f1ec;--qa:#7a3fb8;--qabg:#f6effd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;--muted:#b3b2aa;--card:#1f1f1d;--line:#34332f;--accent:#6ea8f0;--warn:#f0b35a;--warnbg:#2b2416;--code:#262622;--qa:#c39bf0;--qabg:#271d33}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--muted:#b3b2aa;--card:#1f1f1d;--line:#34332f;--accent:#6ea8f0;--warn:#f0b35a;--warnbg:#2b2416;--code:#262622;--qa:#c39bf0;--qabg:#271d33}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 80px}h1{font-size:1.9rem;margin:.2em 0}h2{margin-top:2.2em;padding-top:.6em;border-top:2px solid var(--line)}
h3{margin:0 0 .4em}.lead{color:var(--muted)}.toc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 18px}
.toc ol{margin:.3em 0;padding-left:1.2em}.toc a{color:var(--accent);text-decoration:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:18px 0}
.row{margin:.5em 0}.row b.k{display:inline-block;min-width:8.5em;color:var(--muted);font-weight:600}
.careful{background:var(--warnbg);border-left:4px solid var(--warn);padding:8px 12px;border-radius:6px;margin-top:.6em}
.qa{background:var(--qabg);border-left:4px solid var(--qa);padding:8px 12px;border-radius:6px;margin-top:.7em}
.qa .q{font-weight:700;color:var(--qa)}.qa .q::before{content:"Your question: "}.qa .a{margin-top:.2em}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0;font-size:.92em;color:var(--muted)}
.legend span{padding:2px 8px;border-radius:4px}
figure{margin:14px 0}figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
details{margin-top:10px}summary{cursor:pointer;color:var(--accent)}pre{background:var(--code);padding:10px 12px;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.4}
code{background:var(--code);padding:0 4px;border-radius:4px;overflow-wrap:anywhere}ul,ol{margin:.2em 0 .2em 1.1em;padding:0}
table.t{border-collapse:collapse;margin:.4em 0;font-size:.93em}table.t td,table.t th{border:1px solid var(--line);padding:3px 8px;text-align:left}
@media print{details{display:block}details>summary{display:none}.card,.qa{break-inside:avoid-page}}
"""


class Notebook:
    def __init__(self, path):
        self.path = path
        self.cells = json.load(open(path, encoding='utf-8'))['cells']

    def cell(self, key):
        hits = [c for c in self.cells if c['cell_type'] == 'code' and key in ''.join(c['source'])]
        if len(hits) != 1:
            sys.exit(f'{os.path.basename(self.path)}: expected one code cell containing "{key}", found {len(hits)}')
        return hits[0]


def text_out(c, max_lines=80):
    t = ''.join(''.join(o.get('text', '')) for o in c.get('outputs', [])).rstrip('\n')
    lines = t.split('\n')
    if max_lines and len(lines) > max_lines:
        t = '\n'.join(lines[:max_lines]) + f'\n… (shortened: {len(lines) - max_lines} more lines in the notebook)'
    return t


def images(c):
    return [o['data']['image/png'] for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]


def render(nb, blocks, qa, title, lead_html, summary_html, out_path):
    """blocks: (id, title, code-cell key or None for a part heading, explanation dict, options).
    qa: {block id: [(question, answer html), ...]}: the reader's own questions, shown highlighted."""
    parts, toc, n_q = [], [], 0
    unused = set(qa) - {b[0] for b in blocks}
    if unused:
        sys.exit(f'questions attached to unknown blocks: {sorted(unused)}')
    for sid, btitle, key, ex, opt in blocks:
        if key is None:
            toc.append(f'<li><a href="#{sid}">{html.escape(btitle)}</a></li>')
            parts.append(f'<h2 id="{sid}">{html.escape(btitle)}</h2>')
            continue
        opt = opt or {}
        keys = key if isinstance(key, (list, tuple)) else [key]
        cells = [nb.cell(k) for k in keys]
        rows = []
        for k, label in (('what', 'What it is'), ('read', 'How to read'), ('shows', 'What it shows'), ('why', 'Why it matters')):
            if ex.get(k):
                rows.append(f'<div class="row"><b class="k">{label}</b> {ex[k]}</div>')
        if ex.get('careful'):
            rows.append(f'<div class="careful"><b>Be careful:</b> {ex["careful"]}</div>')
        for q, a in qa.get(sid, []):
            n_q += 1
            rows.append(f'<div class="qa"><div class="q">{html.escape(q)}</div><div class="a">{a}</div></div>')
        figs = ''.join(f'<figure><img alt="{html.escape(btitle)}" src="data:image/png;base64,{im}"></figure>'
                       for c in cells for im in images(c))
        out = '\n'.join(t for t in (text_out(c, opt.get('max_lines', 15)) for c in cells) if t)
        raw = f'<details><summary>Show the printed output</summary><pre>{html.escape(out)}</pre></details>' if out and not opt.get('no_text') else ''
        parts.append(f'<section class="card" id="{sid}"><h3>{html.escape(btitle)}</h3>{"".join(rows)}{figs}{raw}</section>')

    legend = ('<div class="legend"><span class="careful">Be careful = limitations</span>'
              '<span class="qa">Your question = something you asked while going through the notebook</span></div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="lead">{lead_html}</p>{legend}
<div class="toc"><b>Contents</b><ol>{''.join(toc)}</ol></div>
{''.join(parts)}
{summary_html}
</main></body></html>"""
    open(out_path, 'w', encoding='utf-8').write(page)
    print(f'Study guide written: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, {n_q} of your questions)')
    return out_path


EDGE = [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', r'C:\Program Files\Microsoft\Edge\Application\msedge.exe']


def make_pdf(html_path, pdf_path, tmp_dir):
    """Print a light-theme copy (all outputs expanded, long lines wrapped) to PDF with headless Edge."""
    edge = next((p for p in EDGE if os.path.exists(p)), None)
    if edge is None:
        print('Edge not found: PDF skipped')
        return None
    page = open(html_path, encoding='utf-8').read()
    page = page.replace('<html lang="en">', '<html lang="en" data-theme="light">', 1)
    page = page.replace('<details>', '<details open>')
    page = page.replace('</style>', 'pre{white-space:pre-wrap;word-break:break-word;overflow-x:visible}</style>', 1)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, os.path.basename(html_path).replace('.html', '_print.html'))
    open(tmp, 'w', encoding='utf-8').write(page)
    url = 'file:///' + os.path.abspath(tmp).replace('\\', '/')
    subprocess.run([edge, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--virtual-time-budget=20000',
                    f'--print-to-pdf={os.path.abspath(pdf_path)}', url], check=True, capture_output=True, timeout=300)
    print(f'PDF written: {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.1f} MB)')
    return pdf_path
