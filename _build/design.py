# -*- coding: utf-8 -*-
"""Publish the research projects under C:\\Users\\Admin\\Desktop\\design as blog content.

Each project becomes a browsable static subsite at /design/<slug>/ plus one entry in
the blog list. Only self-authored text/markup is published:

  * PDFs          - 1.14 GB of paper PDFs, and third-party papers as well; not ours to host
  * raw/          - scraped conference pages (cvpr2026_all.html is 11 MB of other people's text)
  * __pycache__, *.pyc, *.bin, *.whl, *.tex, session dumps  - build cruft
  * files > MAX_FILE_MB

The two saved WeChat / 公众号 articles (Xspark AI 丁文伯, 斯坦福吴佳俊 ECCV) are
deliberately excluded: they are other people's writing, not blog posts of ours.
"""

import io
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_ROOT = r"C:\Users\Admin\Desktop\design"
OUT_ROOT = os.path.join(ROOT, "design")

MAX_FILE_MB = 6.0

INCLUDE_EXT = {
    ".html", ".htm", ".css", ".js", ".mjs", ".json", ".md", ".markdown",
    ".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".ico",
    ".woff", ".woff2", ".ttf", ".csv", ".yml", ".yaml", ".txt",
}
EXCLUDE_DIRS = {
    "pdf", "pdfs", "raw", "tmp_cvf", "session-backups", "__pycache__",
    "node_modules", ".git", "_files",
}
SKIP_PREFIX = ("_work_", "tmp_")

ARXIV_PDF = re.compile(r'(?:\.\./)*(?:pdfs?/|papers/pdf/)?(\d{4}\.\d{4,5})\.pdf')

# Saved copies of arXiv paper pages: somebody else's full text, and they also pull
# /static/browse/*.css plus figure directories we do not have. Skip them.
SAVED_PAPER = re.compile(r'arxiv-html-papers|/static/browse/|name="citation_title"', re.I)


def _is_saved_paper_page(path):
    try:
        with io.open(path, "rb") as fh:
            head = fh.read(4096).decode("utf-8", "ignore")
    except OSError:
        return False
    return bool(SAVED_PAPER.search(head))

# slug, source subpath (relative to SRC_ROOT, "" = the folder itself), meta
from gen.data_design import DESIGN  # noqa: E402,F401

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="icon" href="/assets/site/favicon.svg" type="image/svg+xml">
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #fff; color: #0c0e14;
         font: 16px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #08090c; color: #eef1f6; }} }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 28px 22px 80px; }}
  a {{ color: #2447d8; }}
  @media (prefers-color-scheme: dark) {{ a {{ color: #8ea6ff; }} }}
  .bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 26px; }}
  .bar a {{ font-size: 13.5px; font-weight: 600; text-decoration: none;
            border: 1px solid currentColor; border-radius: 999px; padding: 6px 13px; }}
  h1 {{ font-size: 1.7rem; line-height: 1.25; margin: 0 0 8px; }}
  p.lead {{ color: #5c6575; margin: 0 0 28px; }}
  @media (prefers-color-scheme: dark) {{ p.lead {{ color: #98a1b1; }} }}
  ul {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 2px; }}
  li a {{ display: block; padding: 11px 14px; border-radius: 10px; text-decoration: none;
          border: 1px solid rgba(128,128,128,.22); font-size: 14.5px; }}
  li a:hover {{ background: rgba(128,128,128,.09); }}
  .meta {{ font: 12px ui-monospace, monospace; color: #8b93a3; margin-left: 8px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="bar"><a href="/">← 返回主页</a><a href="/blog/">博客</a></div>
  <h1>{title}</h1>
  <p class="lead">{lead}</p>
  <ul>
{items}
  </ul>
</div>
</body>
</html>
"""


def _slug_ok(name):
    return not name.startswith(SKIP_PREFIX)


def _rel_size(path):
    try:
        return os.path.getsize(path) / 1048576.0
    except OSError:
        return 0.0


def _rewrite_pdf_links(text):
    """Point arXiv-id PDF links at arxiv.org instead of a file we do not publish."""
    def repl(m):
        return "https://arxiv.org/abs/" + m.group(1)
    return ARXIV_PDF.sub(repl, text)


def _plan(src_dir, skip=()):
    """Yield (abs_path, rel_path) for every publishable file."""
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in EXCLUDE_DIRS and d not in skip and not d.startswith("."))
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in INCLUDE_EXT:
                continue
            if not _slug_ok(fn):
                continue
            abs_p = os.path.join(dirpath, fn)
            rel = os.path.relpath(abs_p, src_dir)
            if _rel_size(abs_p) > MAX_FILE_MB:
                continue
            if ext in (".html", ".htm") and _is_saved_paper_page(abs_p):
                continue
            yield abs_p, rel


def build(report=False):
    made = []
    skipped = []
    # start clean so files that stop qualifying do not linger from an older run
    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT, ignore_errors=True)
    for entry in DESIGN:
        src = os.path.join(SRC_ROOT, entry["src"].replace("/", os.sep))
        if not os.path.isdir(src):
            skipped.append((entry["slug"], "missing source"))
            continue
        dst_root = os.path.join(OUT_ROOT, entry["slug"])
        files = list(_plan(src, tuple(entry.get("skip", ()))))
        if not files:
            skipped.append((entry["slug"], "nothing publishable"))
            continue
        total = 0
        names = []
        for abs_p, rel in files:
            dst = os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if abs_p.lower().endswith((".html", ".htm")):
                with io.open(abs_p, "rb") as fh:
                    raw = fh.read()
                text = raw.decode("utf-8", "surrogateescape")
                if ".pdf" in text:
                    text = _rewrite_pdf_links(text)
                    raw = text.encode("utf-8", "surrogateescape")
                with io.open(dst, "wb") as fh:
                    fh.write(raw)
            else:
                shutil.copy2(abs_p, dst)
            total += os.path.getsize(dst)
            names.append(rel)
        # a browsable entry point
        if not os.path.exists(os.path.join(dst_root, "index.html")):
            links = [n for n in names if n.lower().endswith((".html", ".md"))]
            items = "\n".join(
                '    <li><a href="{h}">{t}<span class="meta">{r}</span></a></li>'.format(
                    h=n.replace("\\", "/"), t=n.replace("\\", "/"), r="")
                for n in links[:200])
            html = INDEX_TEMPLATE.format(
                title=entry["title"], lead=entry["summary"], items=items)
            p = os.path.join(dst_root, "index.html")
            os.makedirs(dst_root, exist_ok=True)
            with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(html)
            total += os.path.getsize(p)
        made.append({"entry": entry, "files": len(files), "mb": total / 1048576.0})
    return made, skipped


if __name__ == "__main__":
    made, skipped = build()
    for m in made:
        print("%-22s %4d files  %7.2f MB" % (m["entry"]["slug"], m["files"], m["mb"]))
    for s in skipped:
        print("SKIPPED %-20s %s" % (s[0], s[1]))
    print("total: %.2f MB" % sum(m["mb"] for m in made))
