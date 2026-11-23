# -*- coding: utf-8 -*-
"""Publish the research projects that live outside this repository.

Source roots come from ``_build/sources.local.json`` (gitignored), so the private
paths used on this machine are not part of the published repository:

* ``design`` — personal research sites
* ``demos``  — research demos; only entries with ``"images": "webp"`` are used

Only ``demos`` entries marked with ``"images": "webp"`` get their screenshots
re-encoded; the ``design`` sources already ship WebP.

What is never published, from either root:

* ``*.pdf``            — paper corpora (1.1 GB in demos alone); not ours to host
* ``raw/``             — scraped conference pages
* saved arXiv paper pages (content sniffed, not just the filename)
* build caches, ``__pycache__``, session dumps, ``*.mp4``
* anything over ``MAX_FILE_MB``

Every published project is also scrubbed for company identifiers. The demos that
were included contain none, but the scrub runs anyway and the build prints how
many replacements it made, so a regression would be visible.
"""

import io
import os
import re
import shutil

from gen.data_design import DESIGN  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_ROOT = os.path.join(ROOT, "design")

def _load_sources():
    import json
    path = os.path.join(HERE, "sources.local.json")
    try:
        with io.open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


SRC_ROOTS = _load_sources()

MAX_FILE_MB = 6.0
IMG_MAX_W = 1500
IMG_QUALITY = 80

INCLUDE_EXT = {
    ".html", ".htm", ".css", ".js", ".mjs", ".json", ".md", ".markdown",
    ".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".ico",
    ".woff", ".woff2", ".ttf", ".csv", ".yml", ".yaml", ".txt",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
EXCLUDE_DIRS = {
    "pdf", "pdfs", "raw", "tmp_cvf", "session-backups", "__pycache__",
    "node_modules", ".git", "_files",
}
SKIP_PREFIX = ("_work_", "tmp_")

ARXIV_PDF = re.compile(r'(?:\.\./)*(?:pdfs?/|papers/pdf/)?(\d{4}\.\d{4,5})\.pdf')
SAVED_PAPER = re.compile(r'arxiv-html-papers|/static/browse/|name="citation_title"', re.I)

# identifiers that must never reach the published copy; the list itself lives in
# the gitignored sources.local.json so no company name is committed here
_SCRUB_WORDS = [w for w in (_load_sources().get("scrub") or []) if w]
SCRUB = [(re.compile(word, re.I), "") for word in _SCRUB_WORDS]

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


def _rel_size(path):
    try:
        return os.path.getsize(path) / 1048576.0
    except OSError:
        return 0.0


def _is_saved_paper_page(path):
    try:
        with io.open(path, "rb") as fh:
            head = fh.read(4096).decode("utf-8", "ignore")
    except OSError:
        return False
    return bool(SAVED_PAPER.search(head))


def _rewrite_pdf_links(text):
    return ARXIV_PDF.sub(lambda m: "https://arxiv.org/abs/" + m.group(1), text)


def _plan(src_dir, skip=()):
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in EXCLUDE_DIRS and d not in skip and not d.startswith("."))
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in INCLUDE_EXT or fn.startswith(SKIP_PREFIX):
                continue
            abs_p = os.path.join(dirpath, fn)
            if _rel_size(abs_p) > MAX_FILE_MB:
                continue
            if ext in (".html", ".htm") and _is_saved_paper_page(abs_p):
                continue
            yield abs_p, os.path.relpath(abs_p, src_dir)


def _to_webp(src, dst):
    from PIL import Image
    im = Image.open(src)
    if im.width > IMG_MAX_W:
        h = round(im.height * IMG_MAX_W / im.width)
        im = im.resize((IMG_MAX_W, h), Image.LANCZOS)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst, "WEBP", quality=IMG_QUALITY, method=4)


def _scrub(text):
    hits = 0
    for pat, repl in SCRUB:
        text, n = pat.subn(repl, text)
        hits += n
    return text, hits


def _entry_date(src_dir, files):
    """Use the newest source mtime — a real number, not an invented date."""
    newest = 0.0
    for abs_p, _rel in files[:400]:
        try:
            newest = max(newest, os.path.getmtime(abs_p))
        except OSError:
            pass
    if newest <= 0:
        try:
            newest = os.path.getmtime(src_dir)
        except OSError:
            return ""
    import datetime
    return datetime.date.fromtimestamp(newest).isoformat()


def build(report=False):
    made, skipped = [], []
    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT, ignore_errors=True)

    total_images, total_scrub = 0, 0
    for entry in DESIGN:
        root_key = entry.get("root", "design")
        base = SRC_ROOTS.get(root_key)
        if not base:
            skipped.append((entry["slug"], "no source root configured"))
            continue
        src = os.path.join(base, entry["src"].replace("/", os.sep))
        if not os.path.isdir(src):
            skipped.append((entry["slug"], "missing source"))
            continue
        files = list(_plan(src, tuple(entry.get("skip", ()))))
        if not files:
            skipped.append((entry["slug"], "nothing publishable"))
            continue

        entry["date"] = _entry_date(src, files)
        dst_root = os.path.join(OUT_ROOT, entry["slug"])
        webp = entry.get("images") == "webp"
        total, names, converted, scrubs = 0, [], 0, 0

        # images first, so the html pass can rewrite to .webp safely
        for abs_p, rel in files:
            ext = os.path.splitext(rel)[1].lower()
            dst = os.path.join(dst_root, rel)
            if webp and ext in IMAGE_EXT:
                try:
                    _to_webp(abs_p, os.path.splitext(dst)[0] + ".webp")
                    converted += 1
                    total += os.path.getsize(os.path.splitext(dst)[0] + ".webp")
                except Exception as exc:                  # noqa: BLE001
                    print("      image failed %s: %s" % (rel, exc))
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if ext in (".html", ".htm", ".md", ".json", ".txt", ".js", ".css", ".svg"):
                raw = io.open(abs_p, "rb").read()
                text = raw.decode("utf-8", "surrogateescape")
                if ".pdf" in text:
                    text = _rewrite_pdf_links(text)
                text, n = _scrub(text)
                scrubs += n
                if webp:
                    text = re.sub(r'(?<=["\'(])([^"\')]+)\.(?:png|jpg|jpeg)(?=["\')])',
                                  r'\1.webp', text, flags=re.I)
                raw = text.encode("utf-8", "surrogateescape")
                with io.open(dst, "wb") as fh:
                    fh.write(raw)
            else:
                shutil.copy2(abs_p, dst)
            total += os.path.getsize(dst)
            names.append(rel)

        total_images += converted
        total_scrub += scrubs

        if not os.path.exists(os.path.join(dst_root, "index.html")):
            links = [n for n in names if n.lower().endswith((".html", ".md"))]
            items = "\n".join(
                '    <li><a href="{h}">{t}</a></li>'.format(h=n.replace("\\", "/"), t=n.replace("\\", "/"))
                for n in links[:200])
            html = INDEX_TEMPLATE.format(title=entry["title"], lead=entry["summary"], items=items)
            p = os.path.join(dst_root, "index.html")
            os.makedirs(dst_root, exist_ok=True)
            with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(html)
            total += os.path.getsize(p)

        made.append({"entry": entry, "files": len(names), "mb": total / 1048576.0,
                     "converted": converted, "scrubbed": scrubs})

    return made, skipped, total_images, total_scrub


if __name__ == "__main__":
    made, skipped, imgs, scrubs = build()
    for m in made:
        print("%-24s %4d files %7.2f MB  webp=%d" % (m["entry"]["slug"], m["files"], m["mb"], m["converted"]))
    for s in skipped:
        print("SKIPPED %-20s %s" % (s[0], s[1]))
    print("total: %.2f MB | images converted: %d | scrub replacements: %d"
          % (sum(m["mb"] for m in made), imgs, scrubs))
