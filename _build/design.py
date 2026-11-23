# -*- coding: utf-8 -*-
"""Publish the study/report sites that live outside this repository.

Everything is written under ``/blog/``:

* a project becomes ``blog/<slug>/index.html`` (plus its assets)
* pages inside a big project can be listed as their own blog entries via ``href``

Source roots come from ``_build/sources.local.json`` (gitignored), so no private
path is committed. Entries may opt into:

``root``     which source root to read (default ``design``)
``images``   ``webp`` — re-encode screenshots and rewrite the html references
``skip``     extra directory names to ignore anywhere in the tree
``exclude``  extra file names (exact) to ignore
``unlock``   strip a password gate from the entry page
``href``     page entry: link into an already published site instead of copying
``src_file`` page entry: the source file used for the entry date

Never published, from any root: PDFs, scraped conference pages, saved arXiv paper
pages, build caches, videos, files over ``MAX_FILE_MB``, and anything matching an
entry's ``skip``/``exclude``.
"""

import io
import os
import re
import shutil

from gen.data_design import DESIGN  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_ROOT = os.path.join(ROOT, "blog")

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
    "node_modules", ".git", "_files", "_build",
}
SKIP_PREFIX = ("_work_", "tmp_")

ARXIV_PDF = re.compile(r'(?:\.\./)*(?:pdfs?/|papers/pdf/)?(\d{4}\.\d{4,5})\.pdf')
SAVED_PAPER = re.compile(r'arxiv-html-papers|/static/browse/|name="citation_title"', re.I)


def _load_sources():
    import json
    try:
        with io.open(os.path.join(HERE, "sources.local.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                  # noqa: BLE001
        return {}


SRC_ROOTS = _load_sources()
_SCRUB_WORDS = [w for w in (SRC_ROOTS.get("scrub") or []) if w]
SCRUB = [(re.compile(w, re.I), "") for w in _SCRUB_WORDS]

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


def _earliest_date(paths):
    import datetime
    earliest = None
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        for ts in (st.st_ctime, st.st_mtime):
            if ts and (earliest is None or ts < earliest):
                earliest = ts
    if earliest is None:
        return ""
    return datetime.date.fromtimestamp(earliest).isoformat()


def _plan(src_dir, skip=(), exclude=()):
    excl = set(exclude)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in EXCLUDE_DIRS and d not in skip and not d.startswith("."))
        for fn in sorted(filenames):
            if fn in excl or fn.startswith(SKIP_PREFIX):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in INCLUDE_EXT:
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
        im = im.resize((IMG_MAX_W, round(im.height * IMG_MAX_W / im.width)), Image.LANCZOS)
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


def _drop_div(html, marker):
    """Remove the balanced <div ...> block whose opening tag contains marker."""
    i = html.find(marker)
    if i < 0:
        return html, 0
    start = html.rfind("<div", 0, i)
    if start < 0:
        return html, 0
    depth, j = 0, start
    while j < len(html):
        nxt_open = html.find("<div", j)
        nxt_close = html.find("</div>", j)
        if nxt_close < 0:
            return html, 0
        if 0 <= nxt_open < nxt_close:
            depth += 1
            j = nxt_open + 4
        else:
            depth -= 1
            j = nxt_close + 6
            if depth == 0:
                return html[:start] + html[j:], 1
    return html, 0


PAYLOAD_RE = re.compile(r'<script[^>]*id="wm-payload"[^>]*>.*?</script>', re.S | re.I)
MAIN_RE = re.compile(r'(<main[^>]*id="wm-content"[^>]*>)', re.I)


def _decrypt_via_node(src_path):
    """Run the WebCrypto decryptor; the password never leaves the local config."""
    import subprocess
    password = SRC_ROOTS.get("world_model_password")
    node = SRC_ROOTS.get("node") or "node"
    script = os.path.join(HERE, "tools", "decrypt_wm.mjs")
    if not password or not os.path.isfile(script):
        return None
    out = os.path.join(HERE, ".wm_plain.tmp")
    try:
        r = subprocess.run([node, script, src_path, password, out],
                           capture_output=True, timeout=120)
    except Exception:                                  # noqa: BLE001
        return None
    if r.returncode != 0 or not os.path.isfile(out):
        return None
    data = io.open(out, "rb").read()
    os.remove(out)
    return data.decode("utf-8", "replace")


def _unlock(html, src_path=None):
    """Remove the password gate.

    The entry page keeps its content AES-GCM encrypted in a JSON payload and the
    gate script is also the renderer, so simply deleting the gate blanks the page.
    Instead the payload is decrypted at build time and injected into <main>.
    """
    plain = ""
    encrypted = bool(src_path and PAYLOAD_RE.search(html))
    if encrypted:
        decrypted = _decrypt_via_node(src_path)
        if decrypted is None:
            # encrypted page we cannot open: refuse rather than publish a blank shell
            return None
        plain = decrypted

    html = PAYLOAD_RE.sub("", html)
    html, _ = _drop_div(html, 'class="wm-lock"')
    # drop the gate script (it binds #lock-form); keep the UX script
    out, pos = [], 0
    for m in re.finditer(r'<script\b[^>]*>(.*?)</script>', html, re.S | re.I):
        if "lock-form" in m.group(1) or "wm-payload" in m.group(1):
            out.append(html[pos:m.start()])
            pos = m.end()
    out.append(html[pos:])
    html = "".join(out)

    html = re.sub(r'(<nav[^>]*id="wm-nav"[^>]*?)\s+hidden', r'\1', html)
    html = re.sub(r'(<main[^>]*id="wm-content"[^>]*?)\s+hidden', r'\1', html)
    if plain:
        mm = MAIN_RE.search(html)
        if mm:
            html = html[:mm.start()] + mm.group(1) + plain + html[mm.end():]
    return html


def build(report=False):
    made, skipped = [], []
    if os.path.isdir(OUT_ROOT):
        # keep the three markdown posts, drop every generated sub-site
        for name in os.listdir(OUT_ROOT):
            p = os.path.join(OUT_ROOT, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)

    total_images, total_scrub = 0, 0
    for entry in DESIGN:
        if entry.get("href"):
            # page entry: an individual page of an already published project
            base = SRC_ROOTS.get(entry.get("root", "design"))
            src_file = os.path.join(base or "", entry.get("src_file", "").replace("/", os.sep))
            entry["date"] = _earliest_date([src_file]) if os.path.isfile(src_file) else ""
            if not entry["date"]:
                skipped.append((entry["slug"], "page entry without a source file"))
            continue

        root_key = entry.get("root", "design")
        base = SRC_ROOTS.get(root_key)
        if not base:
            skipped.append((entry["slug"], "no source root configured"))
            continue
        src = os.path.join(base, entry["src"].replace("/", os.sep))
        if not os.path.isdir(src):
            skipped.append((entry["slug"], "missing source"))
            continue

        files = list(_plan(src, tuple(entry.get("skip", ())), tuple(entry.get("exclude", ()))))
        if not files:
            skipped.append((entry["slug"], "nothing publishable"))
            continue

        entry["date"] = _earliest_date([p for p, _ in files])
        dst_root = os.path.join(OUT_ROOT, entry["slug"])
        webp = entry.get("images") == "webp"
        unlock = bool(entry.get("unlock"))
        total, names, converted, scrubs = 0, [], 0, 0

        for abs_p, rel in files:
            ext = os.path.splitext(rel)[1].lower()
            dst = os.path.join(dst_root, rel)
            if webp and ext in IMAGE_EXT:
                try:
                    _to_webp(abs_p, os.path.splitext(dst)[0] + ".webp")
                    converted += 1
                    total += os.path.getsize(os.path.splitext(dst)[0] + ".webp")
                except Exception as exc:                       # noqa: BLE001
                    print("      image failed %s: %s" % (rel, exc))
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if ext in (".html", ".htm", ".md", ".json", ".txt", ".js", ".css", ".svg"):
                raw = io.open(abs_p, "rb").read()
                text = raw.decode("utf-8", "surrogateescape")
                if ".pdf" in text:
                    text = ARXIV_PDF.sub(lambda m: "https://arxiv.org/abs/" + m.group(1), text)
                if unlock and ext in (".html", ".htm"):
                    unlocked = _unlock(text, abs_p)
                    if unlocked is None:
                        print("      UNLOCK FAILED %s (encrypted page left out)" % rel)
                        continue
                    text = unlocked
                text, n = _scrub(text)
                scrubs += n
                if webp:
                    text = re.sub(r'(?<=["\'(])([^"\')]+)\.(?:png|jpg|jpeg)(?=["\')])',
                                  r"\1.webp", text, flags=re.I)
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
            items = "\n".join('    <li><a href="{h}">{t}</a></li>'.format(
                h=n.replace("\\", "/"), t=n.replace("\\", "/")) for n in links[:200])
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
        print("%-26s %4d files %7.2f MB  webp=%d  scrub=%d"
              % (m["entry"]["slug"], m["files"], m["mb"], m["converted"], m["scrubbed"]))
    for s in skipped:
        print("SKIPPED %-22s %s" % (s[0], s[1]))
    print("total: %.2f MB | images converted: %d | scrub replacements: %d"
          % (sum(m["mb"] for m in made), imgs, scrubs))