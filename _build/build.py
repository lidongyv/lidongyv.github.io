# -*- coding: utf-8 -*-
"""Static site generator for lidongyv.github.io.

Usage:  python _build/build.py
Output: files written into the repository root (served directly by GitHub Pages).
"""

import io
import os
import shutil
import sys
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gen import layout as L_          # noqa: E402
from gen import render as R           # noqa: E402
from gen.data_core import PROFILE, CAREER, EDUCATION  # noqa: E402
from gen.data_works import PAPERS, MANUSCRIPTS        # noqa: E402
from gen.data_repos import REPOS                      # noqa: E402
from gen.layout import url, t                         # noqa: E402

# private source locations live in a gitignored file so nothing company-specific
# or machine-specific is committed
try:
    import json as _json
    with io.open(os.path.join(HERE, "sources.local.json"), encoding="utf-8") as _fh:
        _LOCAL_SOURCES = _json.load(_fh)
except Exception:                                     # noqa: BLE001
    _LOCAL_SOURCES = {}

TODAY = datetime.date.today().isoformat()

# Some pre-existing repository files are not writable from the sandboxed process.
# For those the generator writes a staged copy into _build/out/ and reports it;
# `python _build/deploy.py` promotes them.
STAGE = os.path.join(HERE, "out")
DEFERRED = []


def w(rel, text):
    rel = rel.replace("\\", "/")
    path = os.path.join(ROOT, rel.replace("/", os.sep))
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return rel
    except PermissionError:
        staged = os.path.join(STAGE, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(staged) or ".", exist_ok=True)
        with io.open(staged, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        DEFERRED.append(rel)
        return "_build/out/" + rel


def copy(src, rel):
    rel = rel.replace("\\", "/")
    dst = os.path.join(ROOT, rel.replace("/", os.sep))
    try:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        return rel
    except PermissionError:
        staged = os.path.join(STAGE, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(staged) or ".", exist_ok=True)
        shutil.copy2(src, staged)
        DEFERRED.append(rel)
        return "_build/out/" + rel


# --------------------------------------------------------------------------
def assets():
    made = []
    for fn in ("site.css", "site.js", "favicon.svg"):
        src = os.path.join(HERE, "assets", fn)
        if os.path.exists(src):
            made.append(copy(src, "assets/site/" + fn))
    # project docs that live at the repository root
    for src_name, dest in (("README.md", "README.md"), ("gitignore.txt", ".gitignore")):
        src = os.path.join(HERE, src_name)
        if os.path.exists(src):
            made.append(copy(src, dest))
    # resume PDFs (source directory comes from the gitignored sources.local.json,
    # so no private path is committed)
    resume_dir = _LOCAL_SOURCES.get("resume", "")
    if not resume_dir:
        print("  ! no resume source configured; skipping CV PDFs")
    for fn in ("Lidong-Yu-Resume-EN.pdf", "Lidong-Yu-Resume-CN.pdf",
               "Lidong-Yu-Resume-Detailed-EN.pdf", "Lidong-Yu-Resume-Detailed-CN.pdf"):
        src = os.path.join(resume_dir, fn)
        if os.path.exists(src):
            made.append(copy(src, "resume/" + fn))
        else:
            print("  ! missing resume:", src)
    return made


def og_images():
    """Generate 1200x630 social cards with Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:                     # pragma: no cover
        print("  ! Pillow unavailable, skipping OG images:", exc)
        return []
    made = []
    os.makedirs(os.path.join(ROOT, "assets", "og"), exist_ok=True)

    def font(size, bold=True):
        cands = [
            r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        ]
        for c in cands:
            if os.path.exists(c):
                try:
                    return ImageFont.truetype(c, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def card(path, kicker, title, subtitle, accent=(36, 71, 216)):
        import numpy as np
        W, H = 1200, 630
        yy, xx = np.mgrid[0:H, 0:W]

        # base: very light paper white with a horizontal accent wash on the right
        base = np.ones((H, W, 3), dtype=np.float64) * np.array([252, 252, 254])
        wash = np.clip((xx - 700) / 520.0, 0, 1) ** 1.7
        wash = wash * 0.92
        col = np.array(accent, dtype=np.float64)
        base = base * (1 - wash[..., None]) + col * wash[..., None]

        # a second, softer glow bottom-right for depth
        d = np.sqrt((xx - 1120) ** 2 + (yy - 660) ** 2) / 520.0
        g = np.clip(1 - d, 0, 1) ** 2.2 * 0.45
        base = base * (1 - g[..., None]) + (col * 0.72) * g[..., None]

        # flatten toward white so the text side stays clean
        left_clean = np.clip((620 - xx) / 300.0, 0, 1) ** 1.2
        base = base * (1 - left_clean[..., None]) + 252.0 * left_clean[..., None]

        img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"), "RGB")
        d2 = ImageDraw.Draw(img)

        # accent rule + kicker
        d2.rectangle([72, 92, 128, 97], fill=accent)
        d2.text((144, 78), kicker, font=font(23, False), fill=(110, 118, 132))

        y = 172
        for line in title.split("\n"):
            d2.text((72, y), line, font=font(62), fill=(12, 14, 20))
            y += 78
        y += 22
        for line in subtitle.split("\n"):
            d2.text((72, y), line, font=font(28, False), fill=(88, 97, 113))
            y += 44

        # footer
        d2.line([72, H - 118, W - 72, H - 118], fill=(224, 228, 238), width=1)
        d2.text((72, H - 92), "lidongyv.github.io", font=font(25, False), fill=(120, 128, 144))
        badge_x, badge_y, badge_w, badge_h = W - 72 - 116, H - 104, 116, 52
        d2.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                             radius=14, fill=accent)
        d2.text((badge_x + 30, badge_y + 11), "LY", font=font(26), fill=(255, 255, 255))

        img.save(os.path.join(ROOT, path.lstrip("/")), "PNG", optimize=True)
        made.append(path)

    card("/assets/og/og-default.png", "Lidong Yu · 于立冬",
         "VLA & World Action Models\nfor production vehicles",
         "Head of AD Algorithms · 8 peer-reviewed papers\nNeurIPS / ICCV / CVPR / AAAI")
    card("/assets/og/og-publications.png", "PUBLICATIONS",
         "8 peer-reviewed papers",
         "ICCV 2025 Oral · AAAI 2018 spotlight (first author)\nStereo matching · Depth · 3D lane detection")
    card("/assets/og/og-projects.png", "PROJECTS",
         "From stereo matching\nto one-stage VLA",
         "Perception · tracking · imitation learning\nEnd-to-end driving · world models")
    card("/assets/og/og-opensource.png", "OPEN SOURCE",
         "39 self-created repos",
         "Research · tools · knowledge base\nAll public, all mine — forks excluded", accent=(15, 118, 110))
    card("/assets/og/og-blog.png", "BLOG",
         "Engineering notes",
         "VLA · world models · data closed loops\nWritten from production experience", accent=(123, 63, 242))
    return made


# --------------------------------------------------------------------------
def feed(lang, posts):
    items = []
    for p in posts:
        link = L_.abs_url(lang, "blog/%s.html" % p["slug"])
        title = p["title"] if lang == "zh" else p["title_en"]
        summary = p["summary"] if lang == "zh" else p["summary_en"]
        body = p["html"]
        items.append("""
    <item>
      <title>{title}</title>
      <link>{link}</link>
      <guid isPermaLink="true">{link}</guid>
      <pubDate>{pub}</pubDate>
      <description>{desc}</description>
      <content:encoded><![CDATA[{body}]]></content:encoded>
    </item>""".format(
            title=x(title), link=x(link), pub=pubdate(p["date"]),
            desc=x(summary), body=body))

    channel_title = "Lidong Yu · " + ("博客与研究笔记" if lang == "zh" else "Blog & research notes")
    return """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{title}</title>
    <link>{home}</link>
    <description>{desc}</description>
    <language>{lang}</language>
    <atom:link href="{self}" rel="self" type="application/rss+xml"/>
    <lastBuildDate>{now}</lastBuildDate>
{items}
  </channel>
</rss>
""".format(
        title=x(channel_title),
        home=L_.abs_url(lang, "index.html"),
        desc=x(t({"zh": "VLA、世界模型、数据闭环与感知工程落地的研究笔记。",
                  "en": "Research notes on VLA, world models, data loops and perception engineering."}, lang)),
        lang="zh-CN" if lang == "zh" else "en",
        self=L_.abs_url(lang, "blog/feed.xml"),
        now=pubdate(TODAY),
        items="".join(items),
    )


def x(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def pubdate(iso):
    try:
        d = datetime.date.fromisoformat(iso)
        return d.strftime("%a, %d %b %Y 09:00:00 +0800")
    except Exception:
        return datetime.datetime.now().strftime("%a, %d %b %Y 09:00:00 +0800")


def sitemap(pages):
    rows = []
    for lang, path, prio in pages:
        rows.append("""  <url>
    <loc>{loc}</loc>
    <lastmod>{mod}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>{p}</priority>
  </url>""".format(loc=x(L_.abs_url(lang, path)), mod=TODAY, p=prio))
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{rows}
</urlset>
""".format(rows="\n".join(rows))


def robots():
    return """User-agent: *
Allow: /
Disallow: /_build/
Disallow: /_includes/
Disallow: /_layouts/
Disallow: /README.md

Sitemap: https://lidongyv.github.io/sitemap.xml
"""


# --------------------------------------------------------------------------
def main():
    made = []
    print("[1/6] assets")
    made += assets()

    print("[2/6] og images")
    made += og_images()

    print("[2/6] design subsites")
    import design
    d_made, d_skipped, d_imgs, d_scrub = design.build()
    for m in d_made:
        print("      %-22s %4d files %7.2f MB  webp=%d  scrubbed=%d"
              % (m["entry"]["slug"], m["files"], m["mb"], m["converted"], m["scrubbed"]))
    print("      images re-encoded: %d | company-name replacements: %d" % (d_imgs, d_scrub))
    for s in d_skipped:
        print("      SKIPPED %-18s %s" % (s[0], s[1]))

    print("[3/6] pages")
    posts = R.load_posts(os.path.join(HERE, "posts"))
    print("      posts:", len(posts))

    for lang in ("zh", "en"):
        home_file = "index.html" if lang == "zh" else "en/index.html"
        made.append(w(home_file, R.render_home(lang, posts)))
        made.append(w(url(lang, "design/index.html").lstrip("/"), R.render_design_index(lang)))
        made.append(w(url(lang, "experience.html").lstrip("/"), R.render_experience(lang)))
        made.append(w(url(lang, "publications.html").lstrip("/"), R.render_publications(lang)))
        made.append(w(url(lang, "projects.html").lstrip("/"), R.render_projects(lang)))
        made.append(w(url(lang, "project9.html").lstrip("/"), R.render_foggy(lang)))
        made.append(w(url(lang, "opensource.html").lstrip("/"), R.render_opensource(lang)))
        made.append(w(url(lang, "blog/index.html").lstrip("/"), R.render_blog_index(lang, posts)))
        made.append(w(url(lang, "about.html").lstrip("/"), R.render_about(lang)))
        # Post bodies are written in Chinese only; /en/blog/ links straight to them
        # instead of duplicating the same content under a second URL.
        if lang == "zh":
            for p in posts:
                made.append(w("blog/%s.html" % p["slug"], R.render_post(lang, p, posts)))
        made.append(w(url(lang, "blog/feed.xml").lstrip("/"), feed(lang, posts)))

    made.append(w("404.html", R.render_404("zh")))
    made.append(w("en/404.html", R.render_404("en")))

    print("[4/6] legacy project pages")
    import legacy
    made += legacy.build(w)

    print("[5/6] sitemap / robots / .nojekyll")
    pages = []
    for lang in ("zh", "en"):
        pages.append((lang, "index.html", "1.0"))
        pages.append((lang, "experience.html", "0.8"))
        pages.append((lang, "publications.html", "0.9"))
        pages.append((lang, "projects.html", "0.9"))
        pages.append((lang, "opensource.html", "0.8"))
        pages.append((lang, "about.html", "0.8"))
        pages.append((lang, "blog/index.html", "0.7"))
        pages.append((lang, "design/index.html", "0.6"))
        if lang == "zh":
            for p in posts:
                pages.append((lang, "blog/%s.html" % p["slug"], "0.6"))
    for d in R.DESIGN:
        pages.append(("zh", "design/%s/index.html" % d["slug"], "0.5"))
    made.append(w("sitemap.xml", sitemap(pages)))
    made.append(w("robots.txt", robots()))
    made.append(w(".nojekyll", ""))

    print("[6/6] done — %d files written" % len(made))
    os.makedirs(os.path.join(HERE, "log"), exist_ok=True)
    with io.open(os.path.join(HERE, "log", "built.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(made))
    if DEFERRED:
        print("\n  staged for deploy (%d files could not be written in place):" % len(DEFERRED))
        for f in DEFERRED:
            print("    - " + f)
        print("\n  run:  python _build/deploy.py")


if __name__ == "__main__":
    main()
