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
    # resume PDFs
    resume_dir = r"C:\Users\Admin\Desktop\新石器\最新简历"
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
        W, H = 1200, 630
        img = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(img)
        # soft gradient blobs
        for i in range(300):
            a = int(16 * (1 - i / 300))
            d.ellipse([-260 + i, -320 + i, 760 - i, 520 - i], outline=None,
                      fill=(255 - (255 - accent[0]) * a // 16,
                            255 - (255 - accent[1]) * a // 16,
                            255 - (255 - accent[2]) * a // 16))
        d.rectangle([0, 0, W, 8], fill=accent)
        d.text((72, 78), kicker, font=font(24, False), fill=(110, 118, 132))
        y = 132
        for line in title.split("\n"):
            d.text((72, y), line, font=font(64), fill=(12, 14, 20))
            y += 84
        y += 16
        for line in subtitle.split("\n"):
            d.text((72, y), line, font=font(30, False), fill=(92, 101, 117))
            y += 46
        d.text((72, H - 96), "lidongyv.github.io", font=font(26, False), fill=(120, 128, 144))
        d.text((W - 250, H - 96), "LY", font=font(34), fill=accent)
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

Sitemap: https://lidongyv.github.io/sitemap.xml
"""


# --------------------------------------------------------------------------
def main():
    made = []
    print("[1/5] assets")
    made += assets()

    print("[2/5] og images")
    made += og_images()

    print("[3/5] pages")
    posts = R.load_posts(os.path.join(HERE, "posts"))
    print("      posts:", len(posts))

    for lang in ("zh", "en"):
        home_file = "index.html" if lang == "zh" else "en/index.html"
        made.append(w(home_file, R.render_home(lang, posts)))
        made.append(w(url(lang, "experience.html").lstrip("/"), R.render_experience(lang)))
        made.append(w(url(lang, "publications.html").lstrip("/"), R.render_publications(lang)))
        made.append(w(url(lang, "projects.html").lstrip("/"), R.render_projects(lang)))
        made.append(w(url(lang, "opensource.html").lstrip("/"), R.render_opensource(lang)))
        made.append(w(url(lang, "blog/index.html").lstrip("/"), R.render_blog_index(lang, posts)))
        made.append(w(url(lang, "about.html").lstrip("/"), R.render_about(lang)))
        for p in posts:
            made.append(w(url(lang, "blog/%s.html" % p["slug"]).lstrip("/"),
                          R.render_post(lang, p, posts)))
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
        for p in posts:
            pages.append((lang, "blog/%s.html" % p["slug"], "0.6"))
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
