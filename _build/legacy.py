# -*- coding: utf-8 -*-
"""Convert the 8 legacy Jekyll project layouts into standalone static pages.

The old pages were `projectN.html` with `layout: projectN`. They are linked from
the resume PDFs (https://lidongyv.github.io/projectN.html), so their URLs must keep
working even after Jekyll is switched off (.nojekyll).

Rendered output keeps the original markup and the original stylesheet
(/assets/css/main.css) so nothing visually regresses, and adds a slim top bar that
leads back to the new site.
"""

import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LIQUID_ASSET = re.compile(r'\{\{\s*"([^"]+)"\s*\|\s*prepend:\s*site\.baseurl\s*\}\}')

# Dead references that already existed on the old site; corrected here so the
# converted pages do not ship 404s.
FIXUPS = [
    # project4 / project3: misspelled archived manuscript filenames
    ('href="/assets/pdf/nips-cvprpdf"', 'href="/assets/pdf/nips-cvpr19.pdf"'),
    ('href="/assets/pdf/interpreatble%20tpami.pdf"', 'href="/assets/pdf/interpretable%20tpami.pdf"'),
    # project5: image 14.png was never committed (only 0-13 exist)
    ('<img alt="report" src="/assets/img/project5/new/14.png" />', ''),
]

TITLES = {
    1: ("Deep Stereo Matching with Explicit Cost Aggregation Sub-Architecture", "已于 AAAI 2018 发表（第一作者，spotlight）",
        "Published at AAAI 2018 (first author, spotlight)"),
    2: ("Explicit Context Mapping for Stereo Matching", "未发表投稿稿存档（CVPR'19 / ICCV'19）",
        "Archived submission drafts (CVPR'19 / ICCV'19), not published"),
    3: ("Interpretable Stereo Matching with Region Support", "未发表投稿稿存档（CVPR'18 / ECCV'19 + TPAMI 手稿）",
        "Archived submission drafts (CVPR'18 / ECCV'19 + TPAMI manuscript), not published"),
    4: ("Region-Support Depth Inference from a Single Image", "未发表投稿稿存档（NeurIPS / CVPR'19）",
        "Archived submission drafts (NeurIPS / CVPR'19), not published"),
    5: ("Imitation Learning for Relative Navigation", "工程报告 · 驭势科技 2019—2020",
        "Engineering report · UISEE 2019–2020"),
    6: ("Single Object Tracking by Feature Disentanglement", "工程报告 · 驭势科技 2020",
        "Engineering report · UISEE 2020"),
    7: ("Multiple Object Detection with Deformation Prediction", "工程报告 · 驭势科技 2019—2020",
        "Engineering report · UISEE 2019–2020"),
    8: ("Video Object Detection by Temporal Aggregation", "工程报告 · 驭势科技 2020",
        "Engineering report · UISEE 2020"),
}

SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Lidong Yu</title>
<meta name="description" content="{title} — {note_en}">
<meta name="robots" content="noindex, follow">
<link rel="icon" href="/assets/site/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/assets/css/main.css">
<style>
  .legacy-bar {{
    position: sticky; top: 0; z-index: 50;
    display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
    padding: 11px 20px;
    background: rgba(255,255,255,.92);
    border-bottom: 1px solid #e4e7ee;
    backdrop-filter: saturate(160%) blur(12px);
    -webkit-backdrop-filter: saturate(160%) blur(12px);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
  }}
  .legacy-bar a {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13.5px; font-weight: 600; text-decoration: none;
    color: #2447d8; border: 1px solid #c8d4ff; background: #eef2ff;
    padding: 6px 12px; border-radius: 999px;
  }}
  .legacy-bar a:hover {{ background: #dfe6ff; }}
  .legacy-bar .legacy-note {{
    font-size: 12.5px; color: #5c6575; margin-left: auto;
  }}
  .legacy-badge {{
    font-size: 12px; font-weight: 600; color: #5c6575;
    border: 1px solid #e4e7ee; background: #f6f7f9;
    padding: 5px 10px; border-radius: 999px;
  }}
  body {{ background: #fff; }}
  .project-div {{ max-width: 1080px; margin: 0 auto; padding: 12px 20px 60px; }}
  .project-div img {{ max-width: 100%; height: auto; display: block; margin: 18px auto;
                      border: 1px solid #e4e7ee; border-radius: 12px; }}
  .project-div h1 {{ font-size: 1.7rem; line-height: 1.25; margin: 26px 0 6px; }}
  .project-div h2, .project-div h3 {{ margin-top: 28px; }}
  .project-div p {{ color: #2b3140; }}
  .project-navigator {{ display: none; }}
</style>
</head>
<body>
  <div class="legacy-bar">
    <a href="/">← 返回新主页 / Home</a>
    <a href="{blog}">论文与项目</a>
    <span class="legacy-badge">{note}</span>
    <span class="legacy-note">旧版项目页 · 内容原样保留</span>
  </div>
{body}
</body>
</html>
"""


def convert(n):
    src = os.path.join(ROOT, "_layouts", "project%d.html" % n)
    if not os.path.isfile(src):
        return None
    with io.open(src, encoding="utf-8") as fh:
        raw = fh.read()

    # body contents only
    m = re.search(r"<body[^>]*>(.*)</body>", raw, re.S | re.I)
    body = m.group(1) if m else raw
    body = re.sub(r"\{%\s*include\s+head\.html\s*%\}", "", body)
    body = re.sub(r"\{%\s*include\s*footer\.html\s*%\}", "", body)
    body = re.sub(r"\{\{\s*content\s*\}\}", "", body)
    body = LIQUID_ASSET.sub(lambda mm: mm.group(1), body)
    body = re.sub(r'\{\{\s*"[^"]*"\s*\|\s*prepend:\s*site\.baseurl\s*\}\}', "", body)
    body = body.replace('href="index.html"', 'href="/"')
    body = body.replace('href="project', 'href="/project')
    # URL-encode spaces that the old markup left raw inside href/src
    body = re.sub(r'(href|src)="(/assets/[^"]*)"',
                  lambda mm: '%s="%s"' % (mm.group(1), mm.group(2).replace(" ", "%20")), body)
    for old, new in FIXUPS:
        body = body.replace(old, new)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    title, note_zh, note_en = TITLES[n]
    return SHELL.format(
        title=title,
        note_en=note_en,
        note=note_zh,
        blog="/publications.html",
        body=body,
    )


def build(writer):
    """writer(relpath, text) -> whatever the build helper returns."""
    made = []
    for n in range(1, 9):
        html = convert(n)
        if html is None:
            continue
        made.append(writer("project%d.html" % n, html))
    return made
