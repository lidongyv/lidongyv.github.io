# -*- coding: utf-8 -*-
"""Page builders."""

import html
import os

from . import layout as L_
from . import releases
from .blog import parse_front_matter, render_markdown, plain, word_count
from .data_core import PROFILE, STATS, FOCUS, CAREER, EDUCATION
from .data_works import PAPERS, MANUSCRIPTS, PATENTS, PROJECT_PAGES, PROJECTS_INTRO
from .data_repos import REPOS, REPO_GROUPS, FINANCE, FORKS
from .layout import t, esc, url, icon, icon_solid, ui, page

GITHUB_REPO = "https://github.com/lidongyv/"


# --------------------------------------------------------------------------
# Small components
# --------------------------------------------------------------------------
def section_head(lang, title, level=2):
    """A plain section label. No kicker, no commentary — the content speaks."""
    return ('<div class="sec-head" data-reveal><div class="sec-head__text"><h%d>%s</h%d></div></div>'
            % (level, esc(title), level))


def tag(text, cls=""):
    return '<span class="tag%s">%s</span>' % ((" " + cls) if cls else "", esc(text))


def link_chip(href, label, icon_name="link", external=False):
    ext = ' target="_blank" rel="noopener"' if external else ""
    return ('<a class="link-chip" href="%s"%s>%s<span>%s</span>%s</a>'
            % (esc(href), ext, icon(icon_name), esc(label),
               icon("external") if external else ""))


def btn(href, label, cls="btn", icon_name="arrow", external=False, attr=""):
    ext = ' target="_blank" rel="noopener"' if external else ""
    return ('<a class="%s" href="%s"%s%s>%s<span>%s</span></a>'
            % (cls, esc(href), ext, attr, icon(icon_name) if icon_name else "", esc(label)))


def badge_for(lang, paper):
    out = []
    if "oral" in paper["badges"]:
        out.append(tag("Oral", "tag--oral"))
    if "first" in paper["badges"]:
        out.append(tag("第一作者" if lang == "zh" else "First author", "tag--first"))
    if "spotlight" in paper["badges"]:
        out.append(tag("Spotlight", "tag--first"))
    if "preprint" in paper["badges"]:
        out.append(tag("Preprint", "tag--mono"))
    if "technical report" in paper["badges"]:
        out.append(tag("Tech report", "tag--mono"))
    return "".join(out)


def authors_html(lang, authors):
    parts = []
    for name, is_me in authors:
        if is_me:
            parts.append('<span class="me">%s</span>' % esc(name))
        else:
            parts.append(esc(name))
    return ", ".join(parts)


def paper_card(lang, p, compact=False):
    venue = "%s %s" % (p["venue"], p["year"])
    links = [link_chip(p["paper"], p.get("paper_label") or "Paper", "doc", external=True)]
    if p.get("alt_paper"):
        links.append(link_chip(p["alt_paper"], p.get("alt_paper_label") or "Page", "link", external=True))
    if p.get("code"):
        links.append(link_chip(p["code"], "Code", "github", external=True))
    if p.get("project"):
        proj = p["project"]
        external = proj.startswith("http")
        links.append(link_chip(proj, t({"zh": "项目页", "en": "Project"}, lang), "link", external=external))

    body = ['<article class="card paper" data-reveal>', '<div class="card__top">']
    body.append('<div class="links" style="gap:6px">%s%s</div>' % (tag(venue, "tag--venue"), badge_for(lang, p)))
    body.append("</div>")
    body.append('<h3 class="paper__title">%s</h3>' % esc(p["title"]))
    body.append('<p class="paper__authors">%s</p>' % authors_html(lang, p["authors"]))
    if not compact:
        body.append('<dl class="paper__grid">')
        for label_zh, label_en, key in (("问题", "Problem", "problem"), ("方法", "Method", "method")):
            body.append(
                '<div class="paper__field"><dt>%s</dt><dd>%s</dd></div>'
                % (esc(label_zh if lang == "zh" else label_en), t(p[key], lang)))
        body.append("</dl>")
        body.append('<div class="paper__result">%s</div>' % t(p["result"], lang))
        body.append('<p class="small muted" style="margin-top:12px"><strong>%s</strong> %s</p>'
                    % (esc("为什么值得关注" if lang == "zh" else "Why it matters"), t(p["why"], lang)))
    body.append('<div class="card__foot"><div class="links">%s</div></div>' % "".join(links))
    body.append("</article>")
    return "".join(body)


def stat_strip(lang):
    cells = "".join(
        '<div class="stat"><b>%s</b><span>%s</span></div>' % (esc(s["value"]), esc(t(s["label"], lang)))
        for s in STATS
    )
    return '<div class="stats" data-reveal>%s</div>' % cells


def focus_cards(lang):
    out = []
    for i, f in enumerate(FOCUS):
        out.append(
            '<article class="card card--hover" data-reveal style="transition-delay:%dms">'
            '<div class="sec-kicker" style="margin-bottom:8px">%s</div>'
            "<h3>%s</h3><p style=\"margin-top:10px\">%s</p>"
            '<div class="card__tags">%s</div></article>'
            % (i * 60,
               esc(t(f["kicker"], lang)),
               esc(t(f["title"], lang)),
               t(f["body"], lang),
               "".join(tag(t(x, lang)) for x in f["tags"]))
        )
    return '<div class="grid grid--3">%s</div>' % "".join(out)


def career_timeline(lang, limit=None, show_bullets=True):
    rows = []
    items = CAREER if limit is None else CAREER[:limit]
    for c in items:
        bullets = ""
        if show_bullets:
            bullets = "<ul>%s</ul>" % "".join("<li>%s</li>" % t(b, lang) for b in c["bullets"])
        tags = "".join(tag(x, "tag--mono") for x in c["tags"])
        rows.append("""
      <article class="tl-item" data-reveal>
        <div class="tl-when"><b>{when}</b></div>
        <div class="tl-body">
          <h3><span class="org">{company}</span></h3>
          <div class="tl-role">{role}</div>
          <div class="tl-scope">{scope}</div>
          <p class="small" style="margin-top:10px">{summary}</p>
          {bullets}
          <div class="card__tags">{tags}</div>
        </div>
      </article>""".format(
            when=esc(t(c["when"], lang) if isinstance(c["when"], dict) else c["when"]),
            company=esc(t(c["company"], lang)),
            role=esc(t(c["role"], lang)),
            scope=esc(t(c["scope"], lang)),
            summary=t(c["summary"], lang),
            bullets=bullets,
            tags=tags,
        ))
    return '<div class="timeline">%s</div>' % "".join(rows)


def repo_card(lang, r):
    desc = t(r["desc"], lang)
    lang_tag = ('<span class="repo__lang">%s</span>' % esc(r["lang"])) if r.get("lang") not in (None, "—") else ""
    star = ""
    return """
      <div class="repo" data-kind="{kind}" data-reveal>
        <div class="repo__name">{icon}<a href="{href}" target="_blank" rel="noopener">{name}</a>{lang_tag}</div>
        <div class="repo__desc">{desc}</div>
        <div class="repo__meta"><span>{pushed}</span></div>
      </div>""".format(
        kind=r["kind"],
        icon=icon_solid("github"),
        href=esc(GITHUB_REPO + r["name"]),
        name=esc(r["name"]),
        lang_tag=lang_tag,
        desc=esc(desc),
        pushed=esc(r["pushed"]),
    )


def project_card(lang, p):
    return """
      <article class="card card--hover" data-reveal>
        <div class="card__top">
          <div class="links">{kind}{year}</div>
        </div>
        <h3><a href="{href}">{title}</a></h3>
        <p style="margin-top:10px">{desc}</p>
        <div class="card__tags">{tags}</div>
        <div class="card__foot">
          <div class="links"><a class="link-chip" href="{href}">{arrow}<span>{more}</span></a></div>
        </div>
      </article>""".format(
        kind=tag(t(p["kind"], lang), p["kind_cls"]),
        year=tag(p["year"], "tag--mono"),
        href=esc(p["href"]),
        title=esc(t(p["title"], lang)),
        desc=t(p["desc"], lang),
        tags="".join(tag(x) for x in p["tags"]),
        arrow=icon("arrow"),
        more=esc(ui("read_more", lang)),
    )


# --------------------------------------------------------------------------
# Blog data
# --------------------------------------------------------------------------
def load_posts(posts_dir):
    posts = []
    if not os.path.isdir(posts_dir):
        return posts
    for fn in sorted(os.listdir(posts_dir), reverse=True):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(posts_dir, fn)
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        fm, body = parse_front_matter(raw)
        slug = fm.get("slug") or fn.rsplit(".", 1)[0]
        posts.append({
            "slug": slug,
            "file": fn,
            "title": fm.get("title", slug),
            "title_en": fm.get("title_en", fm.get("title", slug)),
            "date": fm.get("date", ""),
            "tags": [x.strip() for x in (fm.get("tags") or "").split(",") if x.strip()],
            "tags_en": [x.strip() for x in (fm.get("tags_en") or "").split(",") if x.strip()],
            "summary": fm.get("summary", ""),
            "summary_en": fm.get("summary_en", fm.get("summary", "")),
            "body_md": body,
            "html": render_markdown(body),
            "minutes": word_count(body),
            "plain": plain(body),
        })
    posts.sort(key=lambda p: p["date"], reverse=True)
    return posts


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def render_home(lang, posts):
    featured_papers = PAPERS[:3] + [p for p in PAPERS if "first" in p["badges"]]
    seen = set()
    fp = []
    for p in featured_papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            fp.append(p)

    featured_repos = [r for r in REPOS if r.get("featured")][:8]

    body = """
<section class="hero">
  <div class="wrap">
    <div class="hero-grid">
      <div>
        <div class="eyebrow"><span class="dot"></span>{eyebrow}</div>
        <h1>{headline}</h1>
        <p class="hero-lede">{lede}</p>
        <div class="status-row">
          <span class="pill pill--live"><span class="dot"></span>{status}</span>
          <span class="pill">{location}</span>
          <span class="pill pill--accent">{role}</span>
        </div>
        <div class="cta-row">
          {cta_primary}
          {cta_github}
          <button class="btn btn--ghost" type="button" data-copy="{email}">{copy_icon}<span>{copy_label}</span></button>
        </div>
        {stats}
      </div>
      <div class="hero-side">
        <div class="portrait"><img src="/assets/img/head.webp" alt="{name}" width="520" height="520" loading="eager"></div>
        <div class="mini-list">
          <dl class="mini-card"><dt>{now_label}</dt><dd>{now_value}</dd></dl>
          <dl class="mini-card"><dt>{focus_label}</dt><dd>{focus_value}</dd></dl>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section" id="focus">
  <div class="wrap">
    {focus_head}
    {focus}
  </div>
</section>

<section class="section" id="experience">
  <div class="wrap">
    {career_head}
    {career}
    <div class="cta-row" style="margin-top:28px">{career_cta}</div>
  </div>
</section>

<section class="section" id="papers">
  <div class="wrap">
    {paper_head}
    <div class="grid grid--2">{papers}</div>
    <div class="cta-row" style="margin-top:28px">{paper_cta}</div>
  </div>
</section>

<section class="section" id="open-source">
  <div class="wrap">
    {repo_head}
    <div class="repo-grid">{repos}</div>
    <div class="cta-row" style="margin-top:24px">{repo_cta}</div>
  </div>
</section>

<section class="section" id="notes">
  <div class="wrap">
    {blog_head}
    {blog_list}
  </div>
</section>

<section class="section section--tight">
  <div class="wrap">
    <div class="card" style="padding:32px" data-reveal>
      <div class="grid grid--2" style="align-items:center;gap:28px">
        <div>
          <h2 style="font-size:clamp(1.4rem,2.6vw,1.9rem)">{contact_title}</h2>
        </div>
        <div class="btn-row-inline" style="justify-content:flex-start;gap:10px">
          {contact_ctas}
        </div>
      </div>
    </div>
  </div>
</section>
""".format(
        eyebrow=esc(t({"zh": "自动驾驶算法负责人 · 求职中", "en": "Head of AD algorithms · open to work"}, lang)),
        headline=t(PROFILE["headline"], lang),
        lede=t(PROFILE["lede"], lang),
        status=esc(t(PROFILE["status"], lang)),
        location=esc(t(PROFILE["location"], lang)),
        role=esc(t(PROFILE["title"], lang)),
        cta_primary='<a class="btn btn--primary" href="#papers">%s<span>%s</span></a>'
                    % (icon("spark"), esc("看论文与项目" if lang == "zh" else "See papers & projects")),
        cta_github='<a class="btn" href="%s" target="_blank" rel="noopener">%s<span>GitHub</span></a>'
                   % (esc(PROFILE["github"]), icon_solid("github")),
        email=esc(PROFILE["email"]),
        copy_icon=icon("copy"),
        copy_label=esc(ui("copy_email", lang)),
        stats=stat_strip(lang),
        name=esc(t(PROFILE["name"], lang)),
        now_label=esc("目前" if lang == "zh" else "Now"),
        now_value=esc(t(PROFILE["org"], lang)),
        focus_label=esc("关键词" if lang == "zh" else "Keywords"),
        focus_value=esc("VLA · World Action Model"),
        focus_head=section_head(lang, "方向" if lang == "zh" else "Focus", level=2),
        focus=focus_cards(lang),
        career_head=section_head(lang, "经历" if lang == "zh" else "Experience", level=2),
        career=career_timeline(lang, limit=3, show_bullets=True),
        career_cta=btn(url(lang, "experience.html"),
                       "全部经历" if lang == "zh" else "All experience", "btn btn--ghost"),
        paper_head=section_head(lang, "论文" if lang == "zh" else "Publications", level=2),
        papers="".join(paper_card(lang, p, compact=True) for p in fp),
        paper_cta=btn(url(lang, "publications.html"),
                      "全部论文" if lang == "zh" else "All publications", "btn"),
        repo_head=section_head(lang, "开源" if lang == "zh" else "Open source", level=2),
        repos="".join(repo_card(lang, r) for r in featured_repos),
        repo_cta=btn(url(lang, "opensource.html"),
                     "全部仓库" if lang == "zh" else "All repositories", "btn btn--ghost"),
        blog_head=section_head(lang, "笔记" if lang == "zh" else "Notes", level=2),
        blog_list=blog_list_html(lang, posts[:3], external_links=(lang != "zh")),
        contact_title=esc("联系" if lang == "zh" else "Contact"),
        contact_ctas=(btn("mailto:" + PROFILE["email"], PROFILE["email"], "btn btn--primary", "mail")
                      + btn("/resume/Lidong-Yu-Resume-Detailed-EN.pdf", ui("resume", lang), "btn", "download")),
    )

    jsonld = person_jsonld(lang)
    return page(lang, "%s · %s" % (t(PROFILE["name"], lang), t(PROFILE["title"], lang)),
                plain(t(PROFILE["lede"], lang)), "index.html", body, active="home", jsonld=jsonld)


def blog_list_html(lang, posts, external_links=False):
    if not posts:
        return '<p class="muted">%s</p>' % esc(ui("empty_blog", lang))
    items = []
    for p in posts:
        title = p["title"] if lang == "zh" else p["title_en"]
        summary = p["summary"] if lang == "zh" else p["summary_en"]
        # Chinese-only posts: the English index links straight to the Chinese page
        href = ("/blog/%s.html" % p["slug"]) if (external_links or lang == "zh") else url(lang, "blog/%s.html" % p["slug"])
        tags = " / ".join(p["tags"] if lang == "zh" else p.get("tags_en") or p["tags"])
        note = ""
        if lang != "zh":
            badge = "中文" if not external_links else "Chinese"
            note = '<span class="tag tag--mono">%s</span>' % badge
        items.append("""
      <article class="post-item">
        <time datetime="{date}">{date}</time>
        <div>
          <h3><a href="{href}">{title}</a></h3>
          <p>{summary}</p>
          <div class="post-item__meta">{note}<span>{tags}</span><span>·</span><span>{minutes} {min_label}</span></div>
        </div>
      </article>""".format(
            date=esc(p["date"]), href=esc(href), title=esc(title), summary=esc(summary),
            note=note, tags=esc(tags), minutes=p["minutes"],
            min_label=esc(ui("min_read", lang))))
    return '<div class="post-list">%s</div>' % "".join(items)


def person_jsonld(lang):
    return {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": t(PROFILE["name"], lang),
        "alternateName": "Lidong Yu",
        "jobTitle": t(PROFILE["title"], lang),
        "worksFor": {"@type": "Organization", "name": t(PROFILE["org"], lang)},
        "url": L_.SITE_URL,
        "email": "mailto:" + PROFILE["email"],
        "sameAs": [PROFILE["github"], PROFILE["scholar"]],
        "knowsAbout": ["Vision-Language-Action", "World Models", "Autonomous Driving",
                       "Stereo Matching", "Depth Estimation", "Data Closed Loop"],
        "alumniOf": [
            {"@type": "CollegeOrUniversity", "name": t(e["school"], lang)} for e in EDUCATION
        ],
    }


def render_experience(lang):
    rows = []
    for c in CAREER:
        bullets = "".join("<li>%s</li>" % t(b, lang) for b in c["bullets"])
        rows.append("""
      <article class="card" data-reveal style="margin-bottom:16px">
        <div class="card__top">
          <div class="links">{when}</div>
          <div class="links">{tags}</div>
        </div>
        <h3 style="font-size:1.15rem">{company}</h3>
        <div class="tl-role" style="margin-top:6px">{role}</div>
        <div class="tl-scope">{scope}</div>
        <p style="margin-top:12px">{summary}</p>
        <ul style="margin-top:14px;padding-left:18px;display:grid;gap:8px">{bullets}</ul>
      </article>""".format(
            when=tag(t(c["when"], lang) if isinstance(c["when"], dict) else c["when"], "tag--mono"),
            tags="".join(tag(x, "tag--mono") for x in c["tags"][:4]),
            company=esc(t(c["company"], lang)),
            role=esc(t(c["role"], lang)),
            scope=esc(t(c["scope"], lang)),
            summary=t(c["summary"], lang),
            bullets=bullets,
        ))

    edu = "".join("""
      <div class="card" data-reveal>
        <div class="card__top">{badge}{when}</div>
        <h3>{school}</h3>
        <p style="margin-top:8px">{degree} · {dept}</p>
        <p class="small muted" style="margin-top:6px">{where}{note}</p>
      </div>""".format(
        badge=tag(e["badge"], "tag--venue"),
        when=tag(e["when"], "tag--mono"),
        school=esc(t(e["school"], lang)),
        degree=esc(t(e["degree"], lang)),
        dept=esc(t(e["dept"], lang)),
        where=esc(t(e["where"], lang)),
        note=(" · " + esc(t(e["note"], lang))) if e["note"] else "",
    ) for e in EDUCATION)

    pats = "".join(
        '<li><span>%s</span><span class="mono small muted">%s</span></li>' % (esc(t(p["title"], lang)), esc(p["no"]))
        for p in PATENTS
    )

    body = """
<section class="section">
  <div class="wrap">
    {head}
    {rows}
  </div>
</section>
<section class="section">
  <div class="wrap wrap--narrow">
    {edu_head}
    <div class="grid grid--2">{edu}</div>
  </div>
</section>
<section class="section">
  <div class="wrap wrap--narrow">
    {pat_head}
    <div class="card">
      <ul style="list-style:none;padding:0;display:grid;gap:12px" class="pat-list">{pats}</ul>
    </div>
    <p class="small muted" style="margin-top:14px">{pat_note}</p>
  </div>
</section>
""".format(
        head=section_head(lang, "职业经历" if lang == "zh" else "Experience", level=1),
        rows="".join(rows),
        edu_head=section_head(lang, "教育" if lang == "zh" else "Education", level=2),
        edu=edu,
        pat_head=section_head(lang, "专利" if lang == "zh" else "Patents", level=2),
        pats=pats,
        pat_note=esc("均为中国发明专利，编号为申请号。" if lang == "zh"
                     else "All Chinese invention patents; numbers are application numbers."),
    )
    return page(lang, ("经历 · " if lang == "zh" else "Experience · ") + t(PROFILE["name"], lang),
                "工作经历、教育与专利。" if lang == "zh" else "Work experience, education and patents.",
                "experience.html", body, active="experience")


def render_publications(lang):
    cards = "".join(paper_card(lang, p) for p in PAPERS)
    rows = []
    for i, p in enumerate(PAPERS, 1):
        rows.append(
            "<tr><td>%d</td><td>%s</td><td>%s %s</td><td>%s</td></tr>"
            % (i, esc(p["title"]), esc(p["venue"]), esc(p["year"]),
               '<a href="%s" target="_blank" rel="noopener">paper</a>' % esc(p["paper"])
               + ((' · <a href="%s" target="_blank" rel="noopener">code</a>' % esc(p["code"]))
                  if p.get("code") else "")))

    mcards = []
    for m in MANUSCRIPTS:
        pdfs = "".join(
            link_chip(releases.url_for(fn) or ("/assets/pdf/" + fn.replace(" ", "%20")),
                      label if lang == "zh" else label_en, "doc", external=True)
            for fn, label, label_en in m["pdfs"]
        )
        code = link_chip(m["code"], "Code", "github", external=True) if m["code"] else ""
        mcards.append("""
        <article class="card" data-reveal>
          <div class="card__top">{status}</div>
          <h3>{title}</h3>
          <p style="margin-top:10px">{summary}</p>
          <div class="card__foot"><div class="links">{pdfs}{code}{page_link}</div></div>
        </article>""".format(
            status=tag(m["status"]["zh"] if lang == "zh" else m["status"]["en"], "tag--mono"),
            title=esc(m["title"]),
            summary=t(m["summary"], lang),
            pdfs=pdfs, code=code,
            page_link=link_chip(m["page"], "项目页" if lang == "zh" else "Project page", "link"),
        ))

    body = """
<section class="section">
  <div class="wrap">
    {head}
    <div class="stack-lg">{cards}</div>
  </div>
</section>
<section class="section">
  <div class="wrap wrap--narrow">
    {table_head}
    <div class="card" style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="text-align:left">
          <th style="padding:8px 10px;border-bottom:1px solid var(--line)">#</th>
          <th style="padding:8px 10px;border-bottom:1px solid var(--line)">{th_title}</th>
          <th style="padding:8px 10px;border-bottom:1px solid var(--line)">{th_venue}</th>
          <th style="padding:8px 10px;border-bottom:1px solid var(--line)">{th_link}</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </div>
</section>
<section class="section">
  <div class="wrap wrap--narrow">
    {ms_head}
    <div class="notice notice--warn" style="margin-bottom:20px">
      <div class="notice__title">{ms_notice_title}</div>
      <div>{ms_notice}</div>
    </div>
    <div class="grid grid--2">{mcards}</div>
  </div>
</section>
""".format(
        head=section_head(lang, "论文" if lang == "zh" else "Publications", level=1),
        cards=cards,
        table_head=section_head(lang, "一览表" if lang == "zh" else "At a glance", level=2),
        th_title=esc("标题" if lang == "zh" else "Title"),
        th_venue=esc("会议 / 年份" if lang == "zh" else "Venue / year"),
        th_link=esc("链接" if lang == "zh" else "Links"),
        rows="".join(rows),
        ms_head=section_head(lang, "投稿存档" if lang == "zh" else "Manuscript archive", level=2),
        ms_notice_title=esc("关于诚实" if lang == "zh" else "A note on honesty"),
        ms_notice=esc("我把它们单独存档、明确标注「未发表」，而不是混进论文列表。"
                      if lang == "zh" else
                      "They are archived separately and clearly marked as unpublished rather than mixed into the "
                      "publication list."),
        mcards="".join(mcards),
    )
    return page(lang, ("论文 · " if lang == "zh" else "Publications · ") + t(PROFILE["name"], lang),
                "8 篇同行评审论文（ICCV 2025 Oral、AAAI 2018 spotlight、CVPR、NeurIPS），含问题、方法、指标与论文/代码链接。"
                if lang == "zh" else
                "Eight peer-reviewed papers (ICCV 2025 Oral, AAAI 2018 spotlight, CVPR, NeurIPS) with problem, "
                "method, metrics and paper/code links.",
                "publications.html", body, active="publications")


def render_foggy(lang):
    from .data_works import FOGGY
    links = "".join(
        link_chip(href, t(label, lang), icon_name, external=href.startswith("http"))
        for href, label, icon_name in FOGGY["links"]
    )
    body = """
<section class="section section--tight">
  <div class="wrap wrap--narrow">
    <a class="small" href="{projects}">← {back}</a>
  </div>
</section>
<section class="section">
  <div class="wrap wrap--narrow">
    <div class="sec-kicker">{venue}</div>
    <h1 style="font-size:clamp(1.7rem,4vw,2.5rem);letter-spacing:-.03em;max-width:26ch;margin-top:10px">{title}</h1>
    <p class="muted" style="margin-top:12px">{authors}</p>
    <div class="card__tags" style="margin-top:16px">{tags}</div>
    <figure style="margin-top:28px">
      <img src="{figure}" alt="{title}" loading="lazy"
           style="border:1px solid var(--line);border-radius:var(--radius);background:var(--surface-2)">
      <figcaption class="small muted" style="margin-top:10px">{cap}</figcaption>
    </figure>
    <div class="stack-lg" style="margin-top:36px">
      <div><h2 style="font-size:1.15rem">{p_label}</h2><p class="muted" style="margin-top:8px">{problem}</p></div>
      <div><h2 style="font-size:1.15rem">{m_label}</h2><p class="muted" style="margin-top:8px">{method}</p></div>
      <div><h2 style="font-size:1.15rem">{r_label}</h2><p class="muted" style="margin-top:8px">{results}</p></div>
    </div>
    <div class="links" style="margin-top:32px">{links}</div>
  </div>
</section>
""".format(
        projects=url(lang, "projects.html"),
        back=esc("返回项目" if lang == "zh" else "All projects"),
        venue=esc(FOGGY["venue"] + (" · 已发表" if lang == "zh" else " · published")),
        title=esc(FOGGY["title"] if lang == "zh" else FOGGY["title"]),
        authors=esc(FOGGY["authors"]),
        tags="".join(tag(x) for x in FOGGY["tags"]),
        figure=esc(FOGGY["figure"]),
        cap=esc(t(FOGGY["figure_cap"], lang)),
        p_label=esc("问题" if lang == "zh" else "Problem"),
        problem=t(FOGGY["problem"], lang),
        m_label=esc("方法" if lang == "zh" else "Method"),
        method=t(FOGGY["method"], lang),
        r_label=esc("结果" if lang == "zh" else "Results"),
        results=t(FOGGY["results"], lang),
        links=links,
    )
    return page(lang, "%s · %s" % (FOGGY["title"], t(PROFILE["name"], lang)),
                t(FOGGY["problem"], lang)[:150],
                "project9.html", body, active="projects")


def render_projects(lang):
    cards = "".join(project_card(lang, p) for p in PROJECT_PAGES)
    work_blocks = []
    for c in CAREER:
        work_blocks.append("""
        <div class="card" data-reveal>
          <div class="card__top">{when}</div>
          <h3>{company}</h3>
          <div class="tl-role" style="margin-top:6px">{role}</div>
          <p style="margin-top:12px">{summary}</p>
          <ul style="margin-top:12px;padding-left:18px;display:grid;gap:6px">{bullets}</ul>
        </div>""".format(
            when=tag(t(c["when"], lang) if isinstance(c["when"], dict) else c["when"], "tag--mono"),
            company=esc(t(c["company"], lang)),
            role=esc(t(c["role"], lang)),
            summary=t(c["summary"], lang),
            bullets="".join("<li>%s</li>" % t(b, lang) for b in c["bullets"][:3]),
        ))

    body = """
<section class="section">
  <div class="wrap">
    {head}
    <div class="grid grid--3">{cards}</div>
  </div>
</section>
<section class="section">
  <div class="wrap">
    {deliver_head}
    <div class="grid grid--2">{work}</div>
  </div>
</section>
""".format(
        head=section_head(lang, "项目" if lang == "zh" else "Projects", level=1),
        cards=cards,
        deliver_head=section_head(lang, "量产交付" if lang == "zh" else "Shipped work", level=2),
        work="".join(work_blocks),
    )
    return page(lang, ("项目 · " if lang == "zh" else "Projects · ") + t(PROFILE["name"], lang),
                "8 个项目详解：立体匹配、单目深度、跟踪、模仿学习、视频目标检测，以及背后的量产交付。"
                if lang == "zh" else
                "Eight project deep-dives: stereo matching, monocular depth, tracking, imitation learning and video "
                "object detection, plus the production work behind them.",
                "projects.html", body, active="projects")


def render_opensource(lang):
    counts = {}
    for r in REPOS:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    chips = ['<button class="chip" type="button" data-filter="all" aria-pressed="true">%s<span class="count">%d</span></button>'
             % (esc("全部" if lang == "zh" else "All"), len(REPOS))]
    for g in REPO_GROUPS:
        chips.append('<button class="chip" type="button" data-filter="%s" aria-pressed="false">%s<span class="count">%d</span></button>'
                     % (g["kind"], esc(t(g["label"], lang)), counts.get(g["kind"], 0)))
    chips.append('<button class="chip" type="button" data-filter="finance" aria-pressed="false">%s<span class="count">%d</span></button>'
                 % (esc("金融 · Finance" if lang == "zh" else "Finance"), len(FINANCE["items"])))

    groups = []
    for g in REPO_GROUPS:
        items = [r for r in REPOS if r["kind"] == g["kind"]]
        groups.append("""
      <div data-group style="margin-bottom:36px">
        <div class="sec-head" style="margin-bottom:16px">
          <div class="sec-head__text">
            <h3 style="font-size:1.15rem">{label}</h3>
            <p class="small" style="margin-top:6px">{note}</p>
          </div>
        </div>
        <div class="repo-grid">{cards}</div>
      </div>""".format(
            label=esc(t(g["label"], lang)), note=esc(t(g["note"], lang)),
            cards="".join(repo_card(lang, r) for r in items)))

    fin_items = []
    for it in FINANCE["items"]:
        fin_items.append("""
        <div class="repo" data-kind="finance" data-reveal>
          <div class="repo__name">{icon}<a href="{url}" target="_blank" rel="noopener">{name}</a></div>
          <div class="repo__desc">{desc}</div>
          <div class="repo__meta"><a href="{up}" target="_blank" rel="noopener">upstream ↗</a></div>
        </div>""".format(icon=icon_solid("github"), url=esc(it["url"]), name=esc(it["name"]),
                          desc=esc(t(it["desc"], lang)), up=esc(it["upstream"])))

    body = """
<section class="section">
  <div class="wrap">
    {head}
    <div class="notice" style="margin-bottom:28px">
      <div class="notice__title">{count_title}</div>
      <div>{count_body}</div>
    </div>
    <div class="filters">{chips}</div>
    {groups}
    <div data-group style="margin-bottom:36px">
      <div class="sec-head" style="margin-bottom:16px">
        <div class="sec-head__text">
          <h3 style="font-size:1.15rem">{fin_title}</h3>
          <p class="small" style="margin-top:6px">{fin_lead}</p>
        </div>
      </div>
      <div class="repo-grid">{fin_items}</div>
      <p class="small muted" style="margin-top:14px">{fin_note}</p>
    </div>
    <div class="notice notice--warn">
      <div class="notice__title">{fork_title}</div>
      <div>{fork_body} <a href="{github}" target="_blank" rel="noopener">github.com/{handle}</a></div>
    </div>
  </div>
</section>
""".format(
        head=section_head(lang, "开源" if lang == "zh" else "Open source", level=1),
        count_title=esc("口径说明" if lang == "zh" else "How this is counted"),
        count_body=esc("GitHub 上我有 134 个公开仓库，其中只有 39 个是我自己创建的，另外 95 个是 fork。"
                       "这里只展示我自己写的 39 个，按类型分类；fork 不列为作品。"
                       if lang == "zh" else
                       "I have 134 public repos; only 39 were created by me and 95 are forks. Only my own 39 appear "
                       "here, grouped by type — forks are not presented as work."),
        chips="".join(chips),
        groups="".join(groups),
        fin_title=esc(t(FINANCE["title"], lang)),
        fin_lead=esc(t(FINANCE["lead"], lang)),
        fin_items="".join(fin_items),
        fin_note=esc(t(FINANCE["note"], lang) if False else ""),
        fork_title=esc(t(FORKS["title"], lang)),
        fork_body=esc(t(FORKS["body"], lang)),
        github=esc(PROFILE["github"]), handle=esc(PROFILE["handle"]),
    )
    return page(lang, ("开源 · " if lang == "zh" else "Open source · ") + t(PROFILE["name"], lang),
                "39 个自建 GitHub 仓库，按研究、工具、知识媒体、归档与金融方向分类。"
                if lang == "zh" else
                "39 GitHub repos I created, grouped into research, tools, knowledge/media, archive and finance.",
                "opensource.html", body, active="opensource")


def render_blog_index(lang, posts):
    body = """
<section class="section">
  <div class="wrap wrap--narrow">
    {head}
    {list}
    <div class="notice" style="margin-top:32px">
      <div class="notice__title">RSS</div>
      <div>{rss_note} <a href="{feed}">{feed}</a></div>
    </div>
  </div>
</section>
""".format(
        head=section_head(lang, "博客" if lang == "zh" else "Blog", level=1),
        list=blog_list_html(lang, posts, external_links=(lang != "zh")),
        rss_note=esc("订阅：" if lang == "zh" else "Subscribe: "),
        feed=("/blog/feed.xml" if lang == "zh" else "/en/blog/feed.xml"),
    )
    return page(lang, (ui("blog_title", lang) + " · " + t(PROFILE["name"], lang)),
                t_ui_desc(lang), "blog/index.html", body, active="blog")


def t_ui_desc(lang):
    return t({"zh": "关于 VLA、世界模型、数据闭环与感知工程落地的研究笔记。",
              "en": "Research notes on VLA, world models, data loops and perception engineering."}, lang)


def render_post(lang, post, posts):
    title = post["title"] if lang == "zh" else post["title_en"]
    summary = post["summary"] if lang == "zh" else post["summary_en"]
    tags = "".join(tag(x) for x in post["tags"])
    others = [p for p in posts if p["slug"] != post["slug"]][:2]
    more = ""
    if others:
        more = """
<section class="section section--tight">
  <div class="wrap wrap--narrow">
    <h2 style="font-size:1.1rem;margin-bottom:12px">%s</h2>
    %s
  </div>
</section>""" % (esc("继续读" if lang == "zh" else "Keep reading"), blog_list_html(lang, others))

    body = """
<article>
  <header class="post-header wrap wrap--narrow">
    <a class="small" href="{blog}">← {back}</a>
    <h1 style="margin-top:14px">{title}</h1>
    <div class="post-meta">
      <time datetime="{date}">{date}</time><span class="sep"></span>
      <span>{minutes} {min_label}</span><span class="sep"></span>
      <span>{tags}</span>
    </div>
  </header>
  <div class="wrap wrap--narrow">
    <div class="prose">{html}</div>
  </div>
</article>
{more}
""".format(
        blog=url(lang, "blog/index.html"), back=esc(ui("back_blog", lang)),
        title=esc(title), date=esc(post["date"]), minutes=post["minutes"],
        min_label=esc(ui("min_read", lang)), tags=tags,
        html=post["html"], more=more,
    )
    jsonld = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "datePublished": post["date"],
        "inLanguage": "zh-CN",
        "author": {"@type": "Person", "name": "Lidong Yu"},
        "description": summary,
        "mainEntityOfPage": L_.abs_url(lang, "blog/%s.html" % post["slug"]),
    }
    return page(lang, title + " · " + t(PROFILE["name"], lang), summary,
                "blog/%s.html" % post["slug"], body, active="blog",
                jsonld=jsonld, show_progress=True)


def render_about(lang):
    resume_rows = [
        ("EN", "/resume/Lidong-Yu-Resume-EN.pdf", "2 页 · 英文" if lang == "zh" else "2 pages · English"),
        ("中文", "/resume/Lidong-Yu-Resume-CN.pdf", "2 页 · 中文" if lang == "zh" else "2 pages · Chinese"),
        ("EN+", "/resume/Lidong-Yu-Resume-Detailed-EN.pdf",
         "3 页 · 英文详细版" if lang == "zh" else "3 pages · detailed English"),
        ("中文+", "/resume/Lidong-Yu-Resume-Detailed-CN.pdf",
         "3 页 · 中文详细版" if lang == "zh" else "3 pages · detailed Chinese"),
    ]
    rbuttons = "".join(
        '<a class="link-chip" href="%s" target="_blank" rel="noopener">%s<span>%s</span></a>'
        % (esc(href), icon("download"), esc(label + " · " + meta))
        for label, href, meta in resume_rows
    )

    body = """
<section class="section">
  <div class="wrap wrap--narrow">
    {head}
    <div class="grid grid--2" style="align-items:start;gap:28px">
      <div class="prose" style="font-size:16px">
        <p>{p1}</p>
        <p>{p2}</p>
        <p>{p3}</p>
      </div>
      <div class="card">
        <h3>{contact_title}</h3>
        <div class="stack-sm" style="margin-top:14px">
          <div><span class="small muted">{lbl_email}</span><br><a href="mailto:{email}">{email}</a></div>
          <div><span class="small muted">{lbl_phone}</span><br><span class="mono">{phone}</span></div>
          <div><span class="small muted">{lbl_loc}</span><br>{loc}</div>
          <div><span class="small muted">{lbl_web}</span><br>
            <a href="{github}" target="_blank" rel="noopener">GitHub</a> ·
            <a href="{scholar}" target="_blank" rel="noopener">Scholar</a>
          </div>
        </div>
        <div class="btn-row-inline" style="margin-top:20px">
          <button class="btn btn--sm" type="button" data-copy="{email}">{copy_icon}<span>{copy_label}</span></button>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:24px">
      <h3>{resume_title}</h3>
      <p class="small muted" style="margin-top:6px">{resume_desc}</p>
      <div class="links" style="margin-top:16px">{rbuttons}</div>
    </div>
  </div>
</section>
<section class="section">
  <div class="wrap wrap--narrow">
    {timeline_head}
    {timeline}
  </div>
</section>
<section class="section">
  <div class="wrap wrap--narrow">
    {build_head}
    <div class="prose" style="font-size:15px">
      <p>{build_p}</p>
      <ul>
        <li>{build_li1}</li>
        <li>{build_li2}</li>
        <li>{build_li3}</li>
      </ul>
    </div>
  </div>
</section>
""".format(
        head=section_head(lang, "关于" if lang == "zh" else "About", level=1),
        p1=esc("我是于立冬，自动驾驶算法负责人，现在在新石器负责 AD 算法团队（38 人、约 8,700 张 GPU）。"
               "我的技术主线是「把视觉模型的表示能力落到车上的实时约束里」："
               "从立体匹配与单目深度出发，经过跟踪、模仿学习，一路做到今天的端到端与世界模型。"
               if lang == "zh" else
               "I am Lidong Yu, head of autonomous-driving algorithms at Neolix, where I lead a 38-person AD "
               "algorithm org with ~8,700 GPUs. My technical through-line is putting the representational power of "
               "vision models inside the real-time budget of a car: starting from stereo matching and monocular depth, "
               "through tracking and imitation learning, all the way to today's end-to-end driving and world models."),
        p2=esc("我相信「落地」不是一句口号，而是一串可以被核对的东西：模型换代能不能进端到端、"
               "数据闭环能不能把 issue 从 D+10 压到 D+3、世界模型能不能当评测装置而不只是渲染器。"
               "过去几年我在不同公司反复验证这三件事，也都留下了可量化的结果。"
               if lang == "zh" else
               "I believe deployment is not a slogan but a list of checkable things: whether a model generation can "
               "enter the end-to-end stack, whether the data loop can move issue turnaround from D+10 to D+3, and "
               "whether a world model can act as an evaluation harness rather than merely a renderer. I have verified "
               "all three across several companies, each time with numbers attached."),
        p3=esc("我正在寻找 VLA / 世界模型落地方向的机会：既愿意带团队，也愿意自己写代码。"
               if lang == "zh" else
               "I am looking for a role in landing VLA and world models — happy to lead a team, and happy to write "
               "the code myself."),
        contact_title=esc("联系方式" if lang == "zh" else "Contact"),
        lbl_email=esc("邮箱" if lang == "zh" else "Email"),
        lbl_phone=esc("电话" if lang == "zh" else "Phone"),
        lbl_loc=esc("坐标" if lang == "zh" else "Based in"),
        lbl_web=esc("在线" if lang == "zh" else "Elsewhere"),
        email=esc(PROFILE["email"]), phone=esc(PROFILE["phone"]),
        loc=esc(t(PROFILE["location"], lang)),
        github=esc(PROFILE["github"]), scholar=esc(PROFILE["scholar"]),
        copy_icon=icon("copy"), copy_label=esc(ui("copy_email", lang)),
        resume_title=esc("简历下载" if lang == "zh" else "Resume downloads"),
        resume_desc=esc("四个版本：简版 / 详细版 × 中英文。"
                        if lang == "zh" else "Four versions: short / detailed × Chinese / English."),
        rbuttons=rbuttons,
        timeline_head=section_head(lang, "经历速览" if lang == "zh" else "Career at a glance", level=2),
        timeline=career_timeline(lang, limit=6, show_bullets=False),
        build_head=section_head(lang, "关于这个站点" if lang == "zh" else "About this site", level=2),
        build_p=esc("这是一个纯静态站点：生成脚本在 Python 标准库上运行，页面是预先渲染好的 HTML，"
                    "没有任何前端框架、外部字体或追踪脚本。旧版主页的内容全部保留在 archive 里。"
                    if lang == "zh" else
                    "A fully static site: the generator runs on the Python standard library, pages are pre-rendered "
                    "HTML, and there is no front-end framework, external font or tracking script. Everything from the "
                    "previous version of the site is preserved under archive."),
        build_li1=esc("语义化 HTML + 手写 CSS，零运行时依赖，首屏无阻塞资源。"
                      if lang == "zh" else
                      "Semantic HTML and hand-written CSS with zero runtime dependencies and no blocking first-paint assets."),
        build_li2=esc("同时输出中英文两套页面，双语互链并标注 hreflang。"
                      if lang == "zh" else
                      "Bilingual output with cross-links and hreflang annotations."),
        build_li3=esc("提供 RSS 与结构化数据（JSON-LD），便于被检索与分享。"
                      if lang == "zh" else
                      "RSS plus JSON-LD structured data, so it is easy to index and share."),
    )
    return page(lang, ("关于 · " if lang == "zh" else "About · ") + t(PROFILE["name"], lang),
                "于立冬（Lidong Yu）：自动驾驶算法负责人，VLA 与世界模型落地。简历下载、联系方式与经历速览。"
                if lang == "zh" else
                "Lidong Yu: head of autonomous-driving algorithms, VLA and world-model deployment. Resume downloads, "
                "contact details and a career overview.",
                "about.html", body, active="about")


def render_404(lang="zh"):
    body = """
<section class="section" style="min-height:52vh;display:grid;place-items:center">
  <div class="wrap wrap--narrow" style="text-align:center">
    <div class="sec-kicker">404</div>
    <h1 style="font-size:clamp(2rem,6vw,3.2rem);margin-top:10px">%s</h1>
    <p class="muted" style="margin-top:14px;max-width:46ch;margin-inline:auto">%s</p>
    <div class="cta-row" style="justify-content:center">%s%s</div>
  </div>
</section>""" % (
        esc("这个页面不在了" if lang == "zh" else "This page is gone"),
        esc("链接可能已经更新。可以从主页重新开始，或者直接找我要东西。"
            if lang == "zh" else
            "The link may have moved. Start again from the home page, or just ask me for what you need."),
        btn("/", "回到主页" if lang == "zh" else "Home", "btn btn--primary", "arrow"),
        btn("mailto:" + PROFILE["email"], PROFILE["email"], "btn", "mail"),
    )
    return page(lang, "404 · " + t(PROFILE["name"], lang),
                "页面不存在" if lang == "zh" else "Page not found",
                "404.html", body, active="", noindex=True)
