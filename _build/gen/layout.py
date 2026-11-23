# -*- coding: utf-8 -*-
"""HTML shell: head/meta, header/nav, footer, icons, JSON-LD."""

import html

from .data_core import PROFILE, L

SITE_URL = "https://lidongyv.github.io"
SITE_NAME = "Lidong Yu · 于立冬"

LANGS = ("zh", "en")


def t(field, lang):
    """Resolve a bilingual field."""
    if isinstance(field, dict) and "zh" in field:
        return field[lang]
    return field


def esc(s):
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------------
# Icons (inline SVG, 24x24 grid, stroke-based)
# --------------------------------------------------------------------------
_ICONS = {
    "github": '<path d="M12 2C6.48 2 2 6.58 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.2-3.37-1.2-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.62.07-.62 1 .07 1.53 1.06 1.53 1.06.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.06 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05a9.3 9.3 0 0 1 2.5-.34c.85 0 1.71.12 2.5.34 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.93-2.35 4.8-4.58 5.05.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.8 0 .27.18.6.69.49A10.02 10.02 0 0 0 22 12.25C22 6.58 17.52 2 12 2Z"/>',
    "scholar": '<path d="M12 3 2 8l10 5 10-5-10-5Z"/><path d="M6 10.5V15c0 1.7 2.7 3 6 3s6-1.3 6-3v-4.5"/><path d="M22 8v6"/>',
    "mail": '<rect x="2.5" y="4.5" width="19" height="15" rx="2.5"/><path d="m3 6.5 9 6.5 9-6.5"/>',
    "download": '<path d="M12 3v12"/><path d="m7.5 10.5 4.5 4.5 4.5-4.5"/><path d="M4 19.5h16"/>',
    "doc": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/>',
    "person": '<circle cx="12" cy="8" r="3.6"/><path d="M4.8 20.4c0-3.9 3.2-6.6 7.2-6.6s7.2 2.7 7.2 6.6"/>',
    "link": '<path d="M10.5 13.5a4 4 0 0 0 5.66 0l2.83-2.83a4 4 0 0 0-5.66-5.66L11.9 6.4"/><path d="M13.5 10.5a4 4 0 0 0-5.66 0L5 13.34a4 4 0 0 0 5.66 5.66l1.4-1.4"/>',
    "external": '<path d="M14 4h6v6"/><path d="M20 4 9.5 14.5"/><path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z"/>',
    "menu": '<path d="M4 7h16M4 12h16M4 17h16"/>',
    "close": '<path d="M6 6l12 12M18 6 6 18"/>',
    "code": '<path d="m9 8-5 4 5 4"/><path d="m15 8 5 4-5 4"/>',
    "arrow": '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    "arrow-up": '<path d="M12 19V5"/><path d="m6 11 6-6 6 6"/>',
    "copy": '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18Z"/>',
    "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4"/><path d="m6.3 6.3 2.8 2.8M14.9 14.9l2.8 2.8M6.3 17.7l2.8-2.8M14.9 9.1l2.8-2.8"/>',
}


def icon(name, cls=""):
    body = _ICONS.get(name, "")
    return ('<svg class="%s" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
            'aria-hidden="true" focusable="false">%s</svg>' % (cls, body))


def icon_solid(name, cls=""):
    body = _ICONS.get(name, "")
    return ('<svg class="%s" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" '
            'focusable="false">%s</svg>' % (cls, body))


# --------------------------------------------------------------------------
# Navigation + routes
# --------------------------------------------------------------------------
NAV = [
    ("home", L("主页", "Home"), "index.html"),
    ("experience", L("职业经历", "Experience"), "experience.html"),
    ("projects", L("项目", "Projects"), "projects.html"),
    ("publications", L("论文", "Publications"), "publications.html"),
    ("blog", L("博客", "Blog"), "blog/index.html"),
    ("opensource", L("开源", "Open source"), "opensource.html"),
]


def url(lang, path):
    """Route a relative page path for a language."""
    path = path.lstrip("/")
    if path in ("", "index.html"):
        return "/" if lang == "zh" else "/en/"
    return ("/" + path) if lang == "zh" else ("/en/" + path)


def abs_url(lang, path):
    return SITE_URL + url(lang, path)


# --------------------------------------------------------------------------
# UI strings
# --------------------------------------------------------------------------
UI = {
    "skip": L("跳到主要内容", "Skip to content"),
    "nav_label": L("主导航", "Main navigation"),
    "menu": L("菜单", "Menu"),
    "theme_light": L("切换到浅色主题", "Switch to light theme"),
    "theme_dark": L("切换到深色主题", "Switch to dark theme"),
    "lang_switch": L("English", "中文"),
    "lang_switch_label": L("切换语言", "Switch language"),
    "available": L("可到岗时间可谈", "Availability flexible"),
    "copy_email": L("复制邮箱", "Copy email"),
    "copied": L("已复制", "Copied"),
    "copy_failed": L("复制失败，请手动选择", "Copy failed — select manually"),
    "resume": L("下载简历", "Download CV"),
    "read_more": L("了解更多", "Read more"),
    "view_all": L("查看全部", "View all"),
    "papers": L("论文", "paper"),
    "blog_title": L("博客与研究笔记", "Blog & research notes"),
    "blog_desc": L("关于 VLA、世界模型、数据闭环与感知工程落地的一手经验。", "First-hand notes on VLA, world models, data loops and perception engineering."),
    "min_read": L("分钟阅读", "min read"),
    "published": L("发布于", "Published"),
    "toc": L("本文目录", "On this page"),
    "back_blog": L("返回博客", "Back to blog"),
    "empty_blog": L("第一篇文章正在整理中。", "The first post is on its way."),
    "footer_tagline": L("VLA / 世界模型落地 · 自动驾驶算法", "VLA & world-model deployment · autonomous driving algorithms"),
    "footer_explore": L("浏览", "Explore"),
    "footer_elsewhere": L("足迹", "Elsewhere"),
    "footer_note": L("本站为纯静态站点，无外部依赖、无追踪脚本。", "A fully static site: no external dependencies, no trackers."),
    "built": L("构建于", "Built"),
    "last_updated": L("最后更新", "Last updated"),
    "lang_note": L("本站为中英双语，右上角可切换。", "This site is bilingual — switch at the top right."),
}


def ui(key, lang):
    return t(UI[key], lang)


# --------------------------------------------------------------------------
# Page shell
# --------------------------------------------------------------------------
def head(lang, title, desc, path, alt_lang, og_image, extra_head="", jsonld=None, noindex=False):
    canonical = abs_url(lang, path)
    alt = abs_url(alt_lang, path)
    metas = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="theme-color" content="#ffffff">',
        "<title>%s</title>" % esc(title),
        '<meta name="description" content="%s">' % esc(desc),
        '<meta name="author" content="Lidong Yu">',
        '<link rel="canonical" href="%s">' % esc(canonical),
        '<link rel="alternate" hreflang="zh-CN" href="%s">' % esc(abs_url("zh", path)),
        '<link rel="alternate" hreflang="en" href="%s">' % esc(abs_url("en", path)),
        '<link rel="alternate" hreflang="x-default" href="%s">' % esc(abs_url("zh", path)),
        '<meta property="og:type" content="website">',
        '<meta property="og:site_name" content="%s">' % esc(SITE_NAME),
        '<meta property="og:title" content="%s">' % esc(title),
        '<meta property="og:description" content="%s">' % esc(desc),
        '<meta property="og:url" content="%s">' % esc(canonical),
        '<meta property="og:locale" content="%s">' % ("zh_CN" if lang == "zh" else "en_US"),
        '<meta property="og:locale:alternate" content="%s">' % ("en_US" if lang == "zh" else "zh_CN"),
        '<meta name="twitter:card" content="summary_large_image">',
        '<meta name="twitter:title" content="%s">' % esc(title),
        '<meta name="twitter:description" content="%s">' % esc(desc),
        '<link rel="stylesheet" href="/assets/site/site.css">',
    ]
    if og_image:
        metas += [
            '<meta property="og:image" content="%s">' % esc(SITE_URL + og_image),
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            '<meta name="twitter:image" content="%s">' % esc(SITE_URL + og_image),
        ]
    if noindex:
        metas.append('<meta name="robots" content="noindex">')
    icon_svg = (
        '<link rel="icon" href="/assets/site/favicon.svg" type="image/svg+xml">'
        '<link rel="apple-touch-icon" href="/assets/site/favicon.svg">'
    )
    metas.append(icon_svg)
    if extra_head:
        metas.append(extra_head)
    if jsonld:
        import json
        metas.append('<script type="application/ld+json">%s</script>'
                     % json.dumps(jsonld, ensure_ascii=False))
    return "\n".join("  " + m for m in metas)


def header(lang, active, alt_lang, alt_path=None):
    other_lang = alt_lang
    navs = []
    for key, label, path in NAV:
        cur = ' aria-current="page"' if key == active else ""
        navs.append('<a href="%s"%s>%s</a>' % (url(lang, path), cur, t(label, lang)))
    return """
  <header class="site-header">
    <div class="wrap header-inner">
      <a class="brand" href="{home}" aria-label="{name}">
        <span class="brand-mark">LY</span>
        <span class="brand-text">{short}<span class="brand-sub"> · {role}</span></span>
      </a>
      <button class="icon-btn nav-toggle" type="button" aria-label="{menu_l}" aria-expanded="false"
              aria-controls="site-nav">{menu_icon}</button>
      <nav class="nav" id="site-nav" aria-label="{nav_label}">
        {navs}
      </nav>
      <div class="header-tools">
        <button class="icon-btn" type="button" data-theme-toggle aria-pressed="false"
                data-label-light="{tl}" data-label-dark="{td}">{theme_icon}</button>
        <a class="lang-toggle" href="{alt_url}" hreflang="{alt_href}"
           aria-label="{lang_label}" title="{lang_label}">{lang_icon}<span>{lang_text}</span></a>
      </div>
    </div>
  </header>""".format(
        home=url(lang, "index.html"),
        name=esc(t(PROFILE["name"], lang)),
        short=esc(t(PROFILE["name"], lang)),
        role=esc(t(PROFILE["title"], lang)),
        menu_l=esc(ui("menu", lang)),
        menu_icon=icon("menu"),
        nav_label=esc(ui("nav_label", lang)),
        navs="\n        ".join(navs),
        tl=esc(ui("theme_light", lang)),
        td=esc(ui("theme_dark", lang)),
        theme_icon=icon("sun") + icon("moon"),
        alt_url=url(other_lang, alt_path) if alt_path else url(other_lang, _active_path(active)),
        alt_href="en" if lang == "zh" else "zh-CN",
        lang_label=esc(ui("lang_switch_label", lang)),
        lang_icon=icon("globe"),
        lang_text=esc(ui("lang_switch", lang)),
    )


def _active_path(active):
    for key, _label, path in NAV:
        if key == active:
            return path
    return "index.html"


def footer(lang):
    explore = "".join(
        '<a href="%s">%s</a>' % (url(lang, path), t(label, lang))
        for _k, label, path in NAV
    )
    explore += '<a href="%s">%s</a>' % (url(lang, "about.html"), t(L("关于", "About"), lang))
    return """
  <footer class="site-footer">
    <div class="wrap">
      <div class="footer-grid">
        <div>
          <div class="brand" style="margin-bottom:12px">
            <span class="brand-mark">LY</span>
            <span class="brand-text">{name}</span>
          </div>
          <p class="small muted" style="max-width:34ch">{tagline}</p>
        </div>
        <div>
          <h4>{explore_l}</h4>
          <div class="footer-links">{explore}</div>
        </div>
        <div>
          <h4>{elsewhere}</h4>
          <div class="footer-links">
            <a href="mailto:{email}">{email}</a>
            <a href="{github}" rel="me noopener">GitHub @{handle}</a>
            <a href="{scholar}" rel="noopener">Google Scholar</a>
            <a href="{feed}">RSS</a>
          </div>
        </div>
      </div>
      <div class="footer-bottom">
        <span>© <span data-year>2026</span> {name}</span>
      </div>
    </div>
  </footer>
  <button class="to-top" type="button" aria-label="{top_l}">{top_icon}</button>
  <div class="toast" role="status" aria-live="polite" data-copied="{copied}" data-failed="{failed}"></div>""".format(
        name=esc(t(PROFILE["name"], lang)),
        tagline=esc(t(UI["footer_tagline"], lang)),
        explore_l=esc(ui("footer_explore", lang)),
        explore=explore,
        elsewhere=esc(ui("footer_elsewhere", lang)),
        email=esc(PROFILE["email"]),
        github=esc(PROFILE["github"]),
        handle=esc(PROFILE["handle"]),
        scholar=esc(PROFILE["scholar"]),
        feed="/blog/feed.xml" if lang == "zh" else "/en/blog/feed.xml",
        top_l=esc("回到顶部" if lang == "zh" else "Back to top"),
        top_icon=icon("arrow-up"),
        copied=esc(ui("copied", lang)),
        failed=esc(ui("copy_failed", lang)),
    )


def page(lang, title, desc, path, body, active="", og_image="/assets/og/og-default.png",
         extra_head="", jsonld=None, noindex=False, show_progress=False, alt_path=None):
    alt_lang = "en" if lang == "zh" else "zh"
    progress = '<div class="reading-bar" aria-hidden="true"></div>' if show_progress else ""
    return """<!DOCTYPE html>
<html lang="{hl}" class="no-js" data-alt="{alt}">
<head>
{head}
</head>
<body>
{progress}
<a class="skip" href="#main">{skip}</a>
{header}
<main id="main">
{body}
</main>
{footer}
<script src="/assets/site/site.js" defer></script>
</body>
</html>
""".format(
        hl="zh-CN" if lang == "zh" else "en",
        alt=url(alt_lang, path),
        head=head(lang, title, desc, path, alt_lang, og_image, extra_head, jsonld, noindex),
        progress=progress,
        skip=esc(ui("skip", lang)),
        header=header(lang, active, alt_lang, alt_path),
        body=body,
        footer=footer(lang),
    )
