# -*- coding: utf-8 -*-
"""Tiny Markdown subset renderer + front-matter parser (no dependencies)."""

import re
import html


def parse_front_matter(text):
    fm = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            raw = text[3:end]
            body = text[end + 4:].lstrip("\n")
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, body


def _inline(s):
    s = html.escape(s, quote=False)
    # inline code first so its content is not mangled
    codes = []

    def stash(m):
        codes.append(m.group(1))
        return "\x00%d\x00" % (len(codes) - 1)

    s = re.sub(r"`([^`]+)`", stash, s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)",
               lambda m: '<img src="%s" alt="%s" loading="lazy">' % (m.group(2), m.group(1)), s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
               lambda m: '<a href="%s">%s</a>' % (m.group(2), m.group(1)), s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", s)

    def unstash(m):
        return "<code>%s</code>" % codes[int(m.group(1))]

    s = re.sub(r"\x00(\d+)\x00", unstash, s)
    return s


def render_markdown(text):
    lines = text.replace("\r\n", "\n").split("\n")
    out = []
    i = 0
    n = len(lines)

    def flush_para(buf):
        if buf:
            out.append("<p>%s</p>" % _inline(" ".join(x.strip() for x in buf)))
            buf.clear()

    para = []
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code
        if stripped.startswith("```"):
            flush_para(para)
            lang = stripped[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = ' class="language-%s"' % html.escape(lang) if lang else ""
            out.append("<pre><code%s>%s</code></pre>" % (cls, html.escape("\n".join(buf))))
            continue

        # horizontal rule
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush_para(para)
            out.append("<hr>")
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para(para)
            level = min(len(m.group(1)) + 1, 6) if len(m.group(1)) == 1 else len(m.group(1))
            level = len(m.group(1)) if len(m.group(1)) > 1 else 2
            text_h = m.group(2).strip()
            anchor = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text_h).strip("-").lower()[:60]
            out.append('<h%d id="%s">%s</h%d>' % (level, html.escape(anchor), _inline(text_h), level))
            i += 1
            continue

        # blockquote
        if stripped.startswith(">"):
            flush_para(para)
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote><p>%s</p></blockquote>" % _inline(" ".join(buf)))
            continue

        # table
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            flush_para(para)
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join("<th>%s</th>" % _inline(c) for c in header)
            tb = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % _inline(c) for c in r) for r in rows)
            out.append("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (th, tb))
            continue

        # lists
        if re.match(r"^([-*+]|\d+\.)\s+", stripped):
            flush_para(para)
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            items = []
            base_indent = len(line) - len(line.lstrip())
            while i < n:
                cur = lines[i]
                cs = cur.strip()
                m2 = re.match(r"^([-*+]|\d+\.)\s+(.*)$", cs)
                if m2:
                    items.append(m2.group(2).strip())
                    i += 1
                    continue
                if cs and (len(cur) - len(cur.lstrip())) > base_indent and items:
                    items[-1] += " " + cs
                    i += 1
                    continue
                break
            tag = "ol" if ordered else "ul"
            out.append("<%s>%s</%s>" % (tag, "".join("<li>%s</li>" % _inline(x) for x in items), tag))
            continue

        if not stripped:
            flush_para(para)
            i += 1
            continue

        para.append(line)
        i += 1

    flush_para(para)
    return "\n".join(out)


def plain(text, limit=180):
    """Strip markdown to a plain-text summary."""
    s = re.sub(r"```.*?```", " ", text, flags=re.S)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"[#>*_`~|-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit].rstrip() + ("…" if len(s) > limit else "")


def word_count(text):
    s = re.sub(r"```.*?```", " ", text, flags=re.S)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-.%]*", s))
    # ~400 CJK chars per minute, ~220 words per minute
    return max(1, round(cjk / 400 + words / 220))
