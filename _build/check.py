# -*- coding: utf-8 -*-
"""Structural QA for the generated site.

- HTML tag balance
- every internal href/src resolves to an existing file (with _build/out overlay)
- no template leftovers (%s, {name}, None, undefined)
- CSS brace balance
- basic a11y checks (img alt, single h1, lang attr)

    python _build/check.py
"""

import os
import re
import sys
import glob
import urllib.parse
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGE = os.path.join(HERE, "out")

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}

problems = []
notes = []


def resolve(rel):
    rel = urllib.parse.unquote(rel)
    rel = rel.split("#")[0].split("?")[0]
    if rel in ("", "/"):
        rel = "index.html"
    rel = rel.lstrip("/")
    if rel.endswith("/"):
        rel += "index.html"
    for base in (ROOT, STAGE):
        p = os.path.join(base, rel.replace("/", os.sep))
        if os.path.isfile(p):
            return p
    return None


class Checker(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.stack = []
        self.h1 = 0
        self.lang = None
        self.refs = []
        self.imgs_no_alt = 0
        self.ids = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang")
        if tag == "h1":
            self.h1 += 1
        if tag == "img" and "alt" not in a:
            self.imgs_no_alt += 1
        for key in ("href", "src"):
            v = a.get(key)
            if v:
                self.refs.append((v, tag))
        if a.get("id"):
            self.ids[a["id"]] = True
        if tag not in VOID and not self.get_starttag_text().rstrip().endswith("/>"):
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        """Self-closing tag: record references but never touch the stack."""
        a = dict(attrs)
        if tag == "img" and "alt" not in a:
            self.imgs_no_alt += 1
        for key in ("href", "src"):
            v = a.get(key)
            if v:
                self.refs.append((v, tag))
        if a.get("id"):
            self.ids[a["id"]] = True

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            problems.append("%s: stray </%s>" % (self.path, tag))
            return
        if self.stack[-1] == tag:
            self.stack.pop()
        else:
            # tolerate, but report
            if tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    problems.append("%s: unclosed <%s> before </%s>"
                                    % (self.path, self.stack.pop(), tag))
                if self.stack:
                    self.stack.pop()
            else:
                problems.append("%s: unexpected </%s>" % (self.path, tag))


def main():
    files = []
    for pattern in ("*.html", "en/*.html", "blog/*.html", "en/blog/*.html"):
        files += glob.glob(os.path.join(STAGE, pattern))
        files += glob.glob(os.path.join(ROOT, pattern))
    seen = set()
    uniq = []
    for f in files:
        key = os.path.relpath(f, ROOT).replace("\\", "/")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    files = sorted(uniq)
    notes.append("html files checked: %d" % len(files))

    internal_refs = 0
    broken = 0
    for f in files:
        rel = os.path.relpath(f, ROOT).replace("\\", "/")
        with open(f, encoding="utf-8") as fh:
            text = fh.read()

        c = Checker(rel)
        try:
            c.feed(text)
            c.close()
        except Exception as exc:
            problems.append("%s: parse error %s" % (rel, exc))
        if c.stack:
            problems.append("%s: unclosed at EOF: %s" % (rel, ", ".join(c.stack)))
        if c.h1 != 1:
            problems.append("%s: h1 count = %d" % (rel, c.h1))
        if not c.lang:
            problems.append("%s: missing <html lang>" % rel)
        if c.imgs_no_alt:
            problems.append("%s: %d <img> without alt" % (rel, c.imgs_no_alt))

        # template leftovers (skip legacy pages: they contain original LaTeX/markup)
        if not re.match(r"(project\d+|_build/out/project\d+)\.html$", rel):
            scan_text = re.sub(r"(?s)<(?:pre|code)\b[^>]*>.*?</(?:pre|code)>", "", text)
            for pat, label in ((r"%\([a-z_]+\)s", "py format-leftover"),
                               (r"\{[a-z_]+\}", "brace-leftover"),
                               (r"\bNone\b", "None"),
                               (r"\bundefined\b", "undefined"),
                               (r"\{\{|\{%", "liquid")):
                for m in re.finditer(pat, scan_text):
                    ctx = scan_text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                    problems.append("%s: %s -> …%s…" % (rel, label, ctx))
                    break

        for ref, tag in c.refs:
            if ref.startswith(("http://", "https://", "mailto:", "tel:", "#", "data:", "//")):
                continue
            internal_refs += 1
            if not resolve(ref):
                broken += 1
                problems.append("%s: broken %s -> %s" % (rel, tag, ref))

    # CSS sanity
    css = os.path.join(ROOT, "assets", "site", "site.css")
    if os.path.isfile(css):
        with open(css, encoding="utf-8") as fh:
            ctext = fh.read()
        if ctext.count("{") != ctext.count("}"):
            problems.append("site.css: unbalanced braces (%d/%d)"
                            % (ctext.count("{"), ctext.count("}")))
        for sel in [".site-header", ".hero", ".card", ".repo", ".prose", ".post-item",
                    ".timeline", ".stats", ".chip", ".tag", ".btn"]:
            if sel not in ctext:
                problems.append("site.css: missing selector %s" % sel)
        notes.append("css rules: %d bytes" % len(ctext))

    notes.append("internal refs checked: %d (broken: %d)" % (internal_refs, broken))

    print("\n".join(notes))
    if problems:
        print("\nPROBLEMS (%d):" % len(problems))
        for p in problems[:120]:
            print("  " + p)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
