# -*- coding: utf-8 -*-
"""Where the archived PDFs live.

The twelve paper / report PDFs used to sit in `assets/pdf/` (132 MB of a 193 MB
site, and ~120 MB of them were single files big enough to burn through the
100 GB/month Pages bandwidth allowance in a few thousand downloads). They now
ship as GitHub Release assets, which do not count against the Pages quota.

Release: https://github.com/lidongyv/lidongyv.github.io/releases/tag/papers-2026-11
"""

REPO = "lidongyv/lidongyv.github.io"
RELEASE_TAG = "papers-2026-11"
RELEASE_TITLE = "Papers & technical reports"
RELEASE_BASE = "https://github.com/%s/releases/download/%s/" % (REPO, RELEASE_TAG)

# --- the switch -----------------------------------------------------------
# The twelve PDFs are ALREADY uploaded to the release above (12 assets, 133 MB,
# state=uploaded). But this repository is private, and GitHub does not serve
# release assets publicly from a private repository: the
# releases/download/<tag>/<file> URL answers 404 for anonymous visitors, even
# with a valid token (the web endpoint cannot authenticate with a Bearer header).
#
# So while the repo stays private the site must keep serving the local copies in
# assets/pdf/. Once the repository is public:
#
#   1. set ENABLED = True
#   2. python _build/build.py
#   3. git rm -r assets/pdf
#   4. git add -A && git commit -m "site: serve archived PDFs from the release" && git push
#
# and the published site drops from ~193 MB to ~60 MB.
ENABLED = False
# --------------------------------------------------------------------------

# original filename in assets/pdf/  ->  release asset name
ASSETS = {
    "aaai18.pdf": "aaai18.pdf",
    "cvpr19.pdf": "cvpr19.pdf",
    "eccv2020.pdf": "eccv2020.pdf",
    "iccv2019.pdf": "iccv2019.pdf",
    "interpreatble cvpr18.pdf": "interpretable-cvpr18.pdf",
    "interpretable eccv19.pdf": "interpretable-eccv19.pdf",
    "interpretable tpami.pdf": "interpretable-tpami.pdf",
    "mot.pdf": "mot.pdf",
    "nips-cvpr19.pdf": "nips-cvpr19.pdf",
    "nips.pdf": "nips.pdf",
    "sot.pdf": "sot.pdf",
    "vod.pdf": "vod.pdf",
}

RELEASE_BODY = """Archived PDFs that used to live in `assets/pdf/` on the website.

They are here rather than in the repository because they are the biggest files
the site ever served (132 MB of a 193 MB site) and GitHub Pages enforces a
100 GB/month soft bandwidth limit — a single 62 MB report was enough to make a
handful of downloads expensive.

Publication links on the site point at arXiv / CVF Open Access / AAAI instead;
these files are the submission drafts, extended manuscripts and internal
technical reports that have no public venue.
"""


def asset_name(original):
    return ASSETS.get(original)


def url_for(original):
    """Release download URL for a filename that used to live in assets/pdf/."""
    if not ENABLED:
        return None
    name = ASSETS.get(original)
    if not name:
        return None
    return RELEASE_BASE + name
