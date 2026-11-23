# -*- coding: utf-8 -*-
"""Create the PDF release and upload the archived files.

Token comes from the GH_TOKEN environment variable (never from argv).

    python _build/tools/upload_release.py [--dry-run]
"""

import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from gen import releases  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
PDF_DIR = os.path.join(ROOT, "assets", "pdf")
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
TOKEN = os.environ.get("GH_TOKEN", "")


def call(method, url, data=None, ctype="application/json", accept="application/vnd.github+json", timeout=900):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", accept)
    req.add_header("User-Agent", "ldy-site-release")
    if data is not None:
        req.add_header("Content-Type", ctype)
        req.add_header("Content-Length", str(len(data)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, (json.loads(body) if body.strip().startswith(("{", "[")) else body)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:600]
    except Exception as exc:                      # noqa: BLE001
        return 0, str(exc)


def ensure_release():
    status, body = call("GET", "%s/repos/%s/releases/tags/%s" % (API, releases.REPO, releases.RELEASE_TAG))
    if status == 200:
        print("release exists: id=%s" % body["id"])
        return body["id"], {a["name"]: a["id"] for a in body.get("assets", [])}
    payload = json.dumps({
        "tag_name": releases.RELEASE_TAG,
        "name": releases.RELEASE_TITLE,
        "body": releases.RELEASE_BODY,
        "draft": False,
        "prerelease": False,
    }).encode("utf-8")
    status, body = call("POST", "%s/repos/%s/releases" % (API, releases.REPO), payload)
    if status not in (200, 201):
        print("FAILED to create release: %s %s" % (status, body))
        return None, {}
    print("release created: id=%s tag=%s" % (body["id"], body["tag_name"]))
    return body["id"], {}


def main():
    dry = "--dry-run" in sys.argv

    missing = [f for f in releases.ASSETS if not os.path.isfile(os.path.join(PDF_DIR, f))]
    if missing:
        print("WARNING: source files missing:", ", ".join(missing))

    if dry:
        for f, name in sorted(releases.ASSETS.items()):
            p = os.path.join(PDF_DIR, f)
            print("  %-28s -> %-24s %s" % (f, name, "%.1f MB" % (os.path.getsize(p) / 1048576) if os.path.exists(p) else "MISSING"))
        return 0

    rid, existing = ensure_release()
    if not rid:
        return 1
    ok, skipped, failed = 0, 0, 0
    for original, asset in sorted(releases.ASSETS.items()):
        path = os.path.join(PDF_DIR, original)
        if not os.path.isfile(path):
            failed += 1
            print("  MISSING  %s" % original)
            continue
        if asset in existing:
            skipped += 1
            print("  exists   %s" % asset)
            continue
        with open(path, "rb") as fh:
            blob = fh.read()
        url = "%s/repos/%s/releases/%s/assets?name=%s" % (UPLOADS, releases.REPO, rid, asset)
        status, body = call("POST", url, blob, ctype="application/pdf")
        if status in (200, 201):
            ok += 1
            print("  uploaded %-24s %.1f MB" % (asset, len(blob) / 1048576))
        else:
            failed += 1
            print("  FAILED   %-24s %s %s" % (asset, status, body))

    print("\nsummary: uploaded=%d already-there=%d failed=%d" % (ok, skipped, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
