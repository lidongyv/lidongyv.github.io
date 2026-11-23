# -*- coding: utf-8 -*-
"""Convert the legacy project screenshots to WebP.

The eight legacy project pages reference 120+ PNG screenshots (mostly paper
figures and result tables). They are the last meaningful chunk of the published
site: ~52 MB of ~62 MB.

The originals are never lost: they stay in git history and are pinned by the
`images-png-originals` tag, and this script only writes next to them unless
`--prune` is given.

    python _build/tools/optimize_images.py --dry-run
    python _build/tools/optimize_images.py
    python _build/tools/optimize_images.py --prune
"""

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMG = os.path.join(ROOT, "assets", "img")

# Screenshots and paper figures: text must stay crisp, so we start at a high
# quality and only fall back to lossless when the lossy file is not much smaller.
LOSSY_Q = 90
METHOD = 6


def human(n):
    return "%.2f MB" % (n / 1048576.0) if n >= 1048576 else "%.0f KB" % (n / 1024.0)


def encode(path, img_modulus):
    """Return (bytes, kind) for the best encoding of one image."""
    from PIL import Image
    im = Image.open(path)
    lossy = io.BytesIO()
    im.save(lossy, "WEBP", quality=LOSSY_Q, method=METHOD)
    best, kind = len(lossy.getvalue()), "lossy q%d" % LOSSY_Q
    if im.mode in ("RGBA", "RGB", "L", "P"):
        lossless = io.BytesIO()
        try:
            im.save(lossless, "WEBP", lossless=True, method=METHOD)
            if len(lossless.getvalue()) < best * 1.6:
                best, kind = len(lossless.getvalue()), "lossless"
        except Exception:
            pass
    return best, kind


def main():
    dry = "--dry-run" in sys.argv
    prune = "--prune" in sys.argv
    if not os.path.isdir(IMG):
        print("no assets/img")
        return 1

    rows = []
    total_before = total_after = 0
    for dirpath, _dirs, files in os.walk(IMG):
        for fn in sorted(files):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            src = os.path.join(dirpath, fn)
            before = os.path.getsize(src)
            rel = os.path.relpath(src, ROOT).replace("\\", "/")
            try:
                after, kind = encode(src, None)
            except Exception as exc:                      # noqa: BLE001
                print("  SKIP %-46s %s" % (rel, exc))
                continue
            total_before += before
            total_after += after
            rows.append((rel, before, after, kind))
            if not dry:
                from PIL import Image
                dst = os.path.splitext(src)[0] + ".webp"
                Image.open(src).save(dst, "WEBP", quality=LOSSY_Q, method=METHOD,
                                     lossless=(kind == "lossless"))

    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    print("%-52s %10s %10s %6s  %s" % ("file", "before", "after", "ratio", "mode"))
    for rel, b, a, kind in rows[:18]:
        print("%-52s %10s %10s %5.1f%%  %s" % (rel, human(b), human(a), 100.0 * a / b, kind))
    if len(rows) > 18:
        print("  … %d more" % (len(rows) - 18))
    saved = total_before - total_after
    print("\nimages: %d   before: %s   after: %s   saved: %s (%.1f%%)"
          % (len(rows), human(total_before), human(total_after), human(saved),
             100.0 * saved / total_before if total_before else 0))

    if dry:
        print("\ndry run — nothing written")
        return 0

    if prune:
        removed = 0
        for dirpath, _dirs, files in os.walk(IMG):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    webp = os.path.splitext(os.path.join(dirpath, fn))[0] + ".webp"
                    if os.path.exists(webp):
                        os.remove(os.path.join(dirpath, fn))
                        removed += 1
        print("pruned %d original file(s) (recoverable from git)" % removed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
