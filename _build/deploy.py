# -*- coding: utf-8 -*-
"""Promote staged files from _build/out/ into the repository root.

Only needed in environments where some pre-existing repository files are not
writable by the build process. On a normal machine `python _build/build.py`
already writes everything in place and this script is a no-op.
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGE = os.path.join(HERE, "out")


def main():
    if not os.path.isdir(STAGE):
        print("nothing staged in _build/out — nothing to do")
        return 0
    moved = 0
    failed = []
    for dirpath, _dirnames, filenames in os.walk(STAGE):
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, STAGE)
            dst = os.path.join(ROOT, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            try:
                shutil.copy2(src, dst)
                moved += 1
            except PermissionError as exc:
                failed.append((rel, str(exc)))
    print("promoted %d file(s) into the repository root" % moved)
    if failed:
        print("\nstill not writable (%d) — copy these manually or re-run with enough rights:" % len(failed))
        for rel, msg in failed:
            print("  " + rel)
        return 1
    shutil.rmtree(STAGE, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
