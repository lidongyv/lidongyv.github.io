# -*- coding: utf-8 -*-
"""Rebuild the site from its local sources and publish it.

    python _build/publish.py              rebuild, check, commit, push, wait, verify
    python _build/publish.py -m "msg"     same, with your own commit message
    python _build/publish.py --no-push    rebuild and check only
    python _build/publish.py --dry-run    show what would be published

The blog subsites are *copies* of the design/demos source directories, so any edit
outside this repo needs this run before it appears online.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = "lidongyv/lidongyv.github.io"
SITE = "https://lidongyv.github.io"
PY = sys.executable


def sh(args, cwd=ROOT, check=False):
    r = subprocess.run(args, cwd=cwd, capture_output=True)
    out = (r.stdout or b"").decode("utf-8", "replace")
    err = (r.stderr or b"").decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise SystemExit("failed: %s\n%s%s" % (" ".join(args), out, err))
    return r.returncode, out, err


def step(title):
    print("\n== %s ==" % title)


def token():
    try:
        r = subprocess.run(["git", "credential", "fill"], cwd=ROOT, capture_output=True,
                           input=b"protocol=https\nhost=github.com\n\n")
        for line in (r.stdout or b"").decode().splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1].strip()
    except Exception:                                   # noqa: BLE001
        pass
    return os.environ.get("GH_TOKEN", "")


def api(path):
    t = token()
    if not t:
        return None
    req = urllib.request.Request("https://api.github.com" + path)
    req.add_header("Authorization", "Bearer " + t)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "dsh-publish")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    except Exception:                                   # noqa: BLE001
        return None


def build():
    step("rebuild from source")
    rc, out, err = sh([PY, os.path.join(HERE, "build.py")])
    tail = [l for l in out.splitlines() if l.strip()][-6:]
    print("\n".join(tail))
    if rc != 0:
        print(out, err)
        raise SystemExit("build failed")

    step("check the output")
    rc, out, err = sh([PY, os.path.join(HERE, "check.py")])
    print(out.strip().split("\n")[-1] if out.strip() else "")
    if rc != 0:
        print(out[:2000])
        raise SystemExit("check failed")


def changed_files():
    rc, out, _ = sh(["git", "status", "--porcelain"])
    return [l for l in out.splitlines() if l.strip()]


def publish(message):
    files = changed_files()
    if not files:
        print("\nnothing changed — the site already matches the source")
        return False
    print("\n%d path(s) changed" % len(files))

    step("commit")
    sh(["git", "add", "-A"], check=True)
    rc, out, err = sh(["git", "-c", "user.name=Lidong Yu", "-c", "user.email=yvlidong@gmail.com",
                       "commit", "-q", "-m", message])
    if rc != 0:
        print(out, err)
        raise SystemExit("commit failed")
    print("committed: " + message.splitlines()[0])

    step("push")
    for i in range(4):
        rc, out, err = sh(["git", "-c", "http.proxy=", "-c", "https.proxy=",
                           "push", "origin", "master"])
        if rc == 0:
            print("pushed via git")
            return True
        print("  attempt %d failed, retrying" % (i + 1))
        time.sleep(6)

    print("git push blocked; falling back to the GitHub API")
    env = dict(os.environ, GH_TOKEN=token())
    r = subprocess.run([PY, os.path.join(HERE, "tools", "api_push.py"), "--apply"],
                       cwd=ROOT, capture_output=True, env=env)
    print((r.stdout or b"").decode("utf-8", "replace").splitlines()[-1:])
    if r.returncode != 0:
        raise SystemExit("push failed")
    return True


def wait_for_deploy():
    """Wait for the run that belongs to *this* commit, not the previous one."""
    step("wait for GitHub Pages")
    _, out, _ = sh(["git", "rev-parse", "HEAD"])
    want = out.strip()[:7]
    print("  watching for %s" % want)
    for i in range(60):
        runs = api("/repos/%s/actions/runs?per_page=3" % REPO)
        if not runs or not runs.get("workflow_runs"):
            print("  no token or no runs — check https://github.com/%s/actions" % REPO)
            return
        run = next((r for r in runs["workflow_runs"]
                    if r["head_sha"][:7] == want), None)
        if run is None:
            time.sleep(8)
            continue
        status, conclusion = run["status"], run["conclusion"]
        if status == "completed":
            print("  %s %s %s" % (run["head_sha"][:7], status, conclusion))
            if conclusion != "success":
                raise SystemExit("deployment failed: %s" % conclusion)
            return
        if i % 4 == 0:
            print("  %s %s" % (run["head_sha"][:7], status))
        time.sleep(10)
    raise SystemExit("deployment did not finish in time — check the Actions tab")


def verify(paths):
    step("verify")
    for p in paths:
        url = SITE + p + ("&" if "?" in p else "?") + "v=%d" % int(time.time())
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 dsh-publish")
        req.add_header("Cache-Control", "no-cache")
        try:
            body = urllib.request.urlopen(req, timeout=45).read()
            print("  %-34s 200  %d B" % (p, len(body)))
        except Exception as exc:                        # noqa: BLE001
            print("  %-34s FAILED (%s)" % (p, exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", default="/,/blog/,/experience.html,/en/")
    args = ap.parse_args()

    build()
    files = changed_files()
    if args.dry_run:
        print("\n%d path(s) would be committed:" % len(files))
        for f in files[:25]:
            print("  " + f)
        return

    if not files:
        print("\nnothing changed — the site already matches the source")
        return

    if args.no_push:
        print("\n%d path(s) changed; not committing (--no-push)" % len(files))
        return

    msg = args.message or ("content: re-pull from source\n\n"
                           "%d path(s) rebuilt by _build/publish.py" % len(files))
    if publish(msg):
        wait_for_deploy()
        verify([p for p in args.verify.split(",") if p])
    print("\ndone. Pages caches html for 10 minutes; hard-refresh if you still see the old page.")


if __name__ == "__main__":
    main()
