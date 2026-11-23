# -*- coding: utf-8 -*-
"""Push the current HEAD commit through the GitHub Git Data API.

Why this exists: from this network `github.com:443` is unreachable (connect
timeouts / resets) while `api.github.com` works, so `git push` fails while the
REST API still can create the commit.

Two details that matter:

* blob bytes come from git objects (`git cat-file blob`), never from the working
  tree: on Windows the worktree may hold CRLF while the index/commit holds LF, and
  uploading worktree bytes produces different blobs than the commit has.
* GitHub stores commit dates normalised to UTC, while a local commit keeps its
  original offset, which changes the SHA. So the local branch is recreated with
  UTC dates too; after this script both sides have the identical commit object.

    python _build/tools/api_push.py            # dry run
    python _build/tools/api_push.py --apply
"""

import base64
import datetime
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = "lidongyv/lidongyv.github.io"
API = "https://api.github.com"
TOKEN = os.environ.get("GH_TOKEN", "")


def git(*args, env=None):
    out = subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, env=env)
    if out.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), out.stderr.decode("utf-8", "replace")[:400]))
    return out.stdout


def call(method, url, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "ldy-api-push")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:600]


def utc_iso(epoch, offset_seconds):
    """GitHub stores UTC; produce the same instant as an ISO string."""
    return datetime.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    apply = "--apply" in sys.argv
    if not TOKEN:
        print("GH_TOKEN missing")
        return 2

    head = git("rev-parse", "HEAD").decode().strip()
    parent = git("rev-parse", "HEAD^").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    message = git("log", "-1", "--format=%B", head).decode("utf-8").rstrip("\n")
    an = git("log", "-1", "--format=%an", head).decode().strip()
    ae = git("log", "-1", "--format=%ae", head).decode().strip()
    cn = git("log", "-1", "--format=%cn", head).decode().strip()
    ce = git("log", "-1", "--format=%ce", head).decode().strip()
    ad_epoch = int(git("log", "-1", "--format=%at", head).decode().strip())
    cd_epoch = int(git("log", "-1", "--format=%ct", head).decode().strip())
    ad_utc = utc_iso(ad_epoch, 0)
    cd_utc = utc_iso(cd_epoch, 0)

    print("local HEAD : %s" % head)
    print("parent     : %s" % parent)
    print("tree       : %s" % tree)
    print("dates      : author %s / committer %s (UTC)" % (ad_utc, cd_utc))

    status, ref = call("GET", "%s/repos/%s/git/ref/heads/master" % (API, REPO))
    if status != 200:
        print("cannot read remote ref: %s %s" % (status, ref))
        return 1
    remote_sha = ref["object"]["sha"]
    print("remote HEAD: %s" % remote_sha)
    if remote_sha == head:
        print("already up to date")
        return 0
    if remote_sha != parent:
        print("REFUSING: remote (%s) is not the parent of HEAD (%s)" % (remote_sha, parent))
        return 1

    status, rcommit = call("GET", "%s/repos/%s/git/commits/%s" % (API, REPO, remote_sha))
    base_tree = rcommit["tree"]["sha"]

    diff = [l for l in git("diff", "--name-status", "-z", parent, head).decode("utf-8").split("\x00") if l]
    entries = []
    i = 0
    while i < len(diff):
        state = diff[i]
        path = diff[i + 1] if i + 1 < len(diff) else ""
        i += 2 if not state.startswith("R") else 3
        if state.startswith("D"):
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            print("  delete %s" % path)
            continue
        sha = git("rev-parse", "HEAD:%s" % path).decode().strip()
        mode = git("ls-tree", "HEAD", path).decode().split()[0]
        entries.append({"path": path, "mode": mode, "type": "blob", "sha": sha})
        print("  %s %s -> %s" % ("modify" if state.startswith("M") else "add", path, sha[:8]))

    # what the server will produce
    status, rtree_check = call("POST", "%s/repos/%s/git/trees" % (API, REPO),
                               {"base_tree": base_tree, "tree": entries}) if apply else (0, {})
    if apply:
        if status not in (200, 201):
            print("tree failed: %s %s" % (status, rtree_check))
            return 1
        if rtree_check["sha"] != tree:
            print("TREE MISMATCH: server %s != local %s" % (rtree_check["sha"], tree))
            return 1
        print("tree matches local: %s" % tree)

    # upload only blobs the server does not already have
    if apply:
        for e in entries:
            if e["sha"] is None:
                continue
            blob = git("cat-file", "blob", e["sha"])
            st, r = call("POST", "%s/repos/%s/git/blobs" % (API, REPO),
                         {"content": base64.b64encode(blob).decode("ascii"), "encoding": "base64"})
            if st not in (200, 201):
                print("  blob re-upload %s -> %s (already exists is fine)" % (e["path"], st))
            elif r["sha"] != e["sha"]:
                print("  BLOB MISMATCH %s: %s != %s" % (e["path"], r["sha"], e["sha"]))
                return 1

    if not apply:
        print("\ndry run: %d entr(ies)" % len(entries))
        return 0

    status, rnew = call("POST", "%s/repos/%s/git/commits" % (API, REPO), {
        "message": message,
        "tree": tree,
        "parents": [remote_sha],
        "author": {"name": an, "email": ae, "date": ad_utc},
        "committer": {"name": cn, "email": ce, "date": cd_utc},
    })
    if status not in (200, 201):
        print("commit failed: %s %s" % (status, rnew))
        return 1
    server_sha = rnew["sha"]
    print("server commit: %s" % server_sha)

    # recreate locally with UTC dates so both sides carry the same object
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = an
    env["GIT_AUTHOR_EMAIL"] = ae
    env["GIT_AUTHOR_DATE"] = ad_utc
    env["GIT_COMMITTER_NAME"] = cn
    env["GIT_COMMITTER_EMAIL"] = ce
    env["GIT_COMMITTER_DATE"] = cd_utc
    local_utc = git("commit-tree", tree, "-p", parent, "-m", message, env=env).decode().strip()
    print("local  (UTC) : %s" % local_utc)
    if local_utc != server_sha:
        print("WARNING: UTC-normalised local commit still differs; leaving refs alone")
        return 1

    status, rref = call("PATCH", "%s/repos/%s/git/refs/heads/master" % (API, REPO),
                        {"sha": server_sha, "force": False})
    if status not in (200, 201):
        print("ref update failed: %s %s" % (status, rref))
        return 1
    print("master -> %s" % rref["object"]["sha"])

    git("update-ref", "refs/heads/master", local_utc, head)
    git("reset", "--mixed", "-q")
    print("local master repointed to %s (worktree untouched)" % local_utc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
