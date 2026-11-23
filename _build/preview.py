# -*- coding: utf-8 -*-
"""Local preview server with the _build/out overlay applied.

Serves the repository root, but if the same relative path exists under
_build/out it wins (that mirrors what deploy.py would put in place).

    python _build/preview.py 8000
"""

import os
import sys
import http.server
import socketserver
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGE = os.path.join(HERE, "out")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
    ".xml": "application/xml; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ldy-preview/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _resolve(self):
        raw = urllib.parse.urlparse(self.path).path
        rel = urllib.parse.unquote(raw).lstrip("/")
        if rel == "" or rel.endswith("/"):
            rel = rel + "index.html"
        candidates = []
        if os.path.isfile(os.path.join(STAGE, rel.replace("/", os.sep))):
            candidates.append(os.path.join(STAGE, rel.replace("/", os.sep)))
        if os.path.isfile(os.path.join(ROOT, rel.replace("/", os.sep))):
            candidates.append(os.path.join(ROOT, rel.replace("/", os.sep)))
        # directory index fallback
        if not candidates:
            for base in (STAGE, ROOT):
                idx = os.path.join(base, rel.replace("/", os.sep), "index.html")
                if os.path.isfile(idx):
                    candidates.append(idx)
        return candidates[0] if candidates else None

    def do_GET(self):
        path = self._resolve()
        if not path:
            body = (b"<!doctype html><meta charset=utf-8><title>404</title>"
                    b"<h1>404</h1><p>Not found: ")
            body += self.path.encode("utf-8", "replace")
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        ext = os.path.splitext(path)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as fh:
            try:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_HEAD(self):
        self.do_GET()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        print("preview: http://127.0.0.1:%d/  (root=%s, stage overlay=%s)"
              % (port, ROOT, os.path.isdir(STAGE)))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
