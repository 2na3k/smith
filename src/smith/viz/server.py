import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import program_detail, snapshot

PAGE = Path(__file__).with_name("index.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status=200):
        self._send(json.dumps(data, default=str).encode(), "application/json", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(snapshot())
        elif path.startswith("/api/program/"):
            try:
                detail = program_detail(int(path.rsplit("/", 1)[1]))
            except ValueError:
                detail = None
            if detail is None:
                self._json({"error": "unknown program"}, 404)
            else:
                self._json(detail)
        else:
            self._json({"error": "not found"}, 404)


def make_server(port=8000, tries=20):
    last = None
    for offset in range(tries if port else 1):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port + offset), Handler)
        except OSError as error:
            last = error
    raise last


def serve(port=None, open_browser=True, block=True):
    port = int(os.environ.get("SMITH_VIZ_PORT", "8000")) if port is None else port
    server = make_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"smith viz: {url}  (Ctrl-C to stop)", flush=True)
    if open_browser and not os.environ.get("SMITH_VIZ_NO_BROWSER"):
        webbrowser.open(url)
    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server
