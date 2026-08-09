#!/usr/bin/env python3
"""Serve a bounded artifact directory over token-authenticated HTTPS."""

from __future__ import annotations

import argparse
import hmac
import http.server
import ssl
from functools import partial
from pathlib import Path


class BearerTokenHandler(http.server.SimpleHTTPRequestHandler):
    """Static-file handler requiring one bearer token for every request."""

    server_version = "LunarisArtifactServer/1"
    protocol_version = "HTTP/1.1"

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.bearer_token}"  # type: ignore[attr-defined]
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, expected)

    def _reject(self) -> None:
        self.send_response(http.HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._reject()
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._authorized():
            self._reject()
            return
        super().do_HEAD()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    if not args.directory.is_dir():
        raise FileNotFoundError(args.directory)
    if len(args.token) < 32:
        raise ValueError("Bearer token must contain at least 32 characters")

    handler = partial(BearerTokenHandler, directory=str(args.directory.resolve()))
    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    server.bearer_token = args.token  # type: ignore[attr-defined]
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.certificate, args.private_key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(
        f"Serving {args.directory.resolve()} on https://{args.bind}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
