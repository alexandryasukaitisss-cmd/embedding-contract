import json
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from embedding_contract.errors import EmbeddingError
from embedding_contract.providers.gemini import GeminiProvider
from embedding_contract.providers.ollama import OllamaProvider


class _Target(BaseHTTPRequestHandler):
    hits = 0

    def log_message(self, *_args):
        pass

    def do_GET(self):
        type(self).hits += 1
        self._json({"models": [{"name": "embeddinggemma-2:270m", "digest": "sha256:" + "a" * 64}]})

    def do_POST(self):
        type(self).hits += 1
        self._json({"model": "embeddinggemma-2:270m", "embeddings": [[1, 1, 1]]})

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Redirect(BaseHTTPRequestHandler):
    location = ""
    hits = 0

    def log_message(self, *_args):
        pass

    def do_GET(self):
        type(self).hits += 1
        self.send_response(302)
        self.send_header("location", self.location)
        self.end_headers()

    def do_POST(self):
        type(self).hits += 1
        self.send_response(307)
        self.send_header("location", self.location)
        self.end_headers()


@contextmanager
def servers():
    target = ThreadingHTTPServer(("127.0.0.1", 0), type("Target", (_Target,), {"hits": 0}))
    redirect = ThreadingHTTPServer(("127.0.0.1", 0), type("Redirect", (_Redirect,), {"hits": 0, "location": f"http://127.0.0.1:{target.server_port}/v1beta"}))
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (target, redirect)]
    for thread in threads:
        thread.start()
    try:
        yield target, redirect
    finally:
        for server in (target, redirect):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


class RedirectTests(unittest.TestCase):
    def test_ollama_does_not_follow_a_redirect(self):
        with servers() as (target, redirect):
            provider = OllamaProvider(
                model="embeddinggemma-2:270m",
                dimensions=3,
                base_url=f"http://127.0.0.1:{redirect.server_port}",
            )
            with self.assertRaises(EmbeddingError) as context:
                provider.health()
            self.assertEqual(context.exception.code, "provider_http_error")
            self.assertEqual(target.RequestHandlerClass.hits, 0)

    def test_gemini_does_not_follow_a_redirect_with_a_key(self):
        with servers() as (target, redirect):
            provider = GeminiProvider(
                model="gemini-embedding-2",
                dimensions=3,
                base_url=f"http://127.0.0.1:{redirect.server_port}/v1beta",
                api_key="test-key",
            )
            with self.assertRaises(EmbeddingError) as context:
                provider.health()
            self.assertEqual(context.exception.code, "provider_http_error")
            self.assertEqual(target.RequestHandlerClass.hits, 0)


if __name__ == "__main__":
    unittest.main()
