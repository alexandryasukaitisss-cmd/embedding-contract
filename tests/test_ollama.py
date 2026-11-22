import json
import os
import threading
import unittest
from contextlib import contextmanager
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from embedding_contract.errors import EmbeddingError
from embedding_contract.providers.ollama import OllamaProvider

DIGEST = "a" * 64
MODEL = "embeddinggemma-2:270m"


class _Handler(BaseHTTPRequestHandler):
    model = MODEL
    digest = DIGEST
    dimensions = 3
    requests = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/api/tags":
            self.send_error(404)
            return
        self._json({"models": [{"name": self.model, "digest": "sha256:" + self.digest}]})

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append((self.path, body))
        if self.path != "/api/embed":
            self.send_error(404)
            return
        inputs = body.get("input")
        if not isinstance(inputs, list):
            self.send_error(400)
            return
        vectors = [[float(index + 1), 1.0, 1.0][: self.dimensions] for index, _ in enumerate(inputs)]
        self._json({"model": self.model, "embeddings": vectors})

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def fake_ollama(**overrides):
    handler = type("Handler", (_Handler,), {**overrides, "requests": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield handler, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class OllamaTests(unittest.TestCase):
    def test_health_and_identity(self):
        with fake_ollama() as (handler, base):
            provider = OllamaProvider(model=MODEL, dimensions=3, base_url=base)
            health = provider.health()
            self.assertTrue(health.ok)
            self.assertEqual(health.identity.model, MODEL)
            self.assertEqual(health.identity.dimensions, 3)
            self.assertEqual(health.identity.artifact_digest, DIGEST)
            self.assertEqual(health.identity.provider, "ollama")

    def test_embed_applies_purpose_prefix_and_normalizes(self):
        with fake_ollama() as (handler, base):
            provider = OllamaProvider(
                model=MODEL,
                dimensions=3,
                base_url=base,
                prompt_contract={"query_prefix": "Q:", "document_prefix": "D:"},
            )
            batch = provider.embed(["hello"], purpose="query")
            self.assertEqual(handler.requests[-1][1]["input"], ["Q:hello"])
            self.assertEqual(len(batch.vectors[0]), 3)
            self.assertAlmostEqual(sum(value * value for value in batch.vectors[0]), 1.0)
            self.assertEqual(batch.identity.digest, provider.health().identity.digest)

    def test_digest_pin_and_model_identity_fail_closed(self):
        with fake_ollama() as (handler, base):
            with self.assertRaises(EmbeddingError) as context:
                OllamaProvider(model=MODEL, dimensions=3, base_url=base, expected_digest="b" * 64).health()
            self.assertEqual(context.exception.code, "model_digest_mismatch")
            with self.assertRaises(EmbeddingError) as context:
                OllamaProvider(model="other:model", dimensions=3, base_url=base).health()
            self.assertEqual(context.exception.code, "model_not_found")

    def test_remote_urls_are_rejected_by_default(self):
        for url in ("http://localhost:11434", "https://ollama.example:443", "http://192.0.2.1:11434"):
            with self.assertRaises(EmbeddingError) as context:
                OllamaProvider(model=MODEL, dimensions=3, base_url=url)
            self.assertEqual(context.exception.code, "remote_endpoint_forbidden")

    def test_environment_proxy_is_not_used_for_local_ollama(self):
        with fake_ollama() as (_, base):
            with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1"}):
                provider = OllamaProvider(model=MODEL, dimensions=3, base_url=base)
                self.assertTrue(provider.health().ok)

    def test_blob_digest_is_reinspected_before_each_embedding(self):
        with fake_ollama() as (handler, base):
            provider = OllamaProvider(model=MODEL, dimensions=3, base_url=base)
            first = provider.health().identity
            handler.digest = "b" * 64
            second = provider.embed(["hello"], purpose="query").identity
            self.assertNotEqual(first.digest, second.digest)
            self.assertEqual(second.artifact_digest, "b" * 64)

    def test_cloud_tag_is_rejected(self):
        with self.assertRaises(EmbeddingError):
            OllamaProvider(model="qwen3-embedding:0.6b-cloud", dimensions=3, base_url="http://127.0.0.1:11434")

    def test_remote_https_is_explicit_and_keeps_default_port(self):
        provider = OllamaProvider(model=MODEL, dimensions=3, base_url="https://ollama.example", allow_remote=True)
        self.assertEqual(provider.base_url, "https://ollama.example:443")

    def test_bad_dimensions_and_batches_fail_closed(self):
        with fake_ollama() as (handler, base):
            provider = OllamaProvider(model=MODEL, dimensions=4, base_url=base)
            with self.assertRaises(EmbeddingError) as context:
                provider.health()
            self.assertEqual(context.exception.code, "dimension_mismatch")
            with self.assertRaises(EmbeddingError):
                provider.embed(["a", "b", "c"], purpose="query")


if __name__ == "__main__":
    unittest.main()
