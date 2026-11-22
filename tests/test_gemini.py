import json
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from embedding_contract.errors import EmbeddingError
from embedding_contract.providers.gemini import GeminiProvider

DIMENSIONS = 3
KEY = "test-key-not-real"


class _Handler(BaseHTTPRequestHandler):
    requests = []
    reject = False

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append((self.path, self.headers.get("x-goog-api-key"), body))
        if self.reject:
            return self._json({"error": {"message": f"api_key={KEY}"}}, status=400)
        if self.path.endswith(":embedContent"):
            return self._json({"embeddings": [{"values": [1.0, 1.0, 1.0]}]})
        if self.path.endswith(":batchEmbedContents"):
            count = len(body.get("requests", []))
            return self._json({"embeddings": [{"values": [1.0, 1.0, 1.0]} for _ in range(count)]})
        self.send_error(404)

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def fake_gemini(**overrides):
    handler = type("Handler", (_Handler,), {**overrides, "requests": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield handler, f"http://127.0.0.1:{server.server_port}/v1beta"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class GeminiTests(unittest.TestCase):
    def test_health_and_single_embedding(self):
        with fake_gemini() as (handler, base):
            provider = GeminiProvider(model="gemini-embedding-2", dimensions=DIMENSIONS, base_url=base, api_key=KEY)
            health = provider.health()
            self.assertTrue(health.ok)
            self.assertEqual(health.identity.model, "gemini-embedding-2")
            batch = provider.embed(["hello"], purpose="query")
            self.assertEqual(len(batch.vectors[0]), DIMENSIONS)
            path, key, body = handler.requests[-1]
            self.assertEqual(key, KEY)
            self.assertEqual(body["content"]["parts"][0]["text"], "task: search result | query: hello")
            self.assertEqual(body["embedContentConfig"]["outputDimensionality"], DIMENSIONS)

    def test_batch_uses_batch_endpoint(self):
        with fake_gemini() as (handler, base):
            provider = GeminiProvider(model="gemini-embedding-2", dimensions=DIMENSIONS, base_url=base, api_key=KEY)
            batch = provider.embed(["a", "b"], purpose="document")
            self.assertEqual(len(batch.vectors), 2)
            path, _, body = handler.requests[-1]
            self.assertTrue(path.endswith(":batchEmbedContents"))
            self.assertEqual(len(body["requests"]), 2)
            self.assertEqual(body["requests"][0]["content"]["parts"][0]["text"], "title: none | text: a")

    def test_embedding_001_uses_task_types(self):
        with fake_gemini() as (handler, base):
            provider = GeminiProvider(model="gemini-embedding-001", dimensions=DIMENSIONS, base_url=base, api_key=KEY)
            provider.embed(["q"], purpose="query")
            _, _, body = handler.requests[-1]
            self.assertEqual(body["embedContentConfig"]["taskType"], "RETRIEVAL_QUERY")
            provider.embed(["d"], purpose="document")
            _, _, body = handler.requests[-1]
            self.assertEqual(body["embedContentConfig"]["taskType"], "RETRIEVAL_DOCUMENT")

    def test_legacy_task_pair_is_immutable_and_matches_the_request(self):
        with fake_gemini() as (handler, base):
            first = GeminiProvider(model="gemini-embedding-001", dimensions=DIMENSIONS, base_url=base, api_key=KEY, task_type="CLASSIFICATION")
            second = GeminiProvider(model="gemini-embedding-001", dimensions=DIMENSIONS, base_url=base, api_key=KEY, task_type="CLUSTERING")
            self.assertNotEqual(first.identity.digest, second.identity.digest)
            first.embed(["x"], purpose="document")
            self.assertEqual(handler.requests[-1][2]["embedContentConfig"]["taskType"], "CLASSIFICATION")
            with self.assertRaises(TypeError):
                first.prompt_contract["document_task"] = "CLUSTERING"  # type: ignore[index]

    def test_missing_key_and_dimension_mismatch_fail_closed(self):
        with fake_gemini() as (_, base):
            provider = GeminiProvider(model="gemini-embedding-2", dimensions=DIMENSIONS, base_url=base, api_key="")
            with self.assertRaises(EmbeddingError) as context:
                provider.health()
            self.assertEqual(context.exception.code, "auth_missing")
            provider = GeminiProvider(model="gemini-embedding-2", dimensions=4, base_url=base, api_key=KEY)
            with self.assertRaises(EmbeddingError) as context:
                provider.health()
            self.assertEqual(context.exception.code, "dimension_mismatch")

    def test_http_error_does_not_echo_the_key(self):
        with fake_gemini(reject=True) as (_, base):
            provider = GeminiProvider(model="gemini-embedding-2", dimensions=DIMENSIONS, base_url=base, api_key=KEY)
            with self.assertRaises(EmbeddingError) as context:
                provider.health()
            self.assertNotIn(KEY, str(context.exception))
            self.assertEqual(context.exception.code, "provider_http_error")

    def test_custom_https_host_is_rejected(self):
        with self.assertRaises(EmbeddingError):
            GeminiProvider(model="gemini-embedding-2", dimensions=3, base_url="https://example.invalid/v1beta", api_key=KEY)

    def test_key_with_whitespace_is_rejected(self):
        with self.assertRaises(EmbeddingError):
            GeminiProvider(model="gemini-embedding-2", dimensions=3, api_key="bad\nkey")

    def test_repr_hides_the_key(self):
        provider = GeminiProvider(model="gemini-embedding-2", dimensions=3, api_key=KEY)
        self.assertNotIn(KEY, repr(provider))


if __name__ == "__main__":
    unittest.main()
