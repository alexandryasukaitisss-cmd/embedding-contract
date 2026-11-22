import unittest

from embedding_contract.errors import EmbeddingError
from embedding_contract.identity import EmbeddingIdentity, canonical_json, identity_from_metadata


class IdentityTests(unittest.TestCase):
    def test_digest_is_stable_and_sorted(self):
        first = EmbeddingIdentity("ollama", "embeddinggemma-2:270m", 768, {"document_prefix": "title: none | text: ", "query_prefix": "task: search result | query: "})
        second = EmbeddingIdentity("ollama", "embeddinggemma-2:270m", 768, {"query_prefix": "task: search result | query: ", "document_prefix": "title: none | text: "})
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(canonical_json({"b": 1, "a": 2}), '{"a":2,"b":1}')

    def test_every_space_changing_field_changes_the_digest(self):
        base = EmbeddingIdentity("ollama", "model", 3, {"document_prefix": "d"})
        variants = [
            EmbeddingIdentity("gemini", "model", 3, {"document_prefix": "d"}),
            EmbeddingIdentity("ollama", "other", 3, {"document_prefix": "d"}),
            EmbeddingIdentity("ollama", "model", 4, {"document_prefix": "d"}),
            EmbeddingIdentity("ollama", "model", 3, {"document_prefix": "x"}),
            EmbeddingIdentity("ollama", "model", 3, {"document_prefix": "d"}, task_type="RETRIEVAL_DOCUMENT"),
            EmbeddingIdentity("ollama", "model", 3, {"document_prefix": "d"}, artifact_digest="a" * 64),
        ]
        self.assertEqual(len({base.digest, *[item.digest for item in variants]}), 7)

    def test_metadata_round_trip_and_tamper_detection(self):
        identity = EmbeddingIdentity("gemini", "gemini-embedding-2", 768, {"query_prefix": "q"})
        restored = identity_from_metadata(identity.metadata())
        self.assertEqual(restored, identity)
        tampered = dict(identity.metadata())
        tampered["dimensions"] = 3072
        with self.assertRaises(EmbeddingError):
            identity_from_metadata(tampered)
        for key in ("identity_version", "identity_digest", "provider", "model", "dimensions", "prompt_contract", "task_type", "artifact_digest"):
            broken = dict(identity.metadata())
            broken.pop(key)
            with self.assertRaises(EmbeddingError):
                identity_from_metadata(broken)
        boolean_version = dict(identity.metadata())
        boolean_version["identity_version"] = True
        with self.assertRaises(EmbeddingError):
            identity_from_metadata(boolean_version)
        float_dimensions = dict(identity.metadata())
        float_dimensions["dimensions"] = float(float_dimensions["dimensions"])
        with self.assertRaises(EmbeddingError):
            identity_from_metadata(float_dimensions)

    def test_prompt_contract_is_immutable_and_copied(self):
        source = {"document_prefix": "d:"}
        identity = EmbeddingIdentity("ollama", "m", 3, source)
        source["document_prefix"] = "changed"
        self.assertEqual(identity.prompt_contract["document_prefix"], "d:")
        with self.assertRaises(TypeError):
            identity.prompt_contract["document_prefix"] = "nope"  # type: ignore[index]

    def test_artifact_digest_and_bad_dimensions_are_rejected(self):
        with self.assertRaises(EmbeddingError):
            EmbeddingIdentity("ollama", "m", 0)
        with self.assertRaises(EmbeddingError):
            EmbeddingIdentity("ollama", "m", 3, artifact_digest="not-a-digest")
        with self.assertRaises(EmbeddingError):
            EmbeddingIdentity("", "m", 3)
        with self.assertRaises(EmbeddingError):
            EmbeddingIdentity("ollama", "m", 3, prompt_contract="not-a-mapping")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
