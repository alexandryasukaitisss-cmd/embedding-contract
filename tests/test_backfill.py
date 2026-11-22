import unittest

from embedding_contract.backfill import backfill
from embedding_contract.errors import EmbeddingError
from embedding_contract.identity import EmbeddingIdentity
from embedding_contract.providers.base import EmbeddingBatch, ProviderHealth
from embedding_contract.vectors import l2_normalize


class FakeProvider:
    def __init__(self):
        self.identity = EmbeddingIdentity("fake", "fake-3d", 3, {"document_prefix": "d:"})
        self.batch_size = 2
        self.calls = []

    def health(self):
        return ProviderHealth(True, self.identity, {})

    def embed(self, texts, *, purpose="document"):
        self.calls.append((list(texts), purpose))
        return EmbeddingBatch(
            self.identity,
            [l2_normalize([float(len(text)), 1.0, 1.0], 3) for text in texts],
            {},
        )


class ChangingIdentityProvider(FakeProvider):
    def embed(self, texts, *, purpose="document"):
        same = self.identity
        changed = EmbeddingIdentity("fake", "other-3d", 3, {"document_prefix": "d:"})
        return EmbeddingBatch(changed, [l2_normalize([1.0, 1.0, 1.0], 3) for _ in texts], {})


class BadWidthProvider(FakeProvider):
    def embed(self, texts, *, purpose="document"):
        return EmbeddingBatch(self.identity, [[1.0, 1.0] for _ in texts], {})


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.identity = self.provider.identity

    def test_existing_matching_vector_is_skipped(self):
        vector = l2_normalize([1.0, 2.0, 3.0], 3)
        records = [{"text": "same", "embedding": vector, "embedding_metadata": self.identity.metadata()}]
        summary = backfill(records, self.provider)
        self.assertEqual(summary.skipped_existing, 1)
        self.assertEqual(summary.embedded, 0)
        self.assertEqual(self.provider.calls, [])

    def test_missing_invalid_and_stale_vectors_are_replaced(self):
        valid = l2_normalize([1.0, 2.0, 3.0], 3)
        stale_metadata = dict(self.identity.metadata())
        stale_metadata["dimensions"] = 4
        records = [
            {"text": "missing"},
            {"text": "invalid", "embedding": [1.0]},
            {"text": "stale", "embedding": valid, "embedding_metadata": stale_metadata},
            {"text": ""},
        ]
        summary = backfill(records, self.provider)
        self.assertEqual(summary.total, 4)
        self.assertEqual(summary.embedded, 3)
        self.assertEqual(summary.replaced_invalid, 1)
        self.assertEqual(summary.replaced_stale, 1)
        self.assertEqual(summary.skipped_empty, 1)
        for record in records[:3]:
            self.assertEqual(record["embedding_metadata"]["identity_digest"], self.identity.digest)
            self.assertEqual(len(record["embedding"]), 3)

    def test_force_reembeds_valid_records(self):
        vector = l2_normalize([1.0, 2.0, 3.0], 3)
        records = [{"text": "same", "embedding": vector, "embedding_metadata": self.identity.metadata()}]
        summary = backfill(records, self.provider, force=True)
        self.assertEqual(summary.embedded, 1)
        self.assertEqual(summary.skipped_existing, 0)

    def test_custom_getters_and_setters(self):
        records = [{"body": "custom", "old": None, "meta": None}]
        summary = backfill(
            records,
            self.provider,
            get_text=lambda record, index: record["body"],
            get_embedding=lambda record, index: record["old"],
            get_metadata=lambda record, index: record["meta"],
            set_embedding=lambda record, vector, index: record.__setitem__("old", vector),
            set_metadata=lambda record, metadata, index: record.__setitem__("meta", metadata),
        )
        self.assertEqual(summary.embedded, 1)
        self.assertIsNotNone(records[0]["old"])
        self.assertEqual(records[0]["meta"]["model"], "fake-3d")

    def test_provider_vector_width_is_validated_before_mutation(self):
        records = [{"text": "bad width"}]
        with self.assertRaises(EmbeddingError):
            backfill(records, BadWidthProvider())
        self.assertNotIn("embedding", records[0])

    def test_empty_text_cannot_leave_a_stale_existing_vector(self):
        records = [{"text": "", "embedding": [1.0], "embedding_metadata": {"dimensions": 1}}]
        with self.assertRaises(EmbeddingError) as context:
            backfill(records, self.provider)
        self.assertEqual(context.exception.code, "empty_replacement_text")
        self.assertEqual(records[0]["embedding"], [1.0])

    def test_empty_text_without_a_vector_is_allowed(self):
        records = [{"text": ""}]
        summary = backfill(records, self.provider)
        self.assertEqual(summary.skipped_empty, 1)
        self.assertNotIn("embedding", records[0])

    def test_identity_change_during_a_batch_does_not_mutate_records(self):
        records = [{"text": "x"}]
        with self.assertRaises(EmbeddingError):
            backfill(records, ChangingIdentityProvider())
        self.assertNotIn("embedding", records[0])

    def test_atomic_record_setter_is_supported(self):
        records = [{"text": "atomic"}]
        seen = []

        def set_record(record, vector, metadata, index):
            seen.append((len(vector), metadata["identity_digest"]))
            record["vector"] = vector
            record["meta"] = metadata

        summary = backfill(records, self.provider, set_record=set_record)
        self.assertEqual(summary.embedded, 1)
        self.assertEqual(seen[0][0], 3)
        self.assertEqual(records[0]["meta"]["identity_digest"], self.identity.digest)

    def test_batch_size_is_bounded(self):
        with self.assertRaises(Exception):
            backfill([{"text": "x"}], self.provider, batch_size=3)


if __name__ == "__main__":
    unittest.main()
