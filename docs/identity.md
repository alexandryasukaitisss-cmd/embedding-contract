# Embedding identity

## The problem

A vector is meaningless without the exact contract that produced it. Two
128-dimensional vectors can look compatible while coming from different models,
different prompt prefixes, or different task types. Cosine similarity will still
return a number. That number is noise.

This package treats identity as a first-class value.

## Fields

| Field | Why it changes the space |
| --- | --- |
| `provider` | Ollama and Gemini are different runtimes with different preprocessing. |
| `model` | Different weights, different space. |
| `dimensions` | Truncated Matryoshka outputs are not interchangeable with full outputs. |
| `prompt_contract` | Query/document prefixes and Gemini task pairs change the embedding. |
| `artifact_digest` | A local model tag can point at a different blob after an update. |
| `version` | Future contract changes must invalidate old identities. |

## Digest

`EmbeddingIdentity.digest` is SHA-256 over canonical JSON:

- object keys sorted lexically;
- no insignificant whitespace;
- UTF-8;
- lowercase hex output.

Changing any field changes the digest. `metadata()` returns the digest and all
fields. Store that object beside the vector. `identity_from_metadata()` verifies
the digest before accepting it.

## What to store

For every vector, persist:

```json
{
  "identity_version": 1,
  "identity_digest": "<sha256>",
  "provider": "ollama",
  "model": "embeddinggemma-2:270m",
  "dimensions": 768,
  "prompt_contract": {
    "query_prefix": "task: search result | query: ",
    "document_prefix": "title: none | text: "
  },
  "task_type": null,
  "artifact_digest": "<optional local model digest>"
}
```

Do not store only the model name. A model name is not a contract.

## Reindex rule

If the identity changes, the old vectors are stale. Do not compare them with the
new vectors, even if the dimensions match. Re-embed the records, or keep two
separate indexes with separate identities.

`backfill()` follows this rule: it preserves an existing vector only when its
stored metadata matches every identity field. Everything else is replaced.
