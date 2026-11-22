# Embedding Contract

[![ci](https://github.com/alexandryasukaitisss-cmd/embedding-contract/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandryasukaitisss-cmd/embedding-contract/actions/workflows/ci.yml)

**A provider-neutral identity and backfill layer for agent memory.**

A logical continuation of **[MemPalace Retrieval](https://github.com/alexandryasukaitisss-cmd/mempalace-retrieval)** and **[Memory Ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger)**. The retrieval layer decides what may be served; this layer decides which vectors may share an index and how to backfill them without corrupting it.

Version 0.1.0. Python 3.11+. MIT. No runtime dependencies.

Русская версия: [README.ru.md](README.ru.md).

## Why

Most memory bugs are not ranking bugs. They are identity bugs:

- a vector from `model-a` is compared with a vector from `model-b`;
- the same model is used with a different prompt prefix or output dimension;
- an old vector survives a reindex and silently wins;
- a cloud fallback returns a valid-looking vector in a different space.

This package makes those states explicit and rejects them.

## Two models, two meanings

| Route | Model | Downloadable? | Notes |
| --- | --- | --- | --- |
| Local, default | **`embeddinggemma-2:270m`** | **Yes**, via Ollama | 768 dimensions. This is Google **Gemma**, not Gemini. The model runs on your machine; no text leaves it. |
| Cloud, optional | **`gemini-embedding-2`** | **No**, API only | Latest Gemini embedding model. 768 / 1536 / 3072 dimensions. Sends text to Google. |
| Cloud, text-only legacy | `gemini-embedding-001` | No, API only | Uses `taskType`; its space is **incompatible** with `gemini-embedding-2`. |

Do not use `text-embedding-004`; it was shut down on 14 January 2026. See the [official Gemini embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings).

**If you want to download a Google embedding model, download the Gemma one:**

```sh
ollama pull embeddinggemma-2:270m
```

Gemini is not a downloadable model. It is an API. The package supports both, but they are different providers with different identities and different privacy properties.

## Install

```sh
git clone https://github.com/alexandryasukaitisss-cmd/embedding-contract.git
cd embedding-contract
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Health check

Local Gemma through Ollama:

```sh
ollama pull embeddinggemma-2:270m
python -m embedding_contract.cli health \
  --provider ollama \
  --model embeddinggemma-2:270m \
  --dimensions 768 \
  --base-url http://127.0.0.1:11434
```

If your Ollama runs in an isolated service on another loopback port, pass that port explicitly. The model download is the same.

Cloud Gemini:

```sh
read -s GEMINI_API_KEY
export GEMINI_API_KEY
python -m embedding_contract.cli health \
  --provider gemini \
  --model gemini-embedding-2 \
  --dimensions 768
```

The cloud adapter is opt-in. There is no fallback from one provider to another.

## Embed

```sh
python -m embedding_contract.cli embed \
  --provider ollama \
  --model embeddinggemma-2:270m \
  --dimensions 768 \
  --purpose query \
  --text "where did we decide to store the ledger?"
```

The output contains the vector and the exact identity metadata that must be stored with it.

## Backfill

The `examples/` JSON files show constructor fields; the CLI uses explicit flags and applies the Gemma retrieval prefixes automatically for `embeddinggemma-2:270m`.

```sh
python -m embedding_contract.cli backfill \
  --provider ollama \
  --model embeddinggemma-2:270m \
  --dimensions 768 \
  --input records.jsonl \
  --output records.embedded.jsonl
```

Each JSONL record may contain:

```json
{"id":"fact-1","text":"Dispatch window: Thursday morning.","embedding":null,"embedding_metadata":null}
```

The function is storage-neutral. In Python:

```python
from embedding_contract import OllamaProvider, backfill

provider = OllamaProvider(
    model="embeddinggemma-2:270m",
    dimensions=768,
    base_url="http://127.0.0.1:11434",
    prompt_contract={
        "query_prefix": "task: search result | query: ",
        "document_prefix": "title: none | text: ",
    },
)

summary = backfill(
    records,
    provider,
    get_text=lambda record, index: record["body"],
    get_embedding=lambda record, index: record.get("vector"),
    get_metadata=lambda record, index: record.get("vector_metadata"),
    set_embedding=lambda record, vector, index: record.__setitem__("vector", vector),
    set_metadata=lambda record, metadata, index: record.__setitem__("vector_metadata", metadata),
)
```

A valid existing vector is preserved only when its metadata matches the current identity exactly. Missing, invalid, and stale vectors are replaced in bounded batches. The package does not own a database.

## Identity

An identity binds:

- provider and model;
- output dimensions;
- prompt contract (query/document prefixes or task pair);
- optional provider artifact digest (for local Ollama).

The digest is SHA-256 over canonical JSON with sorted keys. `EmbeddingIdentity.metadata()` returns the exact fields to persist with each vector. `identity_from_metadata()` verifies the digest before use.

Changing any field changes the identity. That is deliberate: a new identity means re-embedding, not mixing.

## What this package does not do

- It is not a vector database.
- It does not retrieve or rank results; use `mempalace-retrieval` for that.
- It does not record fact history; use `memory-ledger` for that.
- It does not start an HTTP server.
- It does not fall back to another provider, endpoint, or cloud service.
- It does not store or print API keys.

## Security and privacy

- Local Ollama is loopback-only by default: `http://127.0.0.1:<port>`, no credentials, no path, no redirects.
- Gemini uses HTTPS for every non-loopback origin. Loopback HTTP is accepted only for local tests and sends the key only to `127.0.0.1`. The key is read from the environment, never from a CLI flag, and is never included in errors or `repr`.
- Gemini sends text to Google. Use it only when that egress is acceptable for the data. The local Gemma route does not send text anywhere. Both adapters ignore environment HTTP proxies and refuse redirects. The Gemini adapter is covered by fake-wire tests; run `health` with your own key before a production backfill.
- Provider responses are size-limited, shape-checked, dimension-checked, finite-checked, zero-checked, and L2-normalized.
- There is no silent truncation; dimension mismatches fail closed.

See [SECURITY.md](SECURITY.md) and [docs/identity.md](docs/identity.md).

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The tests use fake HTTP servers and require no model, network, API key, or personal database. They cover identity digests, dimension and normalization failures, stale/invalid backfill, provider identity, prefixes/task types, key redaction, and no-fallback behavior.

## Project family

- [MemPalace Retrieval](https://github.com/alexandryasukaitisss-cmd/mempalace-retrieval) — permission-first search and evidence-based abstention.
- [Memory Ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger) — append-only fact history and supersession.
- **Embedding Contract** — vector identity and safe backfill (this repository).

## License

MIT. See [LICENSE](LICENSE).
