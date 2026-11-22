# Providers

## Local: Ollama + Gemma

Recommended when the text must stay on the machine.

```sh
ollama pull embeddinggemma-2:270m
```

- provider: `ollama`
- model: `embeddinggemma-2:270m`
- dimensions: `768`
- default origin: `http://127.0.0.1:11434`
- prompt contract: Gemma retrieval prefixes

The adapter rejects non-loopback HTTP origins by default, credentials in the
URL, paths, query strings, remote model tags, dimension mismatches, zero
vectors, non-finite values, and a digest that does not match a configured pin.
It has no cloud fallback.

`allow_remote=True` exists for an explicitly operated HTTPS Ollama host. It is
not the default.

## Cloud: Gemini

Gemini is an API, not a downloadable model. Text leaves the machine.

```sh
read -s GEMINI_API_KEY
export GEMINI_API_KEY
```

- provider: `gemini`
- latest multimodal model: `gemini-embedding-2`
- text-only legacy model: `gemini-embedding-001`
- recommended dimensions: `768`, `1536`, or `3072`
- default origin: `https://generativelanguage.googleapis.com/v1beta`

`gemini-embedding-2` uses task instructions in the prompt. The adapter uses the
same retrieval prefixes as the local Gemma contract. `gemini-embedding-001` uses
the `taskType` field instead and its space is incompatible with
`gemini-embedding-2`.

`text-embedding-004` was shut down on 14 January 2026. Do not use it.

The API key is read from `GEMINI_API_KEY`, never from a CLI flag. Prefer a secret manager or `read -s`; do not paste it into shell history. It is not
included in `repr`, error messages, or test fixtures.

The request uses `embedContent` / `batchEmbedContents` with
`embedContentConfig` for `outputDimensionality` and, for the legacy model,
`taskType`. The shape was checked against the current v1beta discovery schema;
the tests use a fake wire server. Run `health` with your own key before a
production backfill.

## Mixing providers

Do not mix. The package will not merge identities. A Gemini vector and a Gemma
vector have different `identity_digest` values even when both are 768-wide.
`backfill()` treats the other provider's vectors as stale.
