# Security policy

## Report a problem

Do not open a public issue for a vulnerability. Use GitHub's private
[security advisory](https://github.com/alexandryasukaitisss-cmd/embedding-contract/security/advisories/new)
form, or contact the maintainer through the address on the GitHub profile.

Include the version, commit, Python version, provider, the smallest
reproduction, and whether the issue crosses the documented boundary. Do not
send API keys, private text, vectors from a personal store, or provider
payloads.

## Scope

In scope:

- a vector accepted with the wrong dimensions, a zero/NaN/infinite value, or a
  mismatched identity;
- a stale vector preserved by `backfill()` when its metadata does not match;
- a local Ollama URL that reaches a non-loopback host by default;
- a Gemini API key exposed in `repr`, an exception, or a CLI argument;
- a redirect, alternate endpoint, or cloud fallback that was not explicit;
- a provider error that echoes submitted text or an API key;
- a provider response parsed without shape/size limits.

Out of scope by design:

- the security of a user-supplied Ollama or Gemini service;
- the correctness of a model's embedding quality;
- a user who deliberately sets `allow_remote=True` for Ollama;
- a user who deliberately sends private text to Gemini.

## Response

The maintainer will acknowledge a report, reproduce it, and publish a fix and
advisory when confirmed. This is a single-maintainer project; no bounty is
offered.
