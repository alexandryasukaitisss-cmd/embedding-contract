"""Small CLI for health checks, single embeddings, and JSONL backfill.

The CLI never accepts an API key as a command-line argument; Gemini reads
GEMINI_API_KEY from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .backfill import backfill
from .errors import EmbeddingError
from .providers.gemini import DEFAULT_BASE_URL, GeminiProvider
from .providers.ollama import OllamaProvider


GEMMA_CONTRACT = {
    "query_prefix": "task: search result | query: ",
    "document_prefix": "title: none | text: ",
}


def _add_provider(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("ollama", "gemini"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--document-prefix", default=None)
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--expected-digest", default=None)
    parser.add_argument("--allow-remote", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedding-contract")
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("health", help="check provider and identity")
    _add_provider(health)

    embed = commands.add_parser("embed", help="embed one text as JSON")
    _add_provider(embed)
    embed.add_argument("--purpose", choices=("query", "document"), default="query")
    embed.add_argument("--text", required=True)

    backfill_cmd = commands.add_parser("backfill", help="backfill a JSONL file")
    _add_provider(backfill_cmd)
    backfill_cmd.add_argument("--input", required=True)
    backfill_cmd.add_argument("--output", required=True)
    backfill_cmd.add_argument("--force", action="store_true")
    return parser


def _provider(args: argparse.Namespace):
    contract = None
    if args.query_prefix is not None or args.document_prefix is not None:
        contract = {
            "query_prefix": args.query_prefix or "",
            "document_prefix": args.document_prefix or "",
        }
    if args.provider == "ollama":
        if contract is None and args.model == "embeddinggemma-2:270m":
            contract = GEMMA_CONTRACT
        return OllamaProvider(
            model=args.model,
            dimensions=args.dimensions,
            base_url=args.base_url or "http://127.0.0.1:11434",
            prompt_contract=contract,
            batch_size=args.batch_size,
            timeout=args.timeout,
            expected_digest=args.expected_digest,
            allow_remote=args.allow_remote,
        )
    return GeminiProvider(
        model=args.model,
        dimensions=args.dimensions,
        base_url=args.base_url or DEFAULT_BASE_URL,
        prompt_contract=contract,
        task_type=args.task_type,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                raise EmbeddingError("invalid_input", f"line {line_number} is not valid JSON") from None
            if not isinstance(value, dict):
                raise EmbeddingError("invalid_input", f"line {line_number} is not a JSON object")
            records.append(value)
    return records


def _forbidden_key_argument(argv: list[str]) -> bool:
    return any(argument == "--api-key" or argument.startswith("--api-key=") or argument.startswith("--api_key") for argument in argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if _forbidden_key_argument(raw):
        print(
            json.dumps(
                {
                    "error": "api_key_argument_forbidden",
                    "message": "Pass GEMINI_API_KEY through the environment, never as a CLI argument.",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    parser = build_parser()
    args = parser.parse_args(raw)
    try:
        provider = _provider(args)
        if args.command == "health":
            health = provider.health()
            print(json.dumps({"ok": health.ok, "identity": health.identity.metadata(), "details": health.details}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "embed":
            batch = provider.embed([args.text], purpose=args.purpose)
            print(json.dumps({"identity": batch.identity.metadata(), "vector": batch.vectors[0]}, ensure_ascii=False))
            return 0
        if args.command == "backfill":
            records = _read_jsonl(args.input)
            summary = backfill(records, provider, force=args.force)
            output = Path(args.output)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(output.parent))
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                os.replace(temporary, output)
            except BaseException:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise
            print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
            return 0
        parser.error("unknown command")
    except EmbeddingError as error:
        print(json.dumps({"error": error.code, "message": error.message}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
