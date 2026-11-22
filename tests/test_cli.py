import io
import unittest
from contextlib import redirect_stderr

from embedding_contract.cli import GEMMA_CONTRACT, _provider, build_parser, main


class CliTests(unittest.TestCase):
    def test_gemma_cli_applies_retrieval_prefixes(self):
        args = build_parser().parse_args([
            "health", "--provider", "ollama", "--model", "embeddinggemma-2:270m", "--dimensions", "768",
        ])
        provider = _provider(args)
        self.assertEqual(provider.prompt_contract, GEMMA_CONTRACT)

    def test_explicit_prefixes_override_defaults(self):
        args = build_parser().parse_args([
            "embed", "--provider", "ollama", "--model", "embeddinggemma-2:270m", "--dimensions", "768",
            "--text", "x", "--query-prefix", "Q:", "--document-prefix", "D:",
        ])
        provider = _provider(args)
        self.assertEqual(provider.prompt_contract, {"query_prefix": "Q:", "document_prefix": "D:"})

    def test_api_key_argument_is_rejected_without_echoing_the_value(self):
        for argv in (["--api-key", "secret"], ["--api-key=secret"]):
            output = io.StringIO()
            with redirect_stderr(output):
                self.assertEqual(main(["health", "--provider", "gemini", "--model", "gemini-embedding-2", "--dimensions", "768", *argv]), 2)
            self.assertNotIn("secret", output.getvalue())
            self.assertIn("api_key_argument_forbidden", output.getvalue())


if __name__ == "__main__":
    unittest.main()
