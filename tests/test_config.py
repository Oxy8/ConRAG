from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from conrag.config import Config


class ConfigEnvTests(unittest.TestCase):
    def make_config(self, **overrides: object) -> Config:
        with tempfile.TemporaryDirectory() as tmpdir:
            return Config(base_dir=Path(tmpdir), **overrides)

    def test_env_values_populate_provider_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CONRAG_LLM_MODEL": "gemini-2.5-pro",
                "CONRAG_VERTEX_API_KEY": "vertex-key",
                "CONRAG_LLM_TIMEOUT_SECONDS": "123",
                "CONRAG_MAX_OUTPUT_TOKENS": "456",
                "CONRAG_TEMPERATURE": "0.25",
                "CONRAG_EMBEDDING_DEVICE": "cpu",
            },
            clear=False,
        ):
            config = self.make_config()

        self.assertEqual(config.llm_model, "gemini-2.5-pro")
        self.assertEqual(config.vertex_api_key, "vertex-key")
        self.assertEqual(config.llm_timeout_seconds, 123)
        self.assertEqual(config.max_output_tokens, 456)
        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(config.embedding_device, "cpu")

    def test_constructor_overrides_environment_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CONRAG_LLM_MODEL": "env-model",
                "CONRAG_VERTEX_API_KEY": "env-key",
            },
            clear=False,
        ):
            config = self.make_config(llm_model="cli-model", vertex_api_key="cli-key")

        self.assertEqual(config.llm_model, "cli-model")
        self.assertEqual(config.vertex_api_key, "cli-key")


if __name__ == "__main__":
    unittest.main()
