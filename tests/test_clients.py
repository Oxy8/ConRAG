from __future__ import annotations

import importlib
import sys
import tempfile
import types as pytypes
import unittest
from pathlib import Path
from unittest.mock import patch

from conrag.config import Config


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    def load_clients_module(self) -> tuple[object, list[object]]:
        fake_clients: list[object] = []

        class FakeAPIError(Exception):
            def __init__(self, code: int, message: str) -> None:
                super().__init__(message)
                self.code = code
                self.message = message

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs: object) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeHttpOptions:
            def __init__(self, **kwargs: object) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeModels:
            def __init__(self, owner: object) -> None:
                self.owner = owner
                self.calls: list[dict[str, object]] = []
                self.responses: list[object] = []

            async def generate_content(self, **kwargs: object) -> object:
                self.calls.append(kwargs)
                response = self.responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                return response

        class FakeAsyncClient:
            def __init__(self, owner: object) -> None:
                self.owner = owner
                self.models = FakeModels(owner)
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.aio = FakeAsyncClient(self)
                fake_clients.append(self)

        fake_google = pytypes.ModuleType("google")
        fake_genai = pytypes.ModuleType("google.genai")
        fake_genai.Client = FakeClient
        fake_genai.errors = pytypes.SimpleNamespace(APIError=FakeAPIError)
        fake_genai.types = pytypes.SimpleNamespace(
            GenerateContentConfig=FakeGenerateContentConfig,
            HttpOptions=FakeHttpOptions,
        )
        fake_google.genai = fake_genai

        fake_sentence_transformers = pytypes.ModuleType("sentence_transformers")
        fake_numpy = pytypes.ModuleType("numpy")
        fake_numpy.ascontiguousarray = lambda value, dtype=None: value
        fake_numpy.float32 = "float32"
        fake_numpy.empty = lambda shape, dtype=None: []
        fake_numpy_typing = pytypes.ModuleType("numpy.typing")
        fake_numpy_typing.NDArray = object
        fake_tenacity = pytypes.ModuleType("tenacity")

        class FakeSentenceTransformer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer

        class FakeAttempt:
            def __init__(self, retrying: object) -> None:
                self.retrying = retrying

            def __enter__(self) -> "FakeAttempt":
                return self

            def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
                if exc is None:
                    self.retrying.completed = True
                    return False
                if self.retrying.retry(exc) and self.retrying.attempt_number < self.retrying.stop:
                    return True
                return False

        class FakeAsyncRetrying:
            def __init__(
                self,
                *,
                stop: int,
                wait: object,
                retry: object,
                before_sleep: object,
                reraise: bool,
            ) -> None:
                self.stop = stop
                self.wait = wait
                self.retry = retry
                self.before_sleep = before_sleep
                self.reraise = reraise
                self.attempt_number = 0
                self.completed = False

            def __aiter__(self) -> "FakeAsyncRetrying":
                return self

            async def __anext__(self) -> FakeAttempt:
                if self.completed or self.attempt_number >= self.stop:
                    raise StopAsyncIteration
                self.attempt_number += 1
                return FakeAttempt(self)

        fake_tenacity.AsyncRetrying = FakeAsyncRetrying
        fake_tenacity.before_sleep_log = lambda logger, level: None
        fake_tenacity.retry_if_exception = lambda predicate: predicate
        fake_tenacity.stop_after_attempt = lambda attempts: attempts
        fake_tenacity.wait_exponential = lambda **kwargs: kwargs

        with patch.dict(
            sys.modules,
            {
                "google": fake_google,
                "google.genai": fake_genai,
                "google.genai.errors": fake_genai.errors,
                "google.genai.types": fake_genai.types,
                "sentence_transformers": fake_sentence_transformers,
                "numpy": fake_numpy,
                "numpy.typing": fake_numpy_typing,
                "tenacity": fake_tenacity,
            },
        ):
            sys.modules.pop("conrag.clients", None)
            clients = importlib.import_module("conrag.clients")
            return importlib.reload(clients), fake_clients

    def make_config(self, **overrides: object) -> Config:
        with tempfile.TemporaryDirectory() as tmpdir:
            return Config(base_dir=Path(tmpdir), vertex_api_key="vertex-key", **overrides)

    async def test_client_uses_vertex_express_mode_and_v1(self) -> None:
        clients, fake_clients = self.load_clients_module()
        llm = clients.LLMClient(self.make_config())

        self.assertEqual(len(fake_clients), 1)
        kwargs = fake_clients[0].kwargs
        self.assertTrue(kwargs["vertexai"])
        self.assertEqual(kwargs["api_key"], "vertex-key")
        self.assertEqual(kwargs["http_options"].api_version, "v1")
        await llm.close()

    async def test_infer_uses_system_instruction_and_cleans_text(self) -> None:
        clients, fake_clients = self.load_clients_module()
        llm = clients.LLMClient(self.make_config())
        fake_clients[0].aio.models.responses.append(pytypes.SimpleNamespace(text="```json\n{\"ok\": true}\n```"))

        text = await llm.infer(instructions="System prompt", input_text="User prompt")

        self.assertEqual(text, "{\"ok\": true}")
        call = fake_clients[0].aio.models.calls[0]
        self.assertEqual(call["model"], llm._model)
        self.assertEqual(call["contents"], "User prompt")
        self.assertEqual(call["config"].system_instruction, "System prompt")
        await llm.close()

    async def test_infer_falls_back_to_candidate_parts(self) -> None:
        clients, fake_clients = self.load_clients_module()
        llm = clients.LLMClient(self.make_config())
        fake_clients[0].aio.models.responses.append(
            pytypes.SimpleNamespace(
                text="",
                parts=[],
                candidates=[
                    pytypes.SimpleNamespace(
                        content=pytypes.SimpleNamespace(
                            parts=[
                                pytypes.SimpleNamespace(text="first line"),
                                pytypes.SimpleNamespace(text="second line"),
                            ]
                        )
                    )
                ],
            )
        )

        text = await llm.infer(instructions="System prompt", input_text="User prompt")

        self.assertEqual(text, "first line\nsecond line")
        await llm.close()

    async def test_transient_api_errors_are_retried(self) -> None:
        clients, fake_clients = self.load_clients_module()
        llm = clients.LLMClient(
            self.make_config(
                llm_retry_count=1,
                llm_retry_backoff_seconds=0.0,
                llm_retry_max_backoff_seconds=0.0,
            )
        )
        fake_clients[0].aio.models.responses.append(clients.errors.APIError(429, "rate limited"))
        fake_clients[0].aio.models.responses.append(pytypes.SimpleNamespace(text="ok"))

        text = await llm.infer(instructions="System prompt", input_text="User prompt")

        self.assertEqual(text, "ok")
        self.assertEqual(len(fake_clients[0].aio.models.calls), 2)
        await llm.close()


if __name__ == "__main__":
    unittest.main()
