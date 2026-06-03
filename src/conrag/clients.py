from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from google import genai
from google.genai import errors, types
from sentence_transformers import SentenceTransformer
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from conrag.config import Config

logger = logging.getLogger(__name__)

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_FENCE_RE = re.compile(r"```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```", re.MULTILINE)


class LLMClient:
    def __init__(self, config: Config) -> None:
        if not config.vertex_api_key.strip():
            raise ValueError(
                "Missing Vertex AI express mode API key. "
                "Set CONRAG_VERTEX_API_KEY in .env or pass --vertex_api_key."
            )
        self._model = config.llm_model
        self._generation_config = types.GenerateContentConfig(
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )
        self._retry_count = config.llm_retry_count
        self._retry_wait = config.llm_retry_backoff_seconds
        self._retry_wait_max = max(
            self._retry_wait, config.llm_retry_max_backoff_seconds
        )
        self._client = genai.Client(
            vertexai=True,
            api_key=config.vertex_api_key,
            http_options=types.HttpOptions(
                api_version="v1",
                async_client_args={"timeout": config.llm_timeout_seconds},
            ),
        ).aio

    async def infer(self, *, instructions: str, input_text: str) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._retry_count + 1),
            wait=wait_exponential(
                multiplier=self._retry_wait,
                min=self._retry_wait,
                max=self._retry_wait_max,
            ),
            retry=retry_if_exception(is_retryable_genai_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                response = await self._client.models.generate_content(
                    model=self._model,
                    contents=input_text.strip(),
                    config=types.GenerateContentConfig(
                        system_instruction=instructions.strip(),
                        max_output_tokens=self._generation_config.max_output_tokens,
                        temperature=self._generation_config.temperature,
                    ),
                )
                return clean_llm_text(extract_response_text(response))
        raise RuntimeError("LLM inference finished without a response")

    async def close(self) -> None:
        await self._client.aclose()


class EmbeddingClient:
    def __init__(self, config: Config) -> None:
        self._batch_size = config.embedding_batch_size
        self._model = SentenceTransformer(
            config.embedding_model,
            trust_remote_code=True,
            device=config.embedding_device,
        )

    async def encode_async(
        self, texts: str | Sequence[str], task: str = "document"
    ) -> NDArray[np.float32]:
        return await asyncio.to_thread(self.encode, texts, task)

    def encode(
        self, texts: str | Sequence[str], task: str = "document"
    ) -> NDArray[np.float32]:
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return np.empty((0, 0), dtype=np.float32)

        encode = getattr(self._model, f"encode_{task}", self._model.encode)
        vectors = np.ascontiguousarray(
            encode(
                batch,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors


def clean_llm_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = _ZERO_WIDTH_RE.sub(
        "", value.replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if match := _FENCE_RE.search(text):
        text = match.group("body").strip()
    elif text.startswith("```") and text.endswith("```") and len(text) >= 6:
        text = text[3:-3].strip()
    if text.lower().startswith("json\n"):
        text = text.split("\n", 1)[1].strip()
    return text


def is_retryable_genai_error(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code == 429 or code >= 500
        message = str(getattr(exc, "message", exc)).lower()
        return "timeout" in message or "tempor" in message
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def extract_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    if parts_text := extract_parts_text(getattr(response, "parts", None)):
        return parts_text

    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, Sequence) or isinstance(candidates, str):
        return ""

    rendered: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if parts_text := extract_parts_text(getattr(content, "parts", None)):
            rendered.append(parts_text)
    return "\n\n".join(part for part in rendered if part)


def extract_parts_text(parts: object) -> str:
    if not isinstance(parts, Sequence) or isinstance(parts, str):
        return ""

    rendered = [
        text
        for part in parts
        if isinstance(text := getattr(part, "text", None), str) and text.strip()
    ]
    return "\n".join(rendered)
