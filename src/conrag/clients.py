from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx
import numpy as np
from numpy.typing import NDArray
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    InternalServerError,
    RateLimitError,
)
from sentence_transformers import SentenceTransformer
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
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
        self._request_args: dict[str, object] = {
            "model": config.llm_model,
            "max_output_tokens": config.max_output_tokens,
            "temperature": config.temperature,
        }
        self._retry_count = config.llm_retry_count
        self._retry_wait = config.llm_retry_backoff_seconds
        self._retry_wait_max = max(
            self._retry_wait, config.llm_retry_max_backoff_seconds
        )
        self._client = AsyncOpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            http_client=DefaultAsyncHttpxClient(
                timeout=config.llm_timeout_seconds,
                transport=httpx.AsyncHTTPTransport(http2=True, trust_env=False),
            ),
        )

    async def infer(self, *, instructions: str, input_text: str) -> str:
        retryable = (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            InternalServerError,
            httpx.TimeoutException,
            httpx.NetworkError,
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._retry_count + 1),
            wait=wait_exponential(
                multiplier=self._retry_wait,
                min=self._retry_wait,
                max=self._retry_wait_max,
            ),
            retry=retry_if_exception_type(retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                response = await self._client.responses.create(
                    **self._request_args,
                    instructions=instructions.strip(),
                    input=input_text.strip(),
                )
                return clean_llm_text(response.output_text)
        raise RuntimeError("LLM inference finished without a response")

    async def close(self) -> None:
        await self._client.close()


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
