from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import cast

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

type Schema = dict[str, list[str]]


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


@dataclass(kw_only=True, slots=True)
class Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    dataset: str = "hotpotqa"
    mode: str = "run"
    rebuild_knowledge_base: bool = False

    log_level: str = "INFO"
    log_file_name: str = "app.log"

    llm_model: str = field(default_factory=lambda: env_str("CONRAG_LLM_MODEL", "gemini-2.5-flash"))
    vertex_api_key: str = field(default_factory=lambda: env_str("CONRAG_VERTEX_API_KEY", ""))
    llm_timeout_seconds: int = field(default_factory=lambda: env_int("CONRAG_LLM_TIMEOUT_SECONDS", 300))
    llm_retry_count: int = field(default_factory=lambda: env_int("CONRAG_LLM_RETRY_COUNT", 3))
    llm_retry_backoff_seconds: float = field(default_factory=lambda: env_float("CONRAG_LLM_RETRY_BACKOFF_SECONDS", 1.0))
    llm_retry_max_backoff_seconds: float = field(default_factory=lambda: env_float("CONRAG_LLM_RETRY_MAX_BACKOFF_SECONDS", 60.0))
    max_output_tokens: int = field(default_factory=lambda: env_int("CONRAG_MAX_OUTPUT_TOKENS", 8192))
    temperature: float = field(default_factory=lambda: env_float("CONRAG_TEMPERATURE", 0.0))

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = field(default_factory=lambda: env_str("CONRAG_EMBEDDING_DEVICE", "cuda"))
    embedding_batch_size: int = 32
    fanout_sample_count: int = field(default_factory=lambda: env_int("CONRAG_FANOUT_SAMPLE_COUNT", 20))
    fanout_chunk_target_chars: int = field(default_factory=lambda: env_int("CONRAG_FANOUT_CHUNK_TARGET_CHARS", 2_000))
    fanout_chunk_soft_max_chars: int = field(default_factory=lambda: env_int("CONRAG_FANOUT_CHUNK_SOFT_MAX_CHARS", 3_000))
    fanout_min_chunk_chars: int = field(default_factory=lambda: env_int("CONRAG_FANOUT_MIN_CHUNK_CHARS", 500))

    sequential_questions: bool = True
    max_workers: int = field(default_factory=lambda: env_int("CONRAG_MAX_WORKERS", 64))
    evidence_search_top_k: int = 16
    final_top_k: int = 3
    anchor_top_k: int = 4
    max_node_relation_summaries: int = 4
    direct_alpha: float = 0.20
    comp_alpha: float = 0.20
    chunk_alpha: float = 0.60
    consensus_lambda: float = 0.05
    beta: float = 0.02
    graph_penalty_floor: float = 0.8

    dataset_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    run_dir: Path = field(init=False)
    log_path: Path = field(init=False)
    schema: Schema = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.expanduser().resolve()
        self.dataset_dir = self.base_dir / "datasets" / self.dataset
        self.output_dir = self.base_dir / "outputs" / self.dataset
        self.run_dir = self.base_dir / "results" / self.dataset / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.log_path = self.run_dir / "logs" / self.log_file_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        schema_resource = resources.files("conrag.prompts").joinpath("schema.json")
        self.schema = parse_schema(json.loads(schema_resource.read_text(encoding="utf-8")))


def parse_schema(raw: object) -> Schema:
    if not isinstance(raw, dict):
        raise TypeError("schema must be a JSON object")
    schema: Schema = {}
    for key in ("nodes", "relations", "attributes"):
        values = raw.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise TypeError(f"schema.{key} must be a non-empty list of strings")
        schema[key] = cast(list[str], values)
    return schema
