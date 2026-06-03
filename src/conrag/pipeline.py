from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import nanoid
import networkx as nx

from conrag.clients import EmbeddingClient, LLMClient
from conrag.common import clean_text, progress_bar, read_json, run_bounded, write_json
from conrag.config import Config
from conrag.evaluation import Evaluator
from conrag.graph import GraphBuilder
from conrag.retrieval import AnswerWithTrace, RetrievalEngine
from conrag.vector_store import VectorStore

logger = logging.getLogger(__name__)

type DatasetRecord = dict[str, object]
type AnswerRecord = dict[str, object]
type TraceRecord = dict[str, object]


class ConRAG:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.embeddings = EmbeddingClient(config)
        self.graph_builder = GraphBuilder(config, self.llm)
        self.store = VectorStore(config, self.embeddings)
        self.graph: nx.MultiDiGraph | None = None
        self.chunks: dict[str, str] = {}
        self.retrieval: RetrievalEngine | None = None

    def run(self) -> None:
        chunks, questions = load_dataset(self.config)
        asyncio.run(self.run_async(chunks, questions))

    def build_knowledge_base(self) -> None:
        chunks, _ = load_dataset(self.config)
        asyncio.run(self.build_knowledge_base_async(chunks))

    def query(self) -> None:
        chunks, questions = load_dataset(self.config)
        asyncio.run(self.query_async(chunks, questions))

    async def run_async(self, chunk_texts: list[str], questions: list[DatasetRecord]) -> None:
        try:
            await self.prepare(chunk_texts)
            await self.answer_dataset(questions)
        finally:
            await self.llm.close()

    async def build_knowledge_base_async(self, chunk_texts: list[str]) -> None:
        try:
            await self.prepare(chunk_texts)
        finally:
            await self.llm.close()

    async def query_async(self, chunk_texts: list[str], questions: list[DatasetRecord]) -> None:
        try:
            await self.prepare(chunk_texts, require_existing_knowledge_base=not self.config.rebuild_knowledge_base)
            await self.answer_dataset(questions)
        finally:
            await self.llm.close()

    async def answer_dataset(self, questions: list[DatasetRecord]) -> None:
        self.retrieval = RetrievalEngine(
            self.config,
            self.llm,
            self.embeddings,
            self.store,
            self.require_graph(),
            self.chunks,
        )
        answers, traces = await self.answer_questions(questions)
        results_path = self.config.run_dir / "results.json"
        trace_path = self.config.run_dir / "trace.json"
        await asyncio.to_thread(write_json, results_path, answers)
        await asyncio.to_thread(write_json, trace_path, make_trace_payload(self.config, traces))
        report = await Evaluator(self.config, self.llm, results_path).run()
        await asyncio.to_thread(enrich_trace_with_evaluation, trace_path, results_path, report)

    async def prepare(self, chunk_texts: list[str], *, require_existing_knowledge_base: bool = False) -> None:
        if not self.config.rebuild_knowledge_base:
            try:
                self.graph = self.graph_builder.load()
                self.chunks = self.store.load(self.graph)
                return
            except Exception as exc:
                if require_existing_knowledge_base:
                    raise RuntimeError(
                        "Persisted knowledge base unavailable. "
                        "Run build mode first or rerun with --rebuild_knowledge_base true."
                    ) from exc
                logger.warning("Persisted knowledge base unavailable: %s", exc)

        self.chunks = make_chunk_map(chunk_texts)
        self.graph = await self.graph_builder.build(self.chunks)
        self.graph_builder.save()
        await asyncio.to_thread(self.store.build, self.graph, self.chunks)
        await asyncio.to_thread(self.store.save)

    async def answer_questions(self, questions: list[DatasetRecord]) -> tuple[list[AnswerRecord], list[TraceRecord]]:
        results: list[AnswerRecord | None] = [None] * len(questions)
        traces: list[TraceRecord | None] = [None] * len(questions)
        with progress_bar(len(questions), "Online Inference", "question") as bar:

            async def process(index: int, record: DatasetRecord) -> None:
                try:
                    answer_record, trace_record = await self.answer_record(index, record)
                    results[index] = answer_record
                    traces[index] = trace_record
                finally:
                    bar.update(1)

            if self.config.sequential_questions:
                for index, record in enumerate(questions):
                    await process(index, record)
            else:
                await run_bounded(questions, self.config.max_workers, process)
        return (
            [result for result in results if result is not None],
            [trace for trace in traces if trace is not None],
        )

    async def answer_record(self, index: int, record: DatasetRecord) -> tuple[AnswerRecord, TraceRecord]:
        question = clean_text(record.get("question", ""))
        gold = clean_text(record.get("answer", ""))
        trace: TraceRecord = {
            "question_index": index,
            "dataset": self.config.dataset,
            "question": question,
            "gold_answer": gold,
            "pred_answer": "",
            "status": "pending",
        }
        if not question:
            prediction = "Error: empty question"
            trace["status"] = "failed"
            trace["error"] = prediction
        else:
            try:
                if self.retrieval is None:
                    raise RuntimeError("Retrieval engine is not initialized")
                outcome = await self.retrieval.answer(question)
                prediction = outcome.answer
                trace.update(outcome.trace)
                trace["status"] = "success" if not prediction.startswith("Error:") else "failed"
            except Exception as exc:
                logger.exception("Question failed: %s", question[:80])
                prediction = f"Error: {exc}"
                trace["status"] = "failed"
                trace["error"] = f"{type(exc).__name__}: {exc}"
        trace["pred_answer"] = prediction
        return {
            "dataset": self.config.dataset,
            "question": question,
            "gold_answer": gold,
            "pred_answer": prediction,
        }, trace

    def require_graph(self) -> nx.MultiDiGraph:
        if self.graph is None:
            raise RuntimeError("Graph is not initialized")
        return self.graph


def load_dataset(config: Config) -> tuple[list[str], list[DatasetRecord]]:
    corpus = read_json(config.dataset_dir / "corpus.json", list)
    raw_questions = read_json(config.dataset_dir / "questions.json", list)
    chunks = [
        f"{title}: {content}"
        for record in corpus
        if isinstance(record, dict)
        if (title := clean_text(record.get("title", ""))) and (content := clean_text(record.get("text", "")))
    ]
    questions = [record for record in raw_questions if isinstance(record, dict)]
    logger.info("Dataset loaded (chunks=%d, questions=%d)", len(chunks), len(questions))
    return chunks, questions


def make_chunk_map(chunks: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in chunks:
        chunk_id = nanoid.generate(size=8)
        while chunk_id in result:
            chunk_id = nanoid.generate(size=8)
        result[chunk_id] = chunk
    return result


def make_trace_payload(config: Config, traces: list[TraceRecord]) -> TraceRecord:
    return {
        "dataset": config.dataset,
        "run_name": config.run_dir.name,
        "run_path": str(config.run_dir),
        "question_count": len(traces),
        "questions": traces,
    }


def enrich_trace_with_evaluation(trace_path: Path, results_path: Path, report: dict[str, object]) -> None:
    trace_payload = read_json(trace_path, dict)
    result_records = [record for record in read_json(results_path, list) if isinstance(record, dict)]
    questions = trace_payload.get("questions", [])
    if not isinstance(questions, list):
        raise TypeError("trace.json must contain a questions list")

    for index, question_trace in enumerate(questions):
        if not isinstance(question_trace, dict):
            continue
        if index >= len(result_records):
            continue
        result = result_records[index]
        question_trace["evaluation"] = {
            "string_accuracy": result.get("string_accuracy", 0.0),
            "string_precision": result.get("string_precision", 0.0),
            "answer_accuracy": result.get("answer_accuracy", 0.0),
        }
        question_trace["pred_answer"] = str(result.get("pred_answer", question_trace.get("pred_answer", "")))
        question_trace["gold_answer"] = str(result.get("gold_answer", question_trace.get("gold_answer", "")))

    trace_payload["evaluation_report"] = dict(report)
    write_json(trace_path, trace_payload)
