from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import json_repair
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from conrag.common import clean_text
from conrag.planning import NOT_FOUND, PlanStep, normalize_plan, render_plan
from conrag.prompts import (
    FINAL_ANSWER_PROMPT,
    QUESTION_DECOMPOSITION_PROMPT,
    SINGLE_QUESTION_ANSWER_PROMPT,
)

if TYPE_CHECKING:
    from conrag.clients import EmbeddingClient, LLMClient
    from conrag.config import Config
    from conrag.graph import Entity
    from conrag.vector_store import VectorStore

logger = logging.getLogger(__name__)


type ScoreMap = dict[str, float]
type SupportMap = dict[str, set[str]]
type JsonObject = dict[str, object]
type Prompt = dict[str, str]
type TraceRecord = dict[str, object]


@dataclass(slots=True, frozen=True)
class Hit:
    chunk_id: str
    score: float


@dataclass(slots=True, frozen=True)
class GraphHits:
    scores: ScoreMap
    supports: SupportMap


@dataclass(slots=True, frozen=True)
class Decomposition:
    acquired_information: str
    plan: list[PlanStep]


@dataclass(slots=True, frozen=True)
class StepResult:
    answer: str
    acquired_information: str


@dataclass(slots=True, frozen=True)
class AnswerWithTrace:
    answer: str
    trace: TraceRecord


class RetrievalEngine:
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        embeddings: EmbeddingClient,
        store: VectorStore,
        graph: nx.MultiDiGraph,
        chunks: dict[str, str],
    ) -> None:
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.store = store
        self.graph = graph
        self.chunks = chunks

    async def answer(self, question: str) -> AnswerWithTrace:
        decomposition, decomposition_trace = await self._decompose_question(question)
        memory = [decomposition.acquired_information] if decomposition.acquired_information else []
        base_memory = list(memory)

        step_answers, step_traces = await self._execute_plan(question, decomposition.plan, memory)
        final_answer, final_trace = await self._answer_from_plan(question, decomposition.plan, step_answers, memory)

        return AnswerWithTrace(
            answer=final_answer,
            trace={
                "question": question,
                "initial_memory": base_memory,
                "decomposition": decomposition_trace,
                "steps": step_traces,
                "step_answers": serialize_answers(step_answers),
                "final_synthesis": final_trace,
                "final_memory": list(memory),
                "final_answer": final_answer,
            },
        )

    async def _decompose_question(self, question: str) -> tuple[Decomposition, TraceRecord]:
        context, retrieval_trace = await self._retrieve_context(question)
        fallback_plan = single_step_plan(question)
        payload, call_trace = await self._json_call(
            "question_decomposition",
            QUESTION_DECOMPOSITION_PROMPT,
            {"question": question, "evidence": context},
            {"acquired_information": "", "plan": fallback_plan},
        )
        raw_plan = payload.get("plan", [])
        normalized_plan = normalize_plan(raw_plan)
        used_fallback = not normalized_plan
        plan = normalized_plan or fallback_plan
        trace = {
            "retrieval": retrieval_trace,
            "prompt": {
                "prompt_name": "question_decomposition",
                "inputs": {
                    "question": question,
                    "evidence": context,
                },
                "raw_response": call_trace.get("raw_response", ""),
                "parsed_payload": call_trace.get("parsed_payload", {"acquired_information": "", "plan": fallback_plan}),
                "error": call_trace.get("error", ""),
            },
            "output": {
                "acquired_information": clean_optional(payload.get("acquired_information")),
                "raw_plan": make_jsonable(raw_plan),
                "normalized_plan": make_jsonable(plan),
                "used_fallback": used_fallback,
                "normalization_changed_plan": make_jsonable(raw_plan) != make_jsonable(plan),
            },
        }
        return (
            Decomposition(
                acquired_information=clean_optional(payload.get("acquired_information")),
                plan=plan,
            ),
            trace,
        )

    async def _execute_plan(
        self,
        question: str,
        plan: list[PlanStep],
        memory: list[str],
    ) -> tuple[dict[int, str], list[TraceRecord]]:
        plan = normalize_plan(plan)
        if not plan:
            return {}, []

        base_memory = tuple(memory)
        events = {step["id"]: asyncio.Event() for step in plan}
        answers: dict[int, str] = {}
        new_memory: dict[int, str] = {}
        step_traces: dict[int, TraceRecord] = {}

        async with asyncio.TaskGroup() as group:
            for step in plan:
                group.create_task(
                    self._execute_step(question, step, base_memory, events, answers, new_memory, step_traces)
                )

        for step in plan:
            if text := new_memory.get(step["id"]):
                memory.append(text)
        return answers, [step_traces[step["id"]] for step in plan]

    async def _execute_step(
        self,
        question: str,
        step: PlanStep,
        base_memory: tuple[str, ...],
        events: dict[int, asyncio.Event],
        answers: dict[int, str],
        new_memory: dict[int, str],
        step_traces: dict[int, TraceRecord],
    ) -> None:
        step_id = step["id"]
        raw_sub_question = step["sub_question"]
        trace: TraceRecord = {
            "step_id": step_id,
            "raw_sub_question": raw_sub_question,
            "dependencies": list(step["dependencies"]),
            "dependency_answers": {},
            "resolved_sub_question": raw_sub_question,
            "memory_before": [],
            "retrieval": {},
            "prompt": {},
            "result": {
                "answer": NOT_FOUND,
                "acquired_information": "",
            },
            "status": "pending",
            "error": "",
        }
        try:
            await self._wait_for_dependencies(step, events)
            trace["dependency_answers"] = {
                str(dep_id): answers.get(dep_id, NOT_FOUND)
                for dep_id in step["dependencies"]
            }
            resolved_sub_question = render_sub_question(step, answers)
            trace["resolved_sub_question"] = resolved_sub_question
            step_memory = collect_step_memory(step, base_memory, new_memory)
            trace["memory_before"] = list(step_memory)

            result, step_detail = await self._answer_step(question, resolved_sub_question, step_memory)
            answers[step_id] = result.answer
            if result.acquired_information:
                new_memory[step_id] = result.acquired_information

            trace["retrieval"] = step_detail["retrieval"]
            trace["prompt"] = step_detail["prompt"]
            trace["result"] = {
                "answer": result.answer,
                "acquired_information": result.acquired_information,
            }
            trace["status"] = "not_found" if result.answer == NOT_FOUND else "success"
        except Exception as exc:
            logger.exception("Plan step failed: step_id=%s question=%s", step_id, raw_sub_question)
            answers[step_id] = NOT_FOUND
            trace["status"] = "failed"
            trace["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            step_traces[step_id] = trace
            events[step_id].set()

    async def _wait_for_dependencies(self, step: PlanStep, events: dict[int, asyncio.Event]) -> None:
        for dep_id in step["dependencies"]:
            await events[dep_id].wait()

    async def _answer_step(
        self,
        question: str,
        sub_question: str,
        memory: tuple[str, ...],
    ) -> tuple[StepResult, TraceRecord]:
        context, retrieval_trace = await self._retrieve_context(sub_question)
        formatted_memory = format_memory(memory)
        payload, call_trace = await self._json_call(
            "single_question_answer",
            SINGLE_QUESTION_ANSWER_PROMPT,
            {
                "original_question": question,
                "acquired_information": formatted_memory,
                "sub_question": sub_question,
                "evidence": context,
            },
            {"answer": NOT_FOUND, "acquired_information": ""},
        )
        result = StepResult(
            answer=clean_answer(payload.get("answer")),
            acquired_information=clean_optional(payload.get("acquired_information")),
        )
        trace = {
            "retrieval": retrieval_trace,
            "prompt": {
                "prompt_name": "single_question_answer",
                "inputs": {
                    "original_question": question,
                    "acquired_information": formatted_memory,
                    "sub_question": sub_question,
                    "evidence": context,
                },
                "raw_response": call_trace.get("raw_response", ""),
                "parsed_payload": call_trace.get("parsed_payload", {"answer": NOT_FOUND, "acquired_information": ""}),
                "error": call_trace.get("error", ""),
            },
        }
        return result, trace

    async def _answer_from_plan(
        self,
        question: str,
        plan: list[PlanStep],
        answers: dict[int, str],
        memory: list[str],
    ) -> tuple[str, TraceRecord]:
        rendered_plan = render_plan(plan, answers)
        rendered_memory = [item for item in memory if item]
        evidence = "\n\n".join(
            item
            for item in (
                rendered_plan,
                "\n".join(f"- {item}" for item in rendered_memory),
            )
            if item
        )
        trace: TraceRecord = {
            "prompt": {
                "prompt_name": "final_answer",
                "inputs": {
                    "question": question,
                    "evidence": evidence,
                },
                "raw_response": "",
                "error": "",
            },
            "rendered_plan": rendered_plan,
            "memory": rendered_memory,
            "evidence": evidence,
            "final_answer": NOT_FOUND,
            "status": "pending",
        }
        try:
            raw = await self.llm.infer(
                instructions=FINAL_ANSWER_PROMPT["instructions"],
                input_text=FINAL_ANSWER_PROMPT["input"].format(
                    question=question,
                    evidence=evidence,
                ),
            )
            answer = clean_answer(raw)
            trace["prompt"]["raw_response"] = raw
            trace["final_answer"] = answer
            trace["status"] = "not_found" if answer == NOT_FOUND else "success"
            return answer, trace
        except Exception as exc:
            logger.exception("Final answer generation failed: question=%s", question[:80])
            trace["prompt"]["error"] = f"{type(exc).__name__}: {exc}"
            trace["status"] = "failed"
            return NOT_FOUND, trace

    async def _retrieve_context(self, text: str) -> tuple[str, TraceRecord]:
        query = await self.embeddings.encode_async(text, "query")
        hits, trace = await self._retrieve(query, self.config.final_top_k, text)
        context = self._render_context(hits)
        trace["rendered_context"] = context
        return context, trace

    async def _retrieve(self, query: NDArray[np.float32], top_k: int, query_text: str) -> tuple[list[Hit], TraceRecord]:
        relation_hits = self.store.search_relations(query, self.config.evidence_search_top_k)
        anchors = self.store.search_nodes(query, self.config.anchor_top_k)
        chunk_hits = self.store.search_chunks(query, self.config.evidence_search_top_k)

        direct = self._direct_graph_hits_from_relations(relation_hits)
        complement = self._complement_graph_hits(anchors)
        chunk_scores = dict(chunk_hits)

        direct_norm = normalize_scores(direct.scores)
        comp_norm = normalize_scores(complement.scores)
        chunk_norm = normalize_scores(chunk_scores)
        supports = merge_supports(direct.supports, complement.supports)

        fused: list[TraceRecord] = []
        hits = [
            Hit(
                chunk_id=chunk_id,
                score=self._fusion_score(
                    chunk_id,
                    direct_norm.get(chunk_id, 0.0),
                    comp_norm.get(chunk_id, 0.0),
                    chunk_norm.get(chunk_id, 0.0),
                    supports,
                ),
            )
            for chunk_id in set(direct_norm) | set(comp_norm) | set(chunk_norm)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)

        for hit in hits:
            chunk_id = hit.chunk_id
            fused.append({
                "chunk_id": chunk_id,
                "final_score": hit.score,
                "direct_score": direct_norm.get(chunk_id, 0.0),
                "complement_score": comp_norm.get(chunk_id, 0.0),
                "chunk_score": chunk_norm.get(chunk_id, 0.0),
                "support_nodes": sorted(supports.get(chunk_id, set())),
                "text": self.chunks.get(chunk_id, ""),
            })

        selected_hits = hits[:top_k]
        trace = {
            "query_text": query_text,
            "top_k": top_k,
            "relation_hits": [serialize_relation_hit(relation, score, self.graph) for relation, score in relation_hits],
            "anchor_hits": [serialize_anchor_hit(anchor, score, self.graph) for anchor, score in anchors],
            "chunk_hits": [serialize_chunk_hit(chunk_id, score, self.chunks) for chunk_id, score in chunk_hits],
            "direct_chunk_scores": serialize_score_entries(direct.scores, direct.supports, self.chunks),
            "complement_chunk_scores": serialize_score_entries(complement.scores, complement.supports, self.chunks),
            "normalized_scores": {
                "direct": serialize_score_entries(direct_norm, direct.supports, self.chunks),
                "complement": serialize_score_entries(comp_norm, complement.supports, self.chunks),
                "chunk": serialize_score_entries(chunk_norm, {}, self.chunks),
            },
            "fusion_scores": fused,
            "selected_hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "text": self.chunks.get(hit.chunk_id, ""),
                }
                for hit in selected_hits
            ],
            "rendered_context": "",
        }
        return selected_hits, trace

    def _direct_graph_hits_from_relations(self, relation_hits: list[tuple[Relation, float]]) -> GraphHits:
        scores: dict[str, list[float]] = defaultdict(list)
        supports: SupportMap = defaultdict(set)
        for relation, score in relation_hits:
            edge_data = self.graph.get_edge_data(relation.source_id, relation.target_id, key=relation.key)
            if not edge_data:
                continue
            for chunk_id in source_chunks(edge_data):
                scores[chunk_id].append(score)
                supports[chunk_id].update((relation.source_id, relation.target_id))

        return GraphHits(
            scores={chunk_id: max_plus_residual(values, self.config.beta) for chunk_id, values in scores.items()},
            supports=dict(supports),
        )

    def _complement_graph_hits(self, anchors: list[tuple[Entity, float]]) -> GraphHits:
        scores: ScoreMap = defaultdict(float)
        supports: SupportMap = defaultdict(set)

        for anchor, anchor_score in anchors:
            if anchor.id not in self.graph:
                continue

            degree_penalty = degree_penalty_value(self.graph.degree(anchor.id))
            self_contribution = anchor_score * degree_penalty * math.log1p(1.0)
            if math.isfinite(self_contribution) and self_contribution > 0.0:
                for chunk_id in source_chunks(self.graph.nodes[anchor.id]):
                    scores[chunk_id] += self_contribution
                    supports[chunk_id].add(anchor.id)

        return GraphHits(scores=dict(scores), supports=dict(supports))

    def _fusion_score(
        self,
        chunk_id: str,
        direct_score: float,
        comp_score: float,
        chunk_score: float,
        supports: SupportMap,
    ) -> float:
        raw_penalty = min(
            (
                degree_penalty_value(self.graph.degree(node_id))
                for node_id in supports.get(chunk_id, set())
                if node_id in self.graph
            ),
            default=1.0,
        )
        graph_penalty = clamp_unit(max(self.config.graph_penalty_floor, raw_penalty))
        graph_score = self.config.direct_alpha * direct_score + self.config.comp_alpha * comp_score * graph_penalty
        base_score = graph_score + self.config.chunk_alpha * chunk_score
        path_count = int(direct_score > 0.0) + int(comp_score > 0.0) + int(chunk_score > 0.0)
        consensus_bonus = 1.0 + self.config.consensus_lambda * max(0, path_count - 1) / 2.0
        return base_score * consensus_bonus

    def _render_context(self, hits: list[Hit]) -> str:
        return "\n\n".join(self.chunks[hit.chunk_id] for hit in hits if hit.chunk_id in self.chunks)

    async def _json_call(
        self,
        prompt_name: str,
        prompt: Prompt,
        values: JsonObject,
        fallback: JsonObject,
    ) -> tuple[JsonObject, TraceRecord]:
        trace: TraceRecord = {
            "prompt_name": prompt_name,
            "inputs": make_jsonable(values),
            "raw_response": "",
            "parsed_payload": make_jsonable(fallback),
            "used_fallback": True,
            "error": "",
        }
        try:
            raw = await self.llm.infer(
                instructions=prompt["instructions"],
                input_text=prompt["input"].format(**values),
            )
            trace["raw_response"] = raw
            payload = json_repair.loads(raw)
            trace["parsed_payload"] = make_jsonable(payload)
            if isinstance(payload, dict):
                trace["used_fallback"] = False
                return cast(JsonObject, payload), trace
            logger.warning("LLM payload was not a JSON object")
            trace["error"] = "LLM payload was not a JSON object"
        except Exception as exc:
            logger.exception("LLM JSON call failed")
            trace["error"] = f"{type(exc).__name__}: {exc}"
        return fallback, trace


def render_sub_question(step: PlanStep, answers: dict[int, str]) -> str:
    sub_question = step["sub_question"]
    for dep_id in step["dependencies"]:
        sub_question = sub_question.replace(f"<dep:{dep_id}>", answers.get(dep_id, NOT_FOUND))
    return sub_question


def collect_step_memory(step: PlanStep, base_memory: tuple[str, ...], new_memory: dict[int, str]) -> tuple[str, ...]:
    return base_memory + tuple(new_memory[dep_id] for dep_id in step["dependencies"] if dep_id in new_memory)


def format_memory(memory: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in memory if item)


def source_chunks(data: Mapping[str, object]) -> list[str]:
    chunks = data.get("source_chunks", [])
    if isinstance(chunks, str):
        return [chunks]
    if isinstance(chunks, list | tuple | set):
        return [str(chunk_id) for chunk_id in chunks if chunk_id]
    return []


def merge_supports(*support_maps: SupportMap) -> SupportMap:
    merged: SupportMap = defaultdict(set)
    for support_map in support_maps:
        for chunk_id, node_ids in support_map.items():
            merged[chunk_id].update(node_ids)
    return dict(merged)


def normalize_scores(scores: ScoreMap) -> ScoreMap:
    finite = {key: value for key, value in scores.items() if math.isfinite(value)}
    if not finite:
        return {}
    low = min(finite.values())
    high = max(finite.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in finite}
    return {key: (value - low) / (high - low) for key, value in finite.items()}


def max_plus_residual(scores: list[float], beta: float) -> float:
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return 0.0
    best = max(finite)
    return best + beta * (sum(finite) - best)


def degree_penalty_value(degree: int) -> float:
    return 1.0 if degree <= 1 else 1.0 / (1.0 + math.log(degree))


def clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def clean_answer(value: object) -> str:
    text = clean_text(NOT_FOUND if value is None else value)
    return text or NOT_FOUND


def clean_optional(value: object) -> str:
    return "" if value is None else clean_text(value)


def single_step_plan(question: str) -> list[PlanStep]:
    sub_question = clean_text(question) or "Answer the original question"
    return [{"id": 0, "sub_question": sub_question, "dependencies": []}]


def serialize_answers(answers: dict[int, str]) -> dict[str, str]:
    return {str(key): value for key, value in sorted(answers.items())}


def serialize_relation_hit(relation: Relation, score: float, graph: nx.MultiDiGraph) -> TraceRecord:
    source = graph.nodes.get(relation.source_id, {})
    target = graph.nodes.get(relation.target_id, {})
    edge_data = graph.get_edge_data(relation.source_id, relation.target_id, key=relation.key) or {}
    return {
        "source_id": relation.source_id,
        "source_name": str(source.get("name", relation.source_id)),
        "target_id": relation.target_id,
        "target_name": str(target.get("name", relation.target_id)),
        "relation": relation.text,
        "score": score,
        "source_chunks": source_chunks(edge_data),
    }


def serialize_anchor_hit(anchor: Entity, score: float, graph: nx.MultiDiGraph) -> TraceRecord:
    node_data = graph.nodes.get(anchor.id, {})
    return {
        "entity_id": anchor.id,
        "name": anchor.name,
        "type": anchor.type,
        "score": score,
        "source_chunks": source_chunks(node_data),
    }


def serialize_chunk_hit(chunk_id: str, score: float, chunks: dict[str, str]) -> TraceRecord:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "text": chunks.get(chunk_id, ""),
    }


def serialize_score_entries(scores: ScoreMap, supports: SupportMap, chunks: dict[str, str]) -> list[TraceRecord]:
    entries = [
        {
            "chunk_id": chunk_id,
            "score": score,
            "support_nodes": sorted(supports.get(chunk_id, set())),
            "text": chunks.get(chunk_id, ""),
        }
        for chunk_id, score in scores.items()
    ]
    entries.sort(key=lambda entry: float(entry["score"]), reverse=True)
    return entries


def make_jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [make_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return make_jsonable({
            str(key): item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        })
    return clean_text(value)
