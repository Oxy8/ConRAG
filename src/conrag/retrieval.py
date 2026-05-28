from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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

    async def answer(self, question: str) -> str:
        decomposition = await self._decompose_question(question)
        memory = [decomposition.acquired_information] if decomposition.acquired_information else []

        step_answers = await self._execute_plan(question, decomposition.plan, memory)
        return await self._answer_from_plan(question, decomposition.plan, step_answers, memory)

    async def _decompose_question(self, question: str) -> Decomposition:
        context = await self._retrieve_context(question)
        fallback_plan = single_step_plan(question)
        payload = await self._json_call(
            QUESTION_DECOMPOSITION_PROMPT,
            {"question": question, "evidence": context},
            {"acquired_information": "", "plan": fallback_plan},
        )
        return Decomposition(
            acquired_information=clean_optional(payload.get("acquired_information")),
            plan=normalize_plan(payload.get("plan", [])) or fallback_plan,
        )

    async def _execute_plan(self, question: str, plan: list[PlanStep], memory: list[str]) -> dict[int, str]:
        plan = normalize_plan(plan)
        if not plan:
            return {}

        base_memory = tuple(memory)
        events = {step["id"]: asyncio.Event() for step in plan}
        answers: dict[int, str] = {}
        new_memory: dict[int, str] = {}

        async with asyncio.TaskGroup() as group:
            for step in plan:
                group.create_task(self._execute_step(question, step, base_memory, events, answers, new_memory))

        for step in plan:
            if text := new_memory.get(step["id"]):
                memory.append(text)
        return answers

    async def _execute_step(
        self,
        question: str,
        step: PlanStep,
        base_memory: tuple[str, ...],
        events: dict[int, asyncio.Event],
        answers: dict[int, str],
        new_memory: dict[int, str],
    ) -> None:
        step_id = step["id"]
        sub_question = step["sub_question"]
        try:
            await self._wait_for_dependencies(step, events)
            sub_question = render_sub_question(step, answers)
            step_memory = collect_step_memory(step, base_memory, new_memory)

            result = await self._answer_step(question, sub_question, step_memory)
            answers[step_id] = result.answer
            if result.acquired_information:
                new_memory[step_id] = result.acquired_information
        except Exception:
            logger.exception("Plan step failed: step_id=%s question=%s", step_id, sub_question)
            answers[step_id] = NOT_FOUND
        finally:
            events[step_id].set()

    async def _wait_for_dependencies(self, step: PlanStep, events: dict[int, asyncio.Event]) -> None:
        for dep_id in step["dependencies"]:
            await events[dep_id].wait()

    async def _answer_step(self, question: str, sub_question: str, memory: tuple[str, ...]) -> StepResult:
        context = await self._retrieve_context(sub_question)
        payload = await self._json_call(
            SINGLE_QUESTION_ANSWER_PROMPT,
            {
                "original_question": question,
                "acquired_information": format_memory(memory),
                "sub_question": sub_question,
                "evidence": context,
            },
            {"answer": NOT_FOUND, "acquired_information": ""},
        )
        return StepResult(
            answer=clean_answer(payload.get("answer")),
            acquired_information=clean_optional(payload.get("acquired_information")),
        )

    async def _answer_from_plan(
        self,
        question: str,
        plan: list[PlanStep],
        answers: dict[int, str],
        memory: list[str],
    ) -> str:
        evidence = "\n\n".join(
            item
            for item in (
                render_plan(plan, answers),
                "\n".join(f"- {item}" for item in memory),
            )
            if item
        )
        try:
            raw = await self.llm.infer(
                instructions=FINAL_ANSWER_PROMPT["instructions"],
                input_text=FINAL_ANSWER_PROMPT["input"].format(
                    question=question,
                    evidence=evidence,
                ),
            )
            return clean_answer(raw)
        except Exception:
            logger.exception("Final answer generation failed: question=%s", question[:80])
            return NOT_FOUND

    async def _retrieve_context(self, text: str) -> str:
        query = await self.embeddings.encode_async(text, "query")
        return self._render_context(await self._retrieve(query, self.config.final_top_k))

    async def _retrieve(self, query: NDArray[np.float32], top_k: int) -> list[Hit]:
        async with asyncio.TaskGroup() as group:
            direct_task = group.create_task(asyncio.to_thread(self._direct_graph_hits, query))
            anchor_task = group.create_task(asyncio.to_thread(self.store.search_nodes, query, self.config.anchor_top_k))
            chunk_task = group.create_task(
                asyncio.to_thread(self.store.search_chunks, query, self.config.evidence_search_top_k)
            )

        direct = direct_task.result()
        anchors = anchor_task.result()
        complement = await asyncio.to_thread(self._complement_graph_hits, anchors)
        chunk_scores = dict(chunk_task.result())

        direct_norm = normalize_scores(direct.scores)
        comp_norm = normalize_scores(complement.scores)
        chunk_norm = normalize_scores(chunk_scores)
        supports = merge_supports(direct.supports, complement.supports)

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
        return hits[:top_k]

    def _direct_graph_hits(self, query: NDArray[np.float32]) -> GraphHits:
        scores: dict[str, list[float]] = defaultdict(list)
        supports: SupportMap = defaultdict(set)
        for relation, score in self.store.search_relations(query, self.config.evidence_search_top_k):
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

    async def _json_call(self, prompt: Prompt, values: JsonObject, fallback: JsonObject) -> JsonObject:
        try:
            raw = await self.llm.infer(
                instructions=prompt["instructions"],
                input_text=prompt["input"].format(**values),
            )
            payload = json_repair.loads(raw)
            if isinstance(payload, dict):
                return cast(JsonObject, payload)
            logger.warning("LLM payload was not a JSON object")
        except Exception:
            logger.exception("LLM JSON call failed")
        return fallback


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
