from __future__ import annotations

import importlib
import json
import sys
import types as pytypes
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch


class FakeGraph:
    def __init__(self) -> None:
        self.nodes = {
            "source::entity": {"name": "Source", "type": "entity", "source_chunks": ["chunk-1"]},
            "target::entity": {"name": "Target", "type": "entity", "source_chunks": ["chunk-1", "chunk-2"]},
        }
        self._edges = {
            ("source::entity", "target::entity", ("source::entity", "target::entity", "related_to")): {
                "relation": "related_to",
                "source_chunks": ["chunk-1"],
            }
        }

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def degree(self, node_id: str) -> int:
        return 1 if node_id in self.nodes else 0

    def get_edge_data(self, source_id: str, target_id: str, key: object = None) -> dict[str, object] | None:
        return self._edges.get((source_id, target_id, key))


class FakeEmbeddings:
    async def encode_async(self, text: str, task: str = "query") -> str:
        return text


class FakeStore:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty

    def search_relations(self, query: object, top_k: int) -> list[tuple[object, float]]:
        if self.empty:
            return []
        relation = SimpleNamespace(
            source_id="source::entity",
            target_id="target::entity",
            key=("source::entity", "target::entity", "related_to"),
            text="related_to",
        )
        return [(relation, 0.9)]

    def search_nodes(self, query: object, top_k: int) -> list[tuple[object, float]]:
        if self.empty:
            return []
        entity = SimpleNamespace(id="target::entity", name="Target", type="entity")
        return [(entity, 0.8)]

    def search_chunks(self, query: object, top_k: int) -> list[tuple[str, float]]:
        if self.empty:
            return []
        return [("chunk-1", 0.7), ("chunk-2", 0.4)]


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def infer(self, *, instructions: str, input_text: str) -> str:
        self.calls.append((instructions, input_text))
        if not self.responses:
            raise RuntimeError("No more fake responses")
        return self.responses.pop(0)


class RetrievalTraceTests(unittest.TestCase):
    @contextmanager
    def retrieval_module(self) -> object:
        fake_tqdm = pytypes.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda *args, **kwargs: SimpleNamespace(update=lambda n: None)

        fake_networkx = pytypes.ModuleType("networkx")
        fake_networkx.MultiDiGraph = object

        fake_numpy = pytypes.ModuleType("numpy")
        fake_numpy.float32 = float
        fake_numpy_typing = pytypes.ModuleType("numpy.typing")
        fake_numpy_typing.NDArray = object

        fake_json_repair = pytypes.ModuleType("json_repair")
        fake_json_repair.loads = json.loads

        with patch.dict(
            sys.modules,
            {
                "tqdm": fake_tqdm,
                "networkx": fake_networkx,
                "numpy": fake_numpy,
                "numpy.typing": fake_numpy_typing,
                "json_repair": fake_json_repair,
            },
        ):
            sys.modules.pop("conrag.common", None)
            sys.modules.pop("conrag.retrieval", None)
            importlib.import_module("conrag.common")
            retrieval = importlib.import_module("conrag.retrieval")
            yield importlib.reload(retrieval)

    def make_config(self) -> object:
        return SimpleNamespace(
            final_top_k=2,
            anchor_top_k=2,
            evidence_search_top_k=3,
            beta=0.02,
            direct_alpha=0.2,
            comp_alpha=0.2,
            chunk_alpha=0.6,
            consensus_lambda=0.05,
            graph_penalty_floor=0.8,
        )

    def test_answer_returns_json_serializable_trace(self) -> None:
        with self.retrieval_module() as retrieval:
            llm = FakeLLM(
                [
                    json.dumps(
                        {
                            "acquired_information": "Known bridge fact.",
                            "plan": [
                                {"id": 0, "sub_question": "Who is the bridge entity?", "dependencies": []},
                                {"id": 1, "sub_question": "Where is <dep:0> located?", "dependencies": [0]},
                            ],
                        }
                    ),
                    json.dumps({"answer": "Alice", "acquired_information": "Alice is the bridge entity."}),
                    json.dumps({"answer": "Paris", "acquired_information": "Alice is located in Paris."}),
                    "Paris",
                ]
            )
            engine = retrieval.RetrievalEngine(
                self.make_config(),
                llm,
                FakeEmbeddings(),
                FakeStore(),
                FakeGraph(),
                {"chunk-1": "Chunk one text", "chunk-2": "Chunk two text"},
            )

            outcome = self.run_async(engine.answer("Where is the bridge entity located?"))

            self.assertEqual(outcome.answer, "Paris")
            self.assertEqual(len(outcome.trace["steps"]), 2)
            self.assertIn("decomposition", outcome.trace)
            self.assertIn("final_synthesis", outcome.trace)
            self.assertTrue(outcome.trace["steps"][0]["retrieval"]["selected_hits"])
            self.assertEqual(outcome.trace["steps"][1]["dependency_answers"]["0"], "Alice")
            json.dumps(outcome.trace)

    def test_malformed_decomposition_uses_fallback_and_empty_retrieval_still_serializes(self) -> None:
        with self.retrieval_module() as retrieval:
            llm = FakeLLM(
                [
                    json.dumps({"acquired_information": "", "plan": "bad-plan"}),
                    json.dumps({"answer": "Information not found", "acquired_information": ""}),
                    "Information not found",
                ]
            )
            engine = retrieval.RetrievalEngine(
                self.make_config(),
                llm,
                FakeEmbeddings(),
                FakeStore(empty=True),
                FakeGraph(),
                {"chunk-1": "Chunk one text"},
            )

            outcome = self.run_async(engine.answer("Fallback question"))

            self.assertEqual(outcome.answer, "Information not found")
            self.assertTrue(outcome.trace["decomposition"]["output"]["used_fallback"])
            self.assertEqual(len(outcome.trace["steps"]), 1)
            self.assertEqual(outcome.trace["steps"][0]["retrieval"]["selected_hits"], [])
            json.dumps(outcome.trace)

    def run_async(self, coro: object) -> object:
        import asyncio

        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
