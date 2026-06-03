from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conrag.debug_web import render_index_page, render_question_page, render_run_page


class DebugWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmpdir.name)
        run_dir = self.base_dir / "results" / "example" / "2026-06-03_00-00-00_000000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trace.json").write_text(
            json.dumps(
                {
                "dataset": "example",
                "run_name": run_dir.name,
                "question_count": 1,
                "evaluation_report": {
                    "string_accuracy": 0.0,
                    "string_precision": 0.5,
                    "answer_accuracy": 0.0,
                },
                "questions": [
                    {
                        "question_index": 0,
                        "question": "Who directed the film?",
                        "gold_answer": "Christopher Nolan",
                        "pred_answer": "Steven Spielberg",
                        "status": "success",
                        "decomposition": {
                            "retrieval": {
                                "query_text": "Who directed the film?",
                                "selected_hits": [
                                    {"chunk_id": "chunk-1", "score": 1.0, "text": "Inception was directed by Christopher Nolan."}
                                ],
                                "rendered_context": "Inception was directed by Christopher Nolan.",
                            },
                            "prompt": {
                                "inputs": {"question": "Who directed the film?", "evidence": "Inception was directed by Christopher Nolan."},
                                "raw_response": "{\"plan\": [{\"id\": 0, \"sub_question\": \"Who directed Inception?\", \"dependencies\": []}]}",
                                "parsed_payload": {"plan": [{"id": 0, "sub_question": "Who directed Inception?", "dependencies": []}]},
                            },
                            "output": {
                                "acquired_information": "",
                                "normalized_plan": [{"id": 0, "sub_question": "Who directed Inception?", "dependencies": []}],
                                "used_fallback": False,
                                "normalization_changed_plan": False,
                            },
                        },
                        "steps": [
                            {
                                "step_id": 0,
                                "raw_sub_question": "Who directed Inception?",
                                "resolved_sub_question": "Who directed Inception?",
                                "dependencies": [],
                                "dependency_answers": {},
                                "memory_before": [],
                                "retrieval": {
                                    "query_text": "Who directed Inception?",
                                    "selected_hits": [
                                        {"chunk_id": "chunk-1", "score": 1.0, "text": "Inception was directed by Christopher Nolan."}
                                    ],
                                    "fusion_scores": [
                                        {"chunk_id": "chunk-1", "final_score": 1.0, "direct_score": 1.0, "complement_score": 0.0, "chunk_score": 1.0, "support_nodes": ["n1"], "text": "Inception was directed by Christopher Nolan."}
                                    ],
                                    "direct_chunk_scores": [],
                                    "complement_chunk_scores": [],
                                    "chunk_hits": [],
                                    "relation_hits": [],
                                    "anchor_hits": [],
                                    "normalized_scores": {},
                                    "rendered_context": "Inception was directed by Christopher Nolan.",
                                },
                                "prompt": {
                                    "inputs": {
                                        "original_question": "Who directed the film?",
                                        "acquired_information": "",
                                        "sub_question": "Who directed Inception?",
                                        "evidence": "Inception was directed by Christopher Nolan.",
                                    },
                                    "raw_response": "{\"answer\": \"Christopher Nolan\"}",
                                    "parsed_payload": {"answer": "Christopher Nolan"},
                                },
                                "result": {"answer": "Christopher Nolan", "acquired_information": "Christopher Nolan directed Inception."},
                                "status": "success",
                                "error": "",
                            }
                        ],
                        "final_synthesis": {
                            "rendered_plan": "[0] Question: Who directed Inception?\nAnswer: Christopher Nolan",
                            "memory": ["Christopher Nolan directed Inception."],
                            "evidence": "evidence block",
                            "prompt": {
                                "inputs": {"question": "Who directed the film?", "evidence": "evidence block"},
                                "raw_response": "Christopher Nolan",
                            },
                            "final_answer": "Christopher Nolan",
                            "status": "success",
                        },
                        "evaluation": {
                            "string_accuracy": 0.0,
                            "string_precision": 0.5,
                            "answer_accuracy": 0.0,
                        },
                    }
                ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_render_index_page_lists_runs(self) -> None:
        html = render_index_page(self.base_dir)
        self.assertIn("ConRAG Trace Browser", html)
        self.assertIn("example", html)
        self.assertIn("2026-06-03_00-00-00_000000", html)

    def test_render_run_page_shows_failure_filter_and_question(self) -> None:
        html = render_run_page(self.base_dir, "example", "2026-06-03_00-00-00_000000", failures_only=True)
        self.assertIn("Show all questions", html)
        self.assertIn("Who directed the film?", html)
        self.assertIn("FAILURE", html)

    def test_render_question_page_shows_trace_sections(self) -> None:
        html = render_question_page(self.base_dir, "example", "2026-06-03_00-00-00_000000", 0)
        self.assertIn("Decomposition", html)
        self.assertIn("Plan Steps", html)
        self.assertIn("Final Synthesis", html)
        self.assertIn("Christopher Nolan directed Inception.", html)


if __name__ == "__main__":
    unittest.main()
