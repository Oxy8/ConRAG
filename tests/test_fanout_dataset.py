from __future__ import annotations

import importlib
import sys
import tempfile
import types as pytypes
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def make_evidence(pageid: int, revid: int, title: str) -> SimpleNamespace:
    return SimpleNamespace(pageid=pageid, revid=revid, title=title, url=f"https://example.test/{pageid}")


class FakeDevSubquestion:
    __slots__ = ("question", "answer", "supporting_facts")

    def __init__(self, question: str, answer: str, supporting_facts: list[str]) -> None:
        self.question = question
        self.answer = answer
        self.supporting_facts = supporting_facts


class FanOutDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def load_modules(self) -> tuple[object, object]:
        fake_tqdm = pytypes.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda *args, **kwargs: SimpleNamespace(update=lambda n: None)

        fake_fanoutqa = pytypes.ModuleType("fanoutqa")
        fake_fanoutqa.load_dev = lambda: []
        fake_fanoutqa.wiki_content = lambda evidence: ""

        with patch.dict(
            sys.modules,
            {
                "fanoutqa": fake_fanoutqa,
                "tqdm": fake_tqdm,
            },
        ):
            sys.modules.pop("conrag.common", None)
            sys.modules.pop("conrag.fanout_dataset", None)
            common = importlib.import_module("conrag.common")
            fanout_dataset = importlib.import_module("conrag.fanout_dataset")
            return importlib.reload(common), importlib.reload(fanout_dataset)

    def make_config(self, **overrides: object) -> object:
        from conrag.config import Config

        return Config(base_dir=Path(self._tmpdir.name), vertex_api_key="unused-for-prep", **overrides)

    def test_split_paragraph_blocks_removes_noise_and_splits_large_blocks(self) -> None:
        common, fanout_dataset = self.load_modules()
        oversized_sentence = " ".join(["sentence"] * (fanout_dataset.DEFAULT_CHUNK_SOFT_MAX_CHARS // 4 + 50))
        content = (
            "\n\n"
            "Overview heading\n\n"
            "!!!\n\n"
            "First paragraph with enough content to keep.\n\n"
            f"{oversized_sentence}.\n\n"
            "Second paragraph with enough content to keep.\n"
        )

        blocks = fanout_dataset.split_paragraph_blocks(content)

        self.assertGreaterEqual(len(blocks), 4)
        self.assertNotIn("!!!", blocks)
        self.assertTrue(all(len(block) <= fanout_dataset.DEFAULT_CHUNK_SOFT_MAX_CHARS for block in blocks))
        self.assertEqual(blocks[0], "Overview heading")

    def test_build_fanout_dataset_writes_chunked_corpus_and_metadata(self) -> None:
        common, fanout_dataset = self.load_modules()
        shared_paragraph = " ".join(["shared"] * 220)
        shared_content = "\n\n".join([
            shared_paragraph,
            shared_paragraph,
            shared_paragraph,
        ])

        records = [
            SimpleNamespace(
                question=f"Question {index}",
                answer=f"Answer {index}",
                decomposition=[
                    FakeDevSubquestion(
                        question=f"sub-question {index}",
                        answer=f"sub-answer {index}",
                        supporting_facts=[f"fact {index}"],
                    )
                ],
                necessary_evidence=[
                    make_evidence(1, 11, "Shared Page"),
                    make_evidence(index + 2, index + 22, f"Unique Page {index}"),
                ],
            )
            for index in range(fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT)
        ]

        fetched_titles: list[str] = []

        def fake_wiki_content(evidence: object) -> str:
            title = str(getattr(evidence, "title"))
            fetched_titles.append(title)
            if title == "Shared Page":
                return shared_content
            return f"Summary for {title}\n\nDetails for {title}"

        with patch.object(fanout_dataset.fanoutqa, "load_dev", return_value=records), patch.object(
            fanout_dataset.fanoutqa,
            "wiki_content",
            side_effect=fake_wiki_content,
        ):
            dataset_dir = fanout_dataset.build_fanout_dataset(self.make_config())

        corpus = common.read_json(dataset_dir / "corpus.json", list)
        questions = common.read_json(dataset_dir / "questions.json", list)
        metadata = common.read_json(dataset_dir / "metadata.json", dict)

        self.assertEqual(dataset_dir.name, fanout_dataset.FANOUT_DATASET_NAME)
        self.assertEqual(len(questions), fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT)
        self.assertGreater(len(corpus), fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT + 1)
        self.assertEqual(len(fetched_titles), fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT + 1)
        self.assertEqual(metadata["question_count"], fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT)
        self.assertEqual(metadata["corpus_page_count"], fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT + 1)
        self.assertEqual(metadata["corpus_chunk_count"], len(corpus))
        expected_token_count = sum(
            fanout_dataset.estimate_chunk_tokens(row["title"], row["text"])
            for row in corpus
        )
        self.assertEqual(metadata["corpus_token_count"], expected_token_count)
        self.assertTrue(all("chunk_id" in row for row in corpus))
        self.assertTrue(all("source_title" in row for row in corpus))
        self.assertEqual(questions[0]["fanout_index"], 0)
        self.assertEqual(questions[0]["required_evidence"][0]["title"], "Shared Page")
        self.assertEqual(questions[0]["decomposition"][0]["question"], "sub-question 0")
        self.assertEqual(questions[0]["decomposition"][0]["supporting_facts"], ["fact 0"])
        self.assertGreaterEqual(len(questions[0]["required_chunk_ids"]), 3)
        shared_page_entry = next(page for page in metadata["pages"] if page["title"] == "Shared Page")
        self.assertGreaterEqual(shared_page_entry["chunk_count"], 2)
        self.assertEqual(shared_page_entry["chunk_count"], len(shared_page_entry["chunk_ids"]))
        metadata_question = metadata["questions"][0]
        self.assertEqual(
            metadata_question["required_chunk_ids"],
            questions[0]["required_chunk_ids"],
        )
        self.assertTrue(
            set(shared_page_entry["chunk_ids"]).issubset(set(questions[0]["required_chunk_ids"]))
        )

    def test_chunked_dataset_remains_compatible_with_pipeline_loader(self) -> None:
        common, fanout_dataset = self.load_modules()
        record = SimpleNamespace(
            question="Question 0",
            answer="Answer 0",
            decomposition=[FakeDevSubquestion("step 0", "answer 0", ["fact 0"])],
            necessary_evidence=[make_evidence(1, 11, "Shared Page")],
        )

        with patch.object(
            fanout_dataset.fanoutqa,
            "load_dev",
            return_value=[record] * fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT,
        ), patch.object(
            fanout_dataset.fanoutqa,
            "wiki_content",
            return_value="Paragraph one.\n\nParagraph two with more text.",
        ):
            dataset_dir = fanout_dataset.build_fanout_dataset(self.make_config())

        common.read_json(dataset_dir / "corpus.json", list)
        common.read_json(dataset_dir / "questions.json", list)

        from conrag.config import Config
        fake_nanoid = pytypes.ModuleType("nanoid")
        fake_nanoid.generate = lambda size=8: "deadbeef"

        fake_networkx = pytypes.ModuleType("networkx")
        fake_networkx.MultiDiGraph = object

        fake_clients = pytypes.ModuleType("conrag.clients")
        fake_clients.EmbeddingClient = object
        fake_clients.LLMClient = object

        fake_evaluation = pytypes.ModuleType("conrag.evaluation")
        fake_evaluation.Evaluator = object

        fake_graph = pytypes.ModuleType("conrag.graph")
        fake_graph.GraphBuilder = object

        fake_retrieval = pytypes.ModuleType("conrag.retrieval")
        fake_retrieval.AnswerWithTrace = object
        fake_retrieval.RetrievalEngine = object

        fake_vector_store = pytypes.ModuleType("conrag.vector_store")
        fake_vector_store.VectorStore = object

        fake_tqdm = pytypes.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda *args, **kwargs: SimpleNamespace(update=lambda n: None)

        with patch.dict(
            sys.modules,
            {
                "nanoid": fake_nanoid,
                "networkx": fake_networkx,
                "tqdm": fake_tqdm,
                "conrag.clients": fake_clients,
                "conrag.evaluation": fake_evaluation,
                "conrag.graph": fake_graph,
                "conrag.retrieval": fake_retrieval,
                "conrag.vector_store": fake_vector_store,
            },
        ):
            sys.modules.pop("conrag.pipeline", None)
            pipeline = importlib.import_module("conrag.pipeline")
            chunks, questions = pipeline.load_dataset(
                Config(
                    base_dir=Path(self._tmpdir.name),
                    dataset=fanout_dataset.FANOUT_DATASET_NAME,
                    vertex_api_key="unused-for-prep",
                )
            )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(len(questions), fanout_dataset.DEFAULT_FANOUT_SAMPLE_COUNT)
        self.assertTrue(all(isinstance(chunk, str) and chunk for chunk in chunks))

    def test_build_fanout_dataset_uses_configured_sample_count(self) -> None:
        common, fanout_dataset = self.load_modules()
        records = [
            SimpleNamespace(
                question=f"Question {index}",
                answer=f"Answer {index}",
                decomposition=[FakeDevSubquestion(f"step {index}", f"answer {index}", [f"fact {index}"])],
                necessary_evidence=[make_evidence(index + 1, index + 101, f"Page {index}")],
            )
            for index in range(6)
        ]

        with patch.object(fanout_dataset.fanoutqa, "load_dev", return_value=records), patch.object(
            fanout_dataset.fanoutqa,
            "wiki_content",
            side_effect=lambda evidence: f"Paragraph for {getattr(evidence, 'title')}",
        ):
            dataset_dir = fanout_dataset.build_fanout_dataset(
                self.make_config(fanout_sample_count=3)
            )

        questions = common.read_json(dataset_dir / "questions.json", list)
        metadata = common.read_json(dataset_dir / "metadata.json", dict)
        self.assertEqual(len(questions), 3)
        self.assertEqual(metadata["question_count"], 3)
        self.assertEqual(metadata["selected_question_indices"], [0, 1, 2])

    def test_build_fanout_dataset_uses_configured_chunk_sizes(self) -> None:
        common, fanout_dataset = self.load_modules()
        paragraph = " ".join(["alpha"] * 120)
        records = [
            SimpleNamespace(
                question="Question 0",
                answer="Answer 0",
                decomposition=[FakeDevSubquestion("step 0", "answer 0", ["fact 0"])],
                necessary_evidence=[make_evidence(1, 101, "Tunable Page")],
            )
        ]

        with patch.object(
            fanout_dataset.fanoutqa,
            "load_dev",
            return_value=records,
        ), patch.object(
            fanout_dataset.fanoutqa,
            "wiki_content",
            return_value="\n\n".join([paragraph, paragraph, paragraph, paragraph]),
        ):
            dataset_dir = fanout_dataset.build_fanout_dataset(
                self.make_config(
                    fanout_sample_count=1,
                    fanout_chunk_target_chars=500,
                    fanout_chunk_soft_max_chars=700,
                    fanout_min_chunk_chars=150,
                )
            )

        corpus = common.read_json(dataset_dir / "corpus.json", list)
        metadata = common.read_json(dataset_dir / "metadata.json", dict)
        self.assertGreaterEqual(len(corpus), 4)
        self.assertEqual(metadata["corpus_chunk_count"], len(corpus))
        self.assertTrue(all(len(f"{row['title']}: {row['text']}") <= 750 for row in corpus))


if __name__ == "__main__":
    unittest.main()
