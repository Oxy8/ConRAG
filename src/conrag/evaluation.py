from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from conrag.common import progress_bar, read_json, run_bounded, write_json
from conrag.prompts import ANSWER_EVALUATION_PROMPT

if TYPE_CHECKING:
    from conrag.clients import LLMClient
    from conrag.config import Config

logger = logging.getLogger(__name__)

type Score = tuple[float, float, float]
type JsonRecord = dict[str, object]
type EvaluationReport = dict[str, object]


class Evaluator:
    def __init__(self, config: Config, llm: LLMClient, results_path: Path) -> None:
        self.config = config
        self.llm = llm
        self.results_path = results_path
        self.records = [record for record in read_json(results_path, list) if isinstance(record, dict)]

    async def run(self) -> EvaluationReport:
        scores: list[Score | None] = [None] * len(self.records)

        with progress_bar(len(self.records), "Evaluation", "sample") as bar:

            async def process(index: int, record: JsonRecord) -> None:
                try:
                    scores[index] = await self.evaluate_record(record)
                finally:
                    bar.update(1)

            await run_bounded(self.records, self.config.max_workers, process)

        report = self.average([score for score in scores if score is not None])

        await asyncio.to_thread(write_json, self.results_path, self.records)
        await asyncio.to_thread(write_json, self.results_path.parent / "evaluation_report.json", report)
        logger.info("Evaluation complete: %s", report)

        return report

    async def evaluate_record(self, record: JsonRecord) -> Score:
        pred_answer = str(record.get("pred_answer", ""))
        gold_answer = str(record.get("gold_answer", ""))

        str_acc, str_prec = self.string_accuracy_and_precision(pred_answer, gold_answer)
        ans_acc = await self.judge(pred_answer, gold_answer)

        score = (str_acc, str_prec, ans_acc)

        record["string_accuracy"] = str_acc
        record["string_precision"] = str_prec
        record["answer_accuracy"] = ans_acc

        return score

    async def judge(self, pred_answer: str, gold_answer: str) -> float:
        if not pred_answer.strip():
            return 0.0
        try:
            response = await self.llm.infer(
                instructions=ANSWER_EVALUATION_PROMPT["instructions"],
                input_text=ANSWER_EVALUATION_PROMPT["input"].format(
                    pred_answer=pred_answer,
                    gold_answer=gold_answer,
                ),
            )
            return 1.0 if response.strip().lower() == "correct" else 0.0
        except Exception:
            logger.exception("LLM judge failed")
            return 0.0

    def string_accuracy_and_precision(self, pred_answer: object, gold_answer: object) -> tuple[float, float]:
        pred_answer = self.normalize_answer(str(pred_answer))
        gold_answer = self.normalize_answer(str(gold_answer))

        accuracy = 1.0 if pred_answer in gold_answer else 0.0

        pred_tokens = pred_answer.split()
        gold_tokens = gold_answer.split()

        if (pred_answer in ["yes", "no", "noanswer"] and pred_answer != gold_answer) or (
            gold_answer in ["yes", "no", "noanswer"] and pred_answer != gold_answer
        ):
            return accuracy, 0.0

        common_tokens = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common_tokens.values())

        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        return accuracy, precision

    @staticmethod
    def normalize_answer(text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = text.replace("-", " ")
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text

    @staticmethod
    def average(scores: list[Score]) -> EvaluationReport:
        if not scores:
            return {
                "string_accuracy": 0.0,
                "string_precision": 0.0,
                "answer_accuracy": 0.0,
            }

        avg_str_acc = sum(score[0] for score in scores) / len(scores)
        avg_str_prec = sum(score[1] for score in scores) / len(scores)
        avg_ans_acc = sum(score[2] for score in scores) / len(scores)

        return {
            "string_accuracy": avg_str_acc,
            "string_precision": avg_str_prec,
            "answer_accuracy": avg_ans_acc,
        }
