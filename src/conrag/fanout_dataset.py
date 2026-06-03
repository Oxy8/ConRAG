from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fanoutqa

from conrag.common import clean_text, write_json

if TYPE_CHECKING:
    from conrag.config import Config

logger = logging.getLogger(__name__)

FANOUT_DATASET_NAME = "fanoutqa_first20"
DEFAULT_FANOUT_SAMPLE_COUNT = 20
DEFAULT_CHUNK_TARGET_CHARS = 2_000
DEFAULT_CHUNK_SOFT_MAX_CHARS = 3_000
DEFAULT_MIN_CHUNK_CHARS = 500
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_TITLE_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True, frozen=True)
class EvidenceRecord:
    pageid: int | None
    revid: int | None
    title: str
    url: str


def build_fanout_dataset(config: Config) -> Path:
    dataset_dir = config.base_dir / "datasets" / FANOUT_DATASET_NAME
    records = list(fanoutqa.load_dev())
    selected = records[:config.fanout_sample_count]
    logger.info("Selected %d FanOutQA dev questions", len(selected))

    evidence_index: dict[tuple[int | None, int | None, str], dict[str, object]] = {}
    question_rows: list[dict[str, object]] = []

    for index, record in enumerate(selected):
        question = clean_text(extract_attr(record, "question"))
        answer = clean_text(extract_attr(record, "answer"))
        decomposition = serialize_json_value(extract_attr(record, "decomposition"))
        necessary_evidence = list(extract_attr(record, "necessary_evidence"))
        evidence_refs: list[dict[str, object]] = []

        for evidence in necessary_evidence:
            parsed = parse_evidence(evidence)
            evidence_key = (parsed.pageid, parsed.revid, parsed.title)
            if evidence_key not in evidence_index:
                logger.info("Fetching FanOutQA evidence page for question %d: %s", index, parsed.title)
                content = normalize_page_text(fanoutqa.wiki_content(evidence))
                if not content:
                    raise RuntimeError(f"Fetched empty FanOutQA evidence for page {parsed.title!r}")
                chunks = build_page_chunks(parsed, content, config)
                if not chunks:
                    raise RuntimeError(f"Chunking produced no usable content for page {parsed.title!r}")
                logger.info(
                    "Chunked FanOutQA evidence page: %s (question=%d, chunks=%d, chunk_ids=%s)",
                    parsed.title,
                    index,
                    len(chunks),
                    summarize_chunk_ids(chunks),
                )
                evidence_index[evidence_key] = {
                    "title": parsed.title,
                    "text": content,
                    "chunk_rows": chunks,
                    "pageid": parsed.pageid,
                    "revid": parsed.revid,
                    "url": parsed.url,
                    "question_indices": [index],
                }
            else:
                append_unique(evidence_index[evidence_key]["question_indices"], index)

            evidence_refs.append({
                "pageid": parsed.pageid,
                "revid": parsed.revid,
                "title": parsed.title,
                "url": parsed.url,
            })

        question_rows.append({
            "fanout_index": index,
            "question": question,
            "answer": answer,
            "decomposition": decomposition,
            "required_evidence": evidence_refs,
        })

    sorted_pages = sorted(evidence_index.values(), key=lambda row: str(row["title"]).lower())
    corpus_rows: list[dict[str, object]] = []
    page_entries: list[dict[str, object]] = []
    page_chunk_ids: dict[tuple[int | None, int | None, str], list[str]] = {}

    for item in sorted_pages:
        chunk_rows = list(expect_list(item["chunk_rows"], "chunk_rows"))
        chunk_ids = [str(chunk["chunk_id"]) for chunk in chunk_rows]
        page_key = (expect_optional_int(item["pageid"]), expect_optional_int(item["revid"]), str(item["title"]))
        page_chunk_ids[page_key] = chunk_ids
        corpus_rows.extend(chunk_rows)
        page_entries.append({
            "title": str(item["title"]),
            "pageid": item["pageid"],
            "revid": item["revid"],
            "url": item["url"],
            "question_indices": list(expect_list(item["question_indices"], "question_indices")),
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
        })

    for row in question_rows:
        required_chunk_ids: list[str] = []
        for evidence_ref in expect_list(row["required_evidence"], "required_evidence"):
            if not isinstance(evidence_ref, dict):
                raise TypeError("required_evidence entries must be objects")
            page_key = (
                parse_optional_int(evidence_ref.get("pageid")),
                parse_optional_int(evidence_ref.get("revid")),
                clean_text(evidence_ref.get("title", "")),
            )
            for chunk_id in page_chunk_ids.get(page_key, []):
                append_unique(required_chunk_ids, chunk_id)
        row["required_chunk_ids"] = required_chunk_ids

    corpus_token_count = sum(estimate_chunk_tokens(row["title"], row["text"]) for row in corpus_rows)

    metadata = {
        "source": "FanOutQA dev",
        "dataset_name": FANOUT_DATASET_NAME,
        "selected_question_indices": list(range(len(selected))),
        "question_count": len(question_rows),
        "corpus_page_count": len(page_entries),
        "corpus_chunk_count": len(corpus_rows),
        "corpus_token_count": corpus_token_count,
        "pages": page_entries,
        "corpus_entries": [
            {
                "chunk_id": row["chunk_id"],
                "title": row["title"],
                "source_title": row["source_title"],
                "pageid": row["pageid"],
                "revid": row["revid"],
                "chunk_index": row["chunk_index"],
            }
            for row in corpus_rows
        ],
        "questions": [
            {
                "fanout_index": row["fanout_index"],
                "question": row["question"],
                "required_evidence": row["required_evidence"],
                "required_chunk_ids": row["required_chunk_ids"],
            }
            for row in question_rows
        ],
    }

    write_json(dataset_dir / "corpus.json", corpus_rows)
    write_json(dataset_dir / "questions.json", question_rows)
    write_json(dataset_dir / "metadata.json", metadata)
    logger.info(
        "Wrote FanOutQA-derived dataset to %s (pages=%d, chunks=%d, questions=%d, tokens=%d)",
        dataset_dir,
        len(page_entries),
        len(corpus_rows),
        len(question_rows),
        corpus_token_count,
    )
    return dataset_dir


def build_page_chunks(evidence: EvidenceRecord, content: str, config: Config) -> list[dict[str, object]]:
    chunk_texts = chunk_page_content(
        content,
        target_chars=config.fanout_chunk_target_chars,
        soft_max_chars=config.fanout_chunk_soft_max_chars,
        min_chunk_chars=config.fanout_min_chunk_chars,
    )
    chunk_rows: list[dict[str, object]] = []
    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk_rows.append({
            "chunk_id": make_chunk_id(evidence, chunk_index),
            "title": evidence.title,
            "text": chunk_text,
            "pageid": evidence.pageid,
            "revid": evidence.revid,
            "source_title": evidence.title,
            "chunk_index": chunk_index,
        })
    return chunk_rows


def summarize_chunk_ids(chunks: list[dict[str, object]]) -> str:
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if not chunk_ids:
        return "none"
    if len(chunk_ids) == 1:
        return chunk_ids[0]
    return f"{chunk_ids[0]}..{chunk_ids[-1]}"


def chunk_page_content(content: str, *, target_chars: int, soft_max_chars: int, min_chunk_chars: int) -> list[str]:
    paragraph_blocks = split_paragraph_blocks(content, soft_max_chars=soft_max_chars)
    merged_chunks = merge_blocks(paragraph_blocks, target_chars, soft_max_chars)
    if len(merged_chunks) >= 2 and len(merged_chunks[-1]) < min_chunk_chars:
        previous = merged_chunks[-2]
        last = merged_chunks[-1]
        if len(previous) + 2 + len(last) <= soft_max_chars:
            merged_chunks[-2] = "\n\n".join((previous, last))
            merged_chunks.pop()
    return merged_chunks


def split_paragraph_blocks(content: str, *, soft_max_chars: int = DEFAULT_CHUNK_SOFT_MAX_CHARS) -> list[str]:
    blocks: list[str] = []
    for raw_block in _BLANK_LINE_RE.split(content):
        block = normalize_block(raw_block)
        if not is_meaningful_block(block):
            continue
        if len(block) > soft_max_chars:
            blocks.extend(split_oversized_block(block, soft_max_chars=soft_max_chars))
        else:
            blocks.append(block)
    return blocks


def split_oversized_block(block: str, *, soft_max_chars: int = DEFAULT_CHUNK_SOFT_MAX_CHARS) -> list[str]:
    sentences = [fragment.strip() for fragment in _SENTENCE_BOUNDARY_RE.split(block) if fragment.strip()]
    if len(sentences) <= 1:
        return force_split_block(block, soft_max_chars)

    blocks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > soft_max_chars:
            if current_parts:
                blocks.append(" ".join(current_parts))
                current_parts = []
                current_len = 0
            blocks.extend(force_split_block(sentence, soft_max_chars))
            continue

        prospective_len = len(sentence) if not current_parts else current_len + 1 + len(sentence)
        if current_parts and prospective_len > soft_max_chars:
            blocks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_len = len(sentence)
        else:
            current_parts.append(sentence)
            current_len = prospective_len

    if current_parts:
        blocks.append(" ".join(current_parts))
    return blocks


def force_split_block(block: str, max_chars: int) -> list[str]:
    words = block.split()
    if not words:
        return []
    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0
    for word in words:
        word_len = len(word)
        if current_words and current_len + 1 + word_len > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len
        else:
            prospective_len = word_len if not current_words else current_len + 1 + word_len
            current_words.append(word)
            current_len = prospective_len
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def merge_blocks(blocks: list[str], target_chars: int, soft_max_chars: int) -> list[str]:
    merged: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for block in blocks:
        if not current_parts:
            current_parts = [block]
            current_len = len(block)
            continue

        prospective_len = current_len + 2 + len(block)
        if current_len < target_chars and prospective_len <= soft_max_chars:
            current_parts.append(block)
            current_len = prospective_len
            continue

        merged.append("\n\n".join(current_parts))
        current_parts = [block]
        current_len = len(block)

    if current_parts:
        merged.append("\n\n".join(current_parts))
    return merged


def normalize_page_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def normalize_block(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return _SPACE_RE.sub(" ", " ".join(lines)).strip()


def is_meaningful_block(block: str) -> bool:
    if not block:
        return False
    alnum = _NON_WORD_RE.sub("", block)
    return len(alnum) >= 3


def make_chunk_id(evidence: EvidenceRecord, chunk_index: int) -> str:
    pageid_label = "na" if evidence.pageid is None else str(evidence.pageid)
    revid_label = "na" if evidence.revid is None else str(evidence.revid)
    title_slug = _TITLE_SLUG_RE.sub("-", evidence.title.lower()).strip("-") or "page"
    return f"fanout-{pageid_label}-{revid_label}-{title_slug}-{chunk_index:03d}"


def estimate_chunk_tokens(title: str, text: str) -> int:
    chunk_text = f"{title}: {text}".strip()
    if not chunk_text:
        return 0
    # Use a simple chars-per-token estimate so reporting works without a model-specific tokenizer.
    return math.ceil(len(chunk_text) / 4)


def serialize_json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): serialize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_json_value(item) for item in value]
    if is_dataclass(value):
        return serialize_json_value(asdict(value))
    if hasattr(value, "_asdict"):
        asdict_method = getattr(value, "_asdict")
        if callable(asdict_method):
            return serialize_json_value(asdict_method())
    object_dict = object_to_dict(value)
    if object_dict is not None:
        return serialize_json_value(object_dict)
    return clean_text(value)


def object_to_dict(value: object) -> dict[str, object] | None:
    if hasattr(value, "__dict__"):
        return {
            str(key): item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    if slots:
        result: dict[str, object] = {}
        for name in slots:
            if not isinstance(name, str) or name.startswith("_") or not hasattr(value, name):
                continue
            result[name] = getattr(value, name)
        if result:
            return result
    return None


def extract_attr(record: object, name: str) -> Any:
    if hasattr(record, name):
        return getattr(record, name)
    if isinstance(record, dict) and name in record:
        return record[name]
    raise AttributeError(f"FanOutQA record is missing attribute {name!r}")


def parse_evidence(evidence: object) -> EvidenceRecord:
    return EvidenceRecord(
        pageid=extract_optional_int(evidence, "pageid"),
        revid=extract_optional_int(evidence, "revid"),
        title=clean_text(extract_evidence_attr(evidence, "title")),
        url=str(extract_evidence_attr(evidence, "url")),
    )


def extract_evidence_attr(evidence: object, name: str) -> object:
    if hasattr(evidence, name):
        return getattr(evidence, name)
    if isinstance(evidence, dict) and name in evidence:
        return evidence[name]
    raise AttributeError(f"FanOutQA evidence is missing attribute {name!r}")


def extract_optional_int(evidence: object, name: str) -> int | None:
    value = extract_evidence_attr(evidence, name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"FanOutQA evidence field {name!r} must be an integer or null")
    return value


def append_unique(items: object, value: object) -> None:
    if not isinstance(items, list):
        raise TypeError("Expected a list")
    if value not in items:
        items.append(value)


def expect_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"Expected {name} to be a list")
    return value


def expect_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Expected an integer or null")
    return value


def parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Expected an integer or null")
    if isinstance(value, int):
        return value
    raise TypeError("Expected an integer or null")
