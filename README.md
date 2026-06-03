<h1 align="center">ConRAG: Consensus-Driven Multi-View Retrieval for Multi-Hop Question Answering</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.28093" target="_blank">
    <img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&style=flat-square" alt="arXiv:2605.28093">
  </a>
  <a href="https://github.com/yikai-zhu/ConRAG" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
</p>

This repository contains the implementation of ConRAG, a consensus-driven multi-view retrieval framework for multi-hop question answering. ConRAG improves both sides of retrieval-augmented generation: it builds an evidence-grounded graph on the corpus side, and it executes dependency-aware sub-questions on the query side. Retrieval is performed through multiple complementary views and fused in a unified evidence-unit space.

## 🎉 News

- **2026-05-28**: Released the ConRAG codebase.

## 🧩 Framework

![ConRAG framework](figures/framework.png)

ConRAG follows four stages:

- Offline construction: extracts triples, entities, and attributes from the corpus, then builds an evidence-grounded knowledge graph whose graph objects retain links to source passages.
- Online reasoning: decomposes the input question into dependency-aware sub-questions and executes them with slot-bound intermediate answers.
- Online retrieval: retrieves from relation, entity-anchor, and text-evidence views, maps all signals back to textual evidence units, and applies consensus ranking.
- Final generation: uses the execution trace and acquired information to produce the final answer.

The three core designs in the paper correspond to these stages:

- Connection grounds graph structures in verifiable evidence units.
- Constraint propagates intermediate answers as lightweight slot bindings for later retrieval steps.
- Consensus fuses relation, entity-anchor, and text-evidence signals within the same evidence-unit ranking space.

## ⚙️ Environment

Create a conda environment with Python 3.12.

```bash
conda create -n conrag python=3.12
conda activate conrag
python -m pip install -e .
```

This repository is configured for `google-genai` with **Vertex AI express mode**.
It is not configured for Google AI Studio and it does not use the standard
project/location ADC-style Vertex AI setup.

Create a local `.env` file from the tracked example:

```bash
cp .env.example .env
```

Then fill in at least:

```dotenv
CONRAG_LLM_MODEL=gemini-2.5-flash
CONRAG_VERTEX_API_KEY=your_vertex_express_mode_api_key
CONRAG_LLM_TIMEOUT_SECONDS=300
CONRAG_EMBEDDING_DEVICE=cpu
```

You can obtain the API key from Vertex AI **express mode**. The runtime reads
only the ConRAG-specific variables above and does not depend on `GOOGLE_API_KEY`,
`GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, or `GOOGLE_CLOUD_LOCATION`.

## 🚀 Quick Start

Run the included example dataset with 10 passages and 3 questions:

```bash
python -u main.py \
  --dataset example
```

You can also split the work by mode:

```bash
python -u main.py --dataset example --mode build
python -u main.py --dataset example --mode query
```

`build` creates or refreshes the persisted knowledge base under
`outputs/<dataset>/`. `query` requires an existing knowledge base unless you
also pass `--rebuild_knowledge_base true`.

## 📁 Data Preparation

This repository includes a small example dataset under `datasets/example/`, containing 10 passages and 3 questions.

`corpus.json` is a list of passages with `title` and `text` fields. `questions.json` is a list of question records with at least `question` and `answer` fields.

## ▶️ Run

Run the pipeline from `main.py`:

```bash
python -u main.py --dataset <dataset_name>
```

Other runtime fields are defined in `src/conrag/config.py`. LLM credentials and
the default model should be provided through `.env`.

## 🐳 Docker Compose

You can run the repo without installing Python packages locally.

Build the image:

```bash
docker compose build
```

Populate the FanOut-derived dataset from the first 20 FanOutQA dev questions
and the union of their required pages:

```bash
docker compose run --rm conrag-fanout-populate
```

This writes:

```text
datasets/fanoutqa_first20/corpus.json
datasets/fanoutqa_first20/questions.json
datasets/fanoutqa_first20/metadata.json
```

The FanOut workflow uses `fanoutqa.wiki_content(evidence)` to fetch only the
needed article revisions on demand, then splits each page into paragraph-based
merged chunks before writing `corpus.json`. It does not require the full
Wikipedia snapshot.

Run the full pipeline:

```bash
docker compose run --rm conrag
```

Build only the graph and vector store:

```bash
docker compose run --rm conrag-build
```

Query only using an existing knowledge base:

```bash
docker compose run --rm conrag-query
```

Start the trace debugger web app:

```bash
docker compose up conrag-debug
```

Then open:

```text
http://localhost:8000
```

To build and query the FanOut-derived dataset:

```bash
CONRAG_DATASET=fanoutqa_first20 docker compose run --rm conrag-build
CONRAG_DATASET=fanoutqa_first20 docker compose run --rm conrag-query
```

The default dataset inside Compose is `example`. To override it:

```bash
CONRAG_DATASET=my_dataset docker compose run --rm conrag-build
CONRAG_DATASET=my_dataset docker compose run --rm conrag-query
```

You can also use the main service directly with subcommands:

```bash
docker compose run --rm conrag build
docker compose run --rm conrag query
docker compose run --rm conrag run
```

## 📤 Outputs

The pipeline writes reusable knowledge-base artifacts to:

```text
outputs/<dataset_name>/
```

Each run writes prediction results, evaluation results, and logs to:

```text
results/<dataset_name>/<timestamp>/
```

Query and run modes also write an always-on debug trace to:

```text
results/<dataset_name>/<timestamp>/trace.json
```

The main result file is `results.json`; the aggregate score file is `evaluation_report.json`. The debug web app reads `trace.json` so you can inspect decomposition, per-step retrieval, dependency substitutions, intermediate answers, and final synthesis when a question fails.

## 🗂️ Code Structure

```text
📦 .
│-- 📂 datasets
│   └── 📂 example                # Small example dataset with 10 passages and 3 questions
│       ├── corpus.json           # Example corpus passages
│       └── questions.json        # Example question records
│-- 📂 figures
│   └── framework.png             # Framework figure from the paper
│-- 📂 src/conrag
│   ├── 📂 prompts                # Prompt templates and extraction schema
│   │   ├── __init__.py
│   │   ├── prompt_templates.py   # Prompt templates for extraction, planning, answering, and evaluation
│   │   └── schema.json           # Entity, relation, and attribute schema
│   ├── __init__.py
│   ├── clients.py                # LLM and embedding clients
│   ├── common.py                 # Shared utilities for logging, JSON, text cleaning, and concurrency
│   ├── config.py                 # Runtime configuration
│   ├── evaluation.py             # String-based and LLM-based evaluation
│   ├── graph.py                  # Evidence-grounded graph construction
│   ├── pipeline.py               # End-to-end pipeline orchestration
│   ├── planning.py               # Question planning data structures and rendering helpers
│   ├── retrieval.py              # Multi-view consensus retrieval and slot-bound execution
│   └── vector_store.py           # FAISS indices for graph and text views
│-- 📜 main.py                    # Pipeline entry point
│-- 📜 pyproject.toml             # Project metadata and dependencies
│-- 📜 README.md
│-- 📜 LICENSE                    # License file
│-- 📜 .gitignore                 # Files to exclude from Git
```

## 📚 Citation

If you find this work helpful, please consider citing us:

```bibtex
@article{zhu2026conrag,
  title={ConRAG: Consensus-Driven Multi-View Retrieval for Multi-Hop Question Answering},
  author={Zhu, Yikai and Chen, Kunfeng and Zhong, Qihuang and Liu, Juhua and Du, Bo},
  journal={arXiv preprint arXiv:2605.28093},
  year={2026},
}
```
