# Docker Commands

`docker compose run --rm conrag-fanout-populate`
Populate `datasets/fanoutqa_first20/` from FanOutQA using the current `.env` settings.

`CONRAG_DATASET=fanoutqa_first20 docker compose run --rm conrag-build`
Build the knowledge graph and vector store for the generated FanOut dataset.

`CONRAG_DATASET=fanoutqa_first20 docker compose run --rm conrag-query`
Query the built graph, write `results.json`, `evaluation_report.json`, and `trace.json`.

`docker compose up conrag-debug`
Open the local debug webpage for browsing saved runs and inspecting traces at `http://localhost:8000`.
