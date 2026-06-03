from __future__ import annotations

import argparse
import html
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Browse ConRAG query traces")
    parser.add_argument("--base_dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--host", default=os.getenv("CONRAG_DEBUG_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CONRAG_DEBUG_PORT", "8000")))
    args = parser.parse_args()

    serve(Path(args.base_dir), args.host, args.port)
    return 0


def serve(base_dir: Path, host: str, port: int) -> None:
    handler = make_handler(base_dir)
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("Starting ConRAG debug web app at http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping ConRAG debug web app")
    finally:
        server.server_close()


def make_handler(base_dir: Path) -> type[BaseHTTPRequestHandler]:
    class DebugHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    content = render_index_page(base_dir)
                elif parsed.path == "/run":
                    dataset = require_param(params, "dataset")
                    run_name = require_param(params, "run")
                    failures_only = params.get("failures", ["0"])[0] == "1"
                    content = render_run_page(base_dir, dataset, run_name, failures_only=failures_only)
                elif parsed.path == "/question":
                    dataset = require_param(params, "dataset")
                    run_name = require_param(params, "run")
                    index = int(require_param(params, "index"))
                    content = render_question_page(base_dir, dataset, run_name, index)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Page not found")
                    return
                encoded = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except FileNotFoundError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception:
                logger.exception("Debug web request failed")
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error")

        def log_message(self, format: str, *args: object) -> None:
            logger.info("debug_web | " + format, *args)

    return DebugHandler


def render_index_page(base_dir: Path) -> str:
    dataset_rows: list[str] = []
    for dataset, runs in discover_runs(base_dir).items():
        run_rows = "".join(render_run_summary_row(dataset, run) for run in runs)
        dataset_rows.append(f"""
        <section class="dataset-card">
          <h2>{escape(dataset)}</h2>
          <table>
            <thead>
              <tr><th>Run</th><th>Questions</th><th>Evaluation</th><th>Open</th></tr>
            </thead>
            <tbody>{run_rows}</tbody>
          </table>
        </section>
        """)
    body = "".join(dataset_rows) if dataset_rows else "<p>No traced runs found under <code>results/</code>.</p>"
    return html_page("ConRAG Trace Browser", f"<h1>ConRAG Trace Browser</h1>{body}")


def render_run_page(base_dir: Path, dataset: str, run_name: str, *, failures_only: bool) -> str:
    payload = load_trace_payload(base_dir, dataset, run_name)
    questions = [item for item in payload.get("questions", []) if isinstance(item, dict)]
    if failures_only:
        questions = [item for item in questions if question_has_failure(item)]

    toggle_params = {"dataset": dataset, "run": run_name}
    toggle_label = "Show all questions" if failures_only else "Show failures only"
    if not failures_only:
        toggle_params["failures"] = "1"

    rows = "".join(
        render_question_summary_row(dataset, run_name, index, question)
        for index, question in enumerate(questions)
    )
    evaluation = payload.get("evaluation_report", {})
    body = f"""
    <nav><a href="/">All runs</a></nav>
    <h1>Run {escape(dataset)} / {escape(run_name)}</h1>
    <p><strong>Questions:</strong> {len([item for item in payload.get("questions", []) if isinstance(item, dict)])}</p>
    {render_evaluation_summary(evaluation)}
    <p><a href="/run?{urlencode(toggle_params)}">{escape(toggle_label)}</a></p>
    <table>
      <thead>
        <tr><th>#</th><th>Status</th><th>Question</th><th>Gold</th><th>Predicted</th><th>Open</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """
    return html_page(f"ConRAG Run {dataset}/{run_name}", body)


def render_question_page(base_dir: Path, dataset: str, run_name: str, index: int) -> str:
    payload = load_trace_payload(base_dir, dataset, run_name)
    questions = [item for item in payload.get("questions", []) if isinstance(item, dict)]
    if index < 0 or index >= len(questions):
        raise FileNotFoundError(f"Question index {index} not found in run {run_name}")
    question = questions[index]
    decomposition = as_dict(question.get("decomposition"))
    steps = [item for item in question.get("steps", []) if isinstance(item, dict)]
    final_synthesis = as_dict(question.get("final_synthesis"))
    evaluation = as_dict(question.get("evaluation"))
    body = f"""
    <nav>
      <a href="/">All runs</a>
      <span> / </span>
      <a href="/run?{urlencode({'dataset': dataset, 'run': run_name})}">{escape(dataset)} / {escape(run_name)}</a>
    </nav>
    <h1>Question {index}</h1>
    {render_question_overview(question, evaluation)}
    <section class="trace-section">
      <h2>Decomposition</h2>
      {render_decomposition_section(decomposition)}
    </section>
    <section class="trace-section">
      <h2>Plan Steps</h2>
      {''.join(render_step_card(step) for step in steps) or '<p>No steps captured.</p>'}
    </section>
    <section class="trace-section">
      <h2>Final Synthesis</h2>
      {render_final_synthesis_section(final_synthesis)}
    </section>
    """
    return html_page(f"ConRAG Question {index}", body)


def discover_runs(base_dir: Path) -> dict[str, list[dict[str, object]]]:
    results_dir = base_dir / "results"
    datasets: dict[str, list[dict[str, object]]] = {}
    if not results_dir.exists():
        return datasets
    for dataset_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        runs: list[dict[str, object]] = []
        for run_dir in sorted((path for path in dataset_dir.iterdir() if path.is_dir()), reverse=True):
            trace_path = run_dir / "trace.json"
            if not trace_path.exists():
                continue
            payload = load_trace_payload(base_dir, dataset_dir.name, run_dir.name)
            runs.append({
                "name": run_dir.name,
                "question_count": int(payload.get("question_count", 0)),
                "evaluation_report": as_dict(payload.get("evaluation_report")),
            })
        if runs:
            datasets[dataset_dir.name] = runs
    return datasets


def load_trace_payload(base_dir: Path, dataset: str, run_name: str) -> dict[str, object]:
    trace_path = base_dir / "results" / dataset / run_name / "trace.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{trace_path} must contain a JSON object")
    return payload


def render_run_summary_row(dataset: str, run: dict[str, object]) -> str:
    evaluation = as_dict(run.get("evaluation_report"))
    return f"""
    <tr>
      <td>{escape(str(run.get('name', 'unknown')))}</td>
      <td>{int(run.get('question_count', 0))}</td>
      <td>{escape(format_eval_brief(evaluation))}</td>
      <td><a href="/run?{urlencode({'dataset': dataset, 'run': str(run.get('name', ''))})}">Open</a></td>
    </tr>
    """


def render_question_summary_row(dataset: str, run_name: str, index: int, question: dict[str, object]) -> str:
    status = "failure" if question_has_failure(question) else "ok"
    return f"""
    <tr>
      <td>{index}</td>
      <td><span class="status {status}">{escape(status.upper())}</span></td>
      <td>{escape(str(question.get('question', '')))}</td>
      <td>{escape(str(question.get('gold_answer', '')))}</td>
      <td>{escape(str(question.get('pred_answer', '')))}</td>
      <td><a href="/question?{urlencode({'dataset': dataset, 'run': run_name, 'index': index})}">Inspect</a></td>
    </tr>
    """


def render_question_overview(question: dict[str, object], evaluation: dict[str, object]) -> str:
    return f"""
    <section class="summary-card">
      <p><strong>Question:</strong> {escape(str(question.get('question', '')))}</p>
      <p><strong>Gold answer:</strong> {escape(str(question.get('gold_answer', '')))}</p>
      <p><strong>Predicted answer:</strong> {escape(str(question.get('pred_answer', '')))}</p>
      <p><strong>Status:</strong> {escape(str(question.get('status', 'unknown')))}</p>
      {render_evaluation_summary(evaluation)}
    </section>
    """


def render_decomposition_section(trace: dict[str, object]) -> str:
    retrieval = as_dict(trace.get("retrieval"))
    prompt = as_dict(trace.get("prompt"))
    output = as_dict(trace.get("output"))
    return "".join([
        render_retrieval_trace(retrieval),
        render_kv_block("Prompt Inputs", as_dict(prompt.get("inputs"))),
        render_pre_block("Raw Response", str(prompt.get("raw_response", ""))),
        render_kv_block("Parsed Payload", as_dict(prompt.get("parsed_payload"))),
        render_kv_block("Output", output),
    ])


def render_step_card(step: dict[str, object]) -> str:
    retrieval = as_dict(step.get("retrieval"))
    prompt = as_dict(step.get("prompt"))
    result = as_dict(step.get("result"))
    return f"""
    <details class="step-card" open>
      <summary>Step {escape(str(step.get('step_id', '?')))}: {escape(str(step.get('resolved_sub_question', step.get('raw_sub_question', ''))))}</summary>
      <p><strong>Raw subquestion:</strong> {escape(str(step.get('raw_sub_question', '')))}</p>
      <p><strong>Resolved subquestion:</strong> {escape(str(step.get('resolved_sub_question', '')))}</p>
      {render_kv_block("Dependencies", {'dependencies': step.get('dependencies', []), 'dependency_answers': step.get('dependency_answers', {})})}
      {render_list_block("Memory Before", step.get("memory_before", []))}
      {render_retrieval_trace(retrieval)}
      {render_kv_block("Prompt Inputs", as_dict(prompt.get("inputs")))}
      {render_pre_block("Raw Response", str(prompt.get("raw_response", "")))}
      {render_kv_block("Parsed Payload", as_dict(prompt.get("parsed_payload")))}
      {render_kv_block("Result", result)}
      {render_error_block(str(step.get('error', '')))}
    </details>
    """


def render_final_synthesis_section(trace: dict[str, object]) -> str:
    prompt = as_dict(trace.get("prompt"))
    return "".join([
        render_pre_block("Rendered Plan", str(trace.get("rendered_plan", ""))),
        render_list_block("Memory", trace.get("memory", [])),
        render_pre_block("Final Evidence", str(trace.get("evidence", ""))),
        render_kv_block("Prompt Inputs", as_dict(prompt.get("inputs"))),
        render_pre_block("Raw Response", str(prompt.get("raw_response", ""))),
        render_error_block(str(prompt.get("error", ""))),
        f"<p><strong>Final answer:</strong> {escape(str(trace.get('final_answer', '')))}</p>",
        f"<p><strong>Status:</strong> {escape(str(trace.get('status', 'unknown')))}</p>",
    ])


def render_retrieval_trace(trace: dict[str, object]) -> str:
    return f"""
    <details class="trace-card" open>
      <summary>Retrieval Trace</summary>
      <p><strong>Query:</strong> {escape(str(trace.get('query_text', '')))}</p>
      {render_table('Selected Hits', list_of_dicts(trace.get('selected_hits')), ['chunk_id', 'score', 'text'])}
      {render_table('Fusion Scores', list_of_dicts(trace.get('fusion_scores')), ['chunk_id', 'final_score', 'direct_score', 'complement_score', 'chunk_score', 'support_nodes', 'text'])}
      {render_table('Direct Chunk Scores', list_of_dicts(trace.get('direct_chunk_scores')), ['chunk_id', 'score', 'support_nodes', 'text'])}
      {render_table('Complement Chunk Scores', list_of_dicts(trace.get('complement_chunk_scores')), ['chunk_id', 'score', 'support_nodes', 'text'])}
      {render_table('Chunk Hits', list_of_dicts(trace.get('chunk_hits')), ['chunk_id', 'score', 'text'])}
      {render_table('Relation Hits', list_of_dicts(trace.get('relation_hits')), ['source_name', 'relation', 'target_name', 'score', 'source_chunks'])}
      {render_table('Anchor Hits', list_of_dicts(trace.get('anchor_hits')), ['name', 'type', 'score', 'source_chunks'])}
      {render_kv_block('Normalized Scores', as_dict(trace.get('normalized_scores')))}
      {render_pre_block('Rendered Context', str(trace.get('rendered_context', '')))}
    </details>
    """


def render_evaluation_summary(evaluation: dict[str, object]) -> str:
    if not evaluation:
        return ""
    return f"""
    <div class="eval-box">
      <strong>Evaluation:</strong>
      <span>string_accuracy={escape(str(evaluation.get('string_accuracy', '')))}</span>
      <span>string_precision={escape(str(evaluation.get('string_precision', '')))}</span>
      <span>answer_accuracy={escape(str(evaluation.get('answer_accuracy', '')))}</span>
    </div>
    """


def render_kv_block(title: str, value: dict[str, object]) -> str:
    if not value:
        return ""
    return f"""
    <details class="trace-card">
      <summary>{escape(title)}</summary>
      <pre>{escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>
    </details>
    """


def render_list_block(title: str, value: object) -> str:
    items = [str(item) for item in value if item] if isinstance(value, list) else []
    if not items:
        return ""
    rendered = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"""
    <details class="trace-card">
      <summary>{escape(title)}</summary>
      <ul>{rendered}</ul>
    </details>
    """


def render_pre_block(title: str, value: str) -> str:
    if not value:
        return ""
    return f"""
    <details class="trace-card">
      <summary>{escape(title)}</summary>
      <pre>{escape(value)}</pre>
    </details>
    """


def render_error_block(error: str) -> str:
    if not error:
        return ""
    return f'<p class="error"><strong>Error:</strong> {escape(error)}</p>'


def render_table(title: str, rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{render_cell(row.get(column))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"""
    <details class="trace-card">
      <summary>{escape(title)} ({len(rows)})</summary>
      <table>
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </details>
    """


def render_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
        return f"<details><summary>View</summary><pre>{escape(rendered)}</pre></details>"
    text = str(value)
    if len(text) > 220 or "\n" in text:
        return f"<details><summary>View</summary><pre>{escape(text)}</pre></details>"
    return escape(text)


def question_has_failure(question: dict[str, object]) -> bool:
    if str(question.get("status", "")) == "failed":
        return True
    evaluation = as_dict(question.get("evaluation"))
    if evaluation:
        answer_accuracy = evaluation.get("answer_accuracy")
        if isinstance(answer_accuracy, (int, float)):
            return float(answer_accuracy) < 1.0
    return str(question.get("pred_answer", "")) != str(question.get("gold_answer", ""))


def format_eval_brief(evaluation: dict[str, object]) -> str:
    if not evaluation:
        return "n/a"
    return (
        f"str_acc={evaluation.get('string_accuracy', 'n/a')} "
        f"ans_acc={evaluation.get('answer_accuracy', 'n/a')}"
    )


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{escape(title)}</title>
    <style>
      body {{ font-family: Georgia, serif; margin: 2rem; background: #f7f5ef; color: #222; }}
      h1, h2 {{ margin-top: 0; }}
      a {{ color: #114b5f; }}
      table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: white; }}
      th, td {{ border: 1px solid #d8d1c0; padding: 0.6rem; vertical-align: top; text-align: left; }}
      th {{ background: #ece6d6; }}
      pre {{ white-space: pre-wrap; word-break: break-word; background: #f3efe4; padding: 0.75rem; border: 1px solid #d8d1c0; }}
      code {{ background: #f3efe4; padding: 0.1rem 0.25rem; }}
      details {{ margin: 0.75rem 0; }}
      .dataset-card, .summary-card, .trace-section {{ margin: 1.5rem 0; }}
      .trace-card, .step-card {{ background: white; padding: 0.75rem; border: 1px solid #d8d1c0; }}
      .status.ok {{ color: #0b6e4f; font-weight: bold; }}
      .status.failure {{ color: #9a031e; font-weight: bold; }}
      .eval-box {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.75rem 0; }}
      .error {{ color: #9a031e; font-weight: bold; }}
      nav {{ margin-bottom: 1rem; }}
    </style>
  </head>
  <body>{body}</body>
</html>"""


def as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def list_of_dicts(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def require_param(params: dict[str, list[str]], name: str) -> str:
    values = params.get(name)
    if not values or not values[0]:
        raise ValueError(f"Missing required parameter: {name}")
    return values[0]


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
