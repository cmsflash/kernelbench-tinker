#!/usr/bin/env python3
"""Serve a live KernelBench RL training report from run artifacts."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse


SKIPPED_RE = re.compile(r"All groups filtered out, skipping")
ERROR_RE = re.compile(r"\b(ERROR|Traceback|Exception|Failed|failed)\b")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    bad_lines = 0
    if not path.exists():
        return records, bad_lines
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records, 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            bad_lines += 1
    return records, bad_lines


def artifact_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime if path.exists() else None
    except OSError:
        return None


def read_log_sources(run_dir: Path) -> dict[str, str]:
    candidates = [
        run_dir / "logs.log",
        run_dir / "nohup.log",
        run_dir.parent / f"{run_dir.name}_nohup.log",
    ]
    candidates.extend(run_dir.parent.glob(f"{run_dir.name}*nohup*.log"))

    sources: dict[str, str] = {}
    for path in sorted(set(candidates)):
        if not path.exists() or not path.is_file():
            continue
        try:
            sources[str(path)] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            sources[str(path)] = ""
    return sources


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def latest(records: list[dict[str, Any]]) -> dict[str, Any]:
    return records[-1] if records else {}


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def infer_expected_batches(config: dict[str, Any], metrics: list[dict[str, Any]]) -> int | None:
    dataset = config.get("dataset_builder") or {}
    num_epochs = finite_number(dataset.get("num_epochs"))
    batch_size = finite_number(dataset.get("batch_size"))
    start_problem = dataset.get("start_problem")
    end_problem = dataset.get("end_problem")

    if num_epochs is not None and batch_size:
        if start_problem is not None and end_problem is not None:
            problems = max(0, int(end_problem) - int(start_problem) + 1)
            return math.ceil(problems / int(batch_size)) * int(num_epochs)
        if int(batch_size) == 1:
            return int(num_epochs)

    last = latest(metrics)
    done_frac = finite_number(last.get("progress/done_frac"))
    step = finite_number(last.get("step"))
    if done_frac and step is not None and done_frac > 0:
        return round((step + 1) / done_frac)
    return None


def bool_metric(*values: Any) -> bool:
    for value in values:
        if value is not None:
            return bool(value)
    return False


def summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    rewards: list[float] = []
    speedups: list[float] = []
    format_ok = compiled = correct = 0
    rows: list[dict[str, Any]] = []

    for idx, trace in enumerate(traces):
        metrics = trace.get("metrics") or {}
        eval_result = trace.get("eval_result") or {}
        response = trace.get("response") or {}

        reward = finite_number(trace.get("reward"))
        speedup = finite_number(first_present(eval_result.get("speedup"), metrics.get("speedup")))
        is_format_ok = bool_metric(metrics.get("format_ok"), response.get("format_ok"), eval_result.get("format_ok"))
        is_compiled = bool_metric(metrics.get("compiled"), eval_result.get("compiled"))
        is_correct = bool_metric(metrics.get("correctness"), eval_result.get("correctness"))

        if reward is not None:
            rewards.append(reward)
        if speedup is not None:
            speedups.append(speedup)
        format_ok += int(is_format_ok)
        compiled += int(is_compiled)
        correct += int(is_correct)

        raw = str(response.get("raw") or "")
        kernel = str(response.get("kernel") or "")
        rows.append(
            {
                "idx": idx,
                "timestamp": trace.get("timestamp"),
                "level": trace.get("level"),
                "problem_id": trace.get("problem_id"),
                "reward": reward,
                "format_ok": is_format_ok,
                "compiled": is_compiled,
                "correctness": is_correct,
                "speedup": speedup,
                "error": first_present(eval_result.get("error_message"), trace.get("error")),
                "output_chars": len(raw),
                "kernel_chars": len(kernel),
            }
        )

    total = len(traces)
    return {
        "total": total,
        "reward_mean": mean(rewards),
        "reward_latest": rewards[-1] if rewards else None,
        "format_rate": format_ok / total if total else None,
        "compile_rate": compiled / total if total else None,
        "correct_rate": correct / total if total else None,
        "mean_speedup": mean(speedups),
        "latest_rows": rows[-200:],
    }


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"latest": {}, "completed_updates": 0, "last_step": None}
    last = metrics[-1]
    numeric_keys = [
        "reward/mean",
        "reward/std",
        "reward/min",
        "reward/max",
        "kernel/format_rate",
        "kernel/compile_rate",
        "kernel/correct_rate",
        "kernel/mean_speedup",
        "time/rollout",
        "time/eval_mean",
        "time/eval_max",
        "time/modal_eval_mean",
        "time/modal_eval_max",
        "time/step_mean",
        "time/train",
        "time/save_checkpoint",
        "time/total",
    ]
    return {
        "latest": {key: last.get(key) for key in numeric_keys if key in last},
        "completed_updates": len(metrics),
        "last_step": last.get("step"),
    }


def collect_status(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    config = read_json(run_dir / "config.json")
    metrics, bad_metric_lines = read_jsonl(run_dir / "metrics.jsonl")
    traces, bad_trace_lines = read_jsonl(run_dir / "traces.jsonl")
    checkpoints, bad_checkpoint_lines = read_jsonl(run_dir / "checkpoints.jsonl")
    log_sources = read_log_sources(run_dir)
    log_text = "\n".join(log_sources.values())

    skipped_batches = len(SKIPPED_RE.findall(log_text))
    error_lines = [
        line[-600:]
        for line in log_text.splitlines()
        if ERROR_RE.search(line)
    ][-20:]

    final_checkpoint = next((ckpt for ckpt in reversed(checkpoints) if ckpt.get("name") == "final"), None)
    latest_checkpoint = latest(checkpoints)

    artifact_paths = [
        run_dir / "config.json",
        run_dir / "metrics.jsonl",
        run_dir / "traces.jsonl",
        run_dir / "checkpoints.jsonl",
        run_dir / "logs.log",
        run_dir.parent / f"{run_dir.name}_nohup.log",
    ]
    mtimes = [mtime for mtime in (artifact_mtime(path) for path in artifact_paths) if mtime is not None]
    now = time.time()
    last_update = max(mtimes) if mtimes else None
    first_update = min(mtimes) if mtimes else None
    if final_checkpoint:
        state = "completed"
    elif last_update is None:
        state = "waiting"
    elif now - last_update < 180:
        state = "active"
    else:
        state = "idle"

    return {
        "run_dir": str(run_dir),
        "state": state,
        "generated_at": now,
        "last_update": last_update,
        "elapsed_s": (last_update - first_update) if last_update and first_update else None,
        "config": config,
        "expected_batches": infer_expected_batches(config, metrics),
        "metrics": summarize_metrics(metrics),
        "traces": summarize_traces(traces),
        "checkpoints": {
            "count": len(checkpoints),
            "latest": latest_checkpoint,
            "final": final_checkpoint,
        },
        "logs": {
            "sources": list(log_sources.keys()),
            "skipped_batches": skipped_batches,
            "error_lines": error_lines,
        },
        "parse_errors": {
            "metrics_jsonl": bad_metric_lines,
            "traces_jsonl": bad_trace_lines,
            "checkpoints_jsonl": bad_checkpoint_lines,
        },
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KernelBench RL Live Report</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #17202a; }
    main { max-width: 1320px; margin: 0 auto; padding: 24px; }
    header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    h1 { margin: 0 0 6px; font-size: 24px; line-height: 1.2; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    .status { display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border-radius: 999px; background: #e7eef8; font-size: 13px; font-weight: 650; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #8b98a7; }
    .active .dot { background: #1f9d55; }
    .completed .dot { background: #2364aa; }
    .idle .dot { background: #d9822b; }
    .waiting .dot { background: #8b98a7; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .panel { background: white; border: 1px solid #dfe4ea; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgb(20 30 40 / 5%); }
    .metric { font-size: 24px; font-weight: 750; line-height: 1.1; }
    .label { color: #5e6b78; font-size: 12px; margin-top: 5px; }
    .wide { grid-column: 1 / -1; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #edf0f3; padding: 8px 7px; text-align: left; vertical-align: top; }
    th { color: #536171; font-size: 12px; background: #fafbfc; position: sticky; top: 0; }
    tbody tr:hover { background: #fbfcfe; }
    .scroll { max-height: 520px; overflow: auto; border: 1px solid #edf0f3; border-radius: 6px; }
    .muted { color: #697887; }
    .ok { color: #137333; font-weight: 650; }
    .bad { color: #b3261e; font-weight: 650; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    pre { white-space: pre-wrap; margin: 0; max-height: 260px; overflow: auto; }
    @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } header { display: block; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>KernelBench RL Live Report</h1>
      <div id="run-dir" class="muted mono"></div>
    </div>
    <div id="state" class="status waiting"><span class="dot"></span><span>waiting</span></div>
  </header>

  <section class="grid">
    <div class="panel"><div id="progress" class="metric">-</div><div class="label">updates / expected batches</div></div>
    <div class="panel"><div id="skipped" class="metric">-</div><div class="label">skipped tied-reward batches</div></div>
    <div class="panel"><div id="reward" class="metric">-</div><div class="label">latest trace reward</div></div>
    <div class="panel"><div id="correct" class="metric">-</div><div class="label">trace correct rate</div></div>
    <div class="panel"><div id="format" class="metric">-</div><div class="label">trace format rate</div></div>
    <div class="panel"><div id="compile" class="metric">-</div><div class="label">trace compile rate</div></div>
    <div class="panel"><div id="speedup" class="metric">-</div><div class="label">mean speedup</div></div>
    <div class="panel"><div id="checkpoints" class="metric">-</div><div class="label">checkpoints</div></div>
  </section>

  <section class="grid">
    <div class="panel wide">
      <h2>Run Setup</h2>
      <div id="setup" class="mono"></div>
    </div>
    <div class="panel wide">
      <h2>Latest Metrics</h2>
      <div id="latest-metrics" class="mono"></div>
    </div>
    <div class="panel wide">
      <h2>Per-Rollout Attempts</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>#</th><th>time</th><th>problem</th><th>reward</th><th>format</th><th>compile</th><th>correct</th><th>speedup</th><th>output chars</th><th>error</th></tr></thead>
          <tbody id="trace-body"></tbody>
        </table>
      </div>
    </div>
    <div class="panel wide">
      <h2>Warnings And Errors</h2>
      <pre id="errors" class="mono muted"></pre>
    </div>
  </section>
</main>
<script>
const fmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
function text(id, value) { document.getElementById(id).textContent = value; }
function pct(value) { return value == null ? "-" : `${fmt.format(value * 100)}%`; }
function num(value) { return value == null ? "-" : fmt.format(value); }
function seconds(value) { return value == null ? "-" : `${fmt.format(value)}s`; }
function yes(value) { return value ? "yes" : "no"; }
function cls(value) { return value ? "ok" : "bad"; }
function when(ts) { return ts ? new Date(ts * 1000).toLocaleTimeString() : "-"; }
function setupText(data) {
  const cfg = data.config || {};
  const ds = cfg.dataset_builder || {};
  return [
    `model: ${cfg.model_name || "-"}`,
    `problem: L${ds.level || "-"}.${ds.start_problem || ds.end_problem || "-"}`,
    `batch_size: ${ds.batch_size ?? "-"}`,
    `group_size: ${ds.group_size ?? "-"}`,
    `num_epochs: ${ds.num_epochs ?? "-"}`,
    `max_tokens: ${cfg.max_tokens ?? "-"}`,
    `remove_constant_reward_groups: ${cfg.remove_constant_reward_groups ?? "-"}`,
    `latest checkpoint: ${(data.checkpoints.latest || {}).sampler_path || "-"}`,
    `final checkpoint: ${(data.checkpoints.final || {}).sampler_path || "-"}`
  ].join("\\n");
}
function metricsText(data) {
  const latest = (data.metrics || {}).latest || {};
  const lines = Object.entries(latest).map(([k, v]) => `${k}: ${num(v)}`);
  lines.push(`elapsed: ${seconds(data.elapsed_s)}`);
  lines.push(`last update: ${data.last_update ? new Date(data.last_update * 1000).toLocaleString() : "-"}`);
  const parse = data.parse_errors || {};
  lines.push(`parse errors: metrics=${parse.metrics_jsonl || 0}, traces=${parse.traces_jsonl || 0}, checkpoints=${parse.checkpoints_jsonl || 0}`);
  return lines.join("\\n");
}
async function refresh() {
  const res = await fetch("/api/status", { cache: "no-store" });
  const data = await res.json();
  text("run-dir", data.run_dir);
  const state = document.getElementById("state");
  state.className = `status ${data.state}`;
  state.lastElementChild.textContent = data.state;
  text("progress", `${(data.metrics || {}).completed_updates || 0} / ${data.expected_batches ?? "-"}`);
  text("skipped", (data.logs || {}).skipped_batches ?? 0);
  text("reward", num((data.traces || {}).reward_latest));
  text("correct", pct((data.traces || {}).correct_rate));
  text("format", pct((data.traces || {}).format_rate));
  text("compile", pct((data.traces || {}).compile_rate));
  text("speedup", num((data.traces || {}).mean_speedup));
  text("checkpoints", (data.checkpoints || {}).count || 0);
  text("setup", setupText(data));
  text("latest-metrics", metricsText(data));
  text("errors", ((data.logs || {}).error_lines || []).join("\\n") || "No error lines found.");

  const body = document.getElementById("trace-body");
  body.textContent = "";
  for (const row of ((data.traces || {}).latest_rows || []).slice().reverse()) {
    const tr = document.createElement("tr");
    const cells = [
      row.idx,
      when(row.timestamp),
      `L${row.level}.${row.problem_id}`,
      num(row.reward),
      yes(row.format_ok),
      yes(row.compiled),
      yes(row.correctness),
      num(row.speedup),
      `${row.output_chars || 0} / ${row.kernel_chars || 0}`,
      row.error || ""
    ];
    cells.forEach((value, i) => {
      const td = document.createElement("td");
      td.textContent = value;
      if ([4, 5, 6].includes(i)) td.className = cls(value === "yes");
      if (i === 9) td.className = "mono";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
}
refresh().catch(err => text("errors", String(err)));
setInterval(() => refresh().catch(err => text("errors", String(err))), 2000);
</script>
</body>
</html>
"""


class ReportHandler(BaseHTTPRequestHandler):
    run_dir: Path

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_bytes(self, content_type: str, data: bytes, include_body: bool = True) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], include_body: bool = True) -> None:
        data = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_bytes("application/json; charset=utf-8", data, include_body)

    def send_html(self, include_body: bool = True) -> None:
        self.send_bytes("text/html; charset=utf-8", HTML.encode("utf-8"), include_body)

    def dispatch(self, include_body: bool) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(collect_status(self.run_dir), include_body)
        elif path in {"/", "/index.html"}:
            self.send_html(include_body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        self.dispatch(include_body=True)

    def do_HEAD(self) -> None:
        self.dispatch(include_body=False)


def make_handler(run_dir: Path) -> type[ReportHandler]:
    class BoundReportHandler(ReportHandler):
        pass

    BoundReportHandler.run_dir = run_dir
    return BoundReportHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Training run artifact directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", default=8765, type=int, help="Bind port")
    args = parser.parse_args()

    args.run_dir.expanduser().mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.run_dir))
    url = f"http://{args.host}:{args.port}"
    print(f"Serving KernelBench live report for {args.run_dir} at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
