"""Entry point: `disempower-eval run` and `disempower-eval report`."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import anthropic
from rich.console import Console

from . import evaluator, report, runner
from .models import Dataset, GradeRecord

console = Console()


def _write_jsonl(path: Path, items) -> None:
    path.write_text("".join(item.model_dump_json() + "\n" for item in items))


def cmd_run(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.scenarios)
    scenarios = dataset.scenarios
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s.id in wanted or s.primitive in wanted]

    out = Path(args.out) / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(max_retries=4)

    total = len(scenarios) * args.samples
    console.print(f"Running {total} conversation(s) against [cyan]{args.target_model}[/]...")
    transcripts = runner.run_all(
        client,
        scenarios,
        model=args.target_model,
        samples=args.samples,
        effort=args.target_effort,
        system=args.target_system,
        concurrency=args.concurrency,
        on_done=lambda t: console.print(f"  [dim]target done[/] {t.scenario_id}#{t.sample}" + (f" [red]{t.error}[/]" if t.error else "")),
    )
    _write_jsonl(out / "transcripts.jsonl", transcripts)

    console.print(f"Grading with judge [cyan]{args.judge_model}[/]...")
    grades = evaluator.grade_all(
        client,
        dataset,
        transcripts,
        judge_model=args.judge_model,
        effort=args.judge_effort,
        concurrency=args.concurrency,
        on_done=lambda r: console.print(
            f"  [dim]graded[/] {r.scenario_id}#{r.sample} -> {r.grade.verdict if r.grade else '[red]' + str(r.error) + '[/]'}"
        ),
    )
    _write_jsonl(out / "grades.jsonl", grades)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    console.print(f"\nSaved results to [bold]{out}[/]\n")
    report.render(grades, console)


def cmd_report(args: argparse.Namespace) -> None:
    path = Path(args.run_dir) / "grades.jsonl"
    records = [GradeRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    report.render(records, console)


def main() -> None:
    parser = argparse.ArgumentParser(prog="disempower-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run scenarios against a target model and grade them")
    run.add_argument("--scenarios", default="scenarios.json")
    run.add_argument("--target-model", default=runner.DEFAULT_TARGET_MODEL)
    run.add_argument("--target-effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                     help="Effort for the target model (omit to use the model's default)")
    run.add_argument("--target-system", default=None, help="Optional system prompt for the target model")
    run.add_argument("--judge-model", default=evaluator.DEFAULT_JUDGE_MODEL)
    run.add_argument("--judge-effort", choices=["low", "medium", "high", "xhigh", "max"], default="high")
    run.add_argument("--samples", type=int, default=1, help="Independent samples per scenario")
    run.add_argument("--only", default=None, help="Comma-separated scenario ids or primitive names")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--out", default="results")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="Re-render the report for a saved run")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
