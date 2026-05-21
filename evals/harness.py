"""
Evaluation harness — the key L3 differentiator.

Runs a suite of test cases and measures:
  - Task completion rate (did the agent produce a report?)
  - Tool call efficiency (avg calls per task)
  - Error rate (how often did tools fail?)
  - Fallback rate (how often did error recovery trigger?)
  - Report quality (section coverage, data density)
  - Latency per run

Usage:
    python evals/harness.py                    # run full suite
    python evals/harness.py --quick            # 3 test cases only
    python evals/harness.py --ticker NVDA      # single ticker
"""
from __future__ import annotations
import sys
import os
import json
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import run_agent
from agent.state import AgentState


# ── Test suite ────────────────────────────────────────────────────────────────

FULL_TEST_CASES = [
    {
        "id": "tc_001",
        "query": "Analyse Apple (AAPL) and compare it against Microsoft and Google",
        "expected_ticker": "AAPL",
        "expected_sections": ["Executive Summary", "Financial Performance", "Competitive Landscape"],
        "tags": ["mega_cap", "tech"],
    },
    {
        "id": "tc_002",
        "query": "Is NVIDIA overvalued compared to AMD and Intel?",
        "expected_ticker": "NVDA",
        "expected_sections": ["Executive Summary", "Analyst Verdict"],
        "tags": ["semiconductors", "valuation"],
    },
    {
        "id": "tc_003",
        "query": "Give me a market intelligence report on Tesla vs Ford and GM",
        "expected_ticker": "TSLA",
        "expected_sections": ["Competitive Landscape", "Key Risks & Opportunities"],
        "tags": ["automotive", "ev"],
    },
    {
        "id": "tc_004",
        "query": "How is Amazon performing and who are its main competitors?",
        "expected_ticker": "AMZN",
        "expected_sections": ["Company Overview", "Financial Performance"],
        "tags": ["ecommerce", "cloud"],
    },
    {
        "id": "tc_005",
        "query": "Analyse Meta Platforms vs Snap and Pinterest",
        "expected_ticker": "META",
        "expected_sections": ["Executive Summary", "Competitive Landscape"],
        "tags": ["social_media", "advertising"],
    },
]

QUICK_TEST_CASES = FULL_TEST_CASES[:2]


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    test_id: str
    query: str
    success: bool
    latency_s: float
    tool_call_count: int
    error_count: int
    fallback_used: bool
    report_length: int
    sections_found: list[str]
    sections_missing: list[str]
    data_points: int           # count of $ amounts / % in report
    notes: str = ""

    @property
    def section_coverage(self) -> float:
        total = len(self.sections_found) + len(self.sections_missing)
        return len(self.sections_found) / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "query": self.query,
            "success": self.success,
            "latency_s": round(self.latency_s, 2),
            "tool_calls": self.tool_call_count,
            "errors": self.error_count,
            "fallback_used": self.fallback_used,
            "report_length": self.report_length,
            "section_coverage": round(self.section_coverage, 2),
            "data_points": self.data_points,
            "sections_found": self.sections_found,
            "sections_missing": self.sections_missing,
            "notes": self.notes,
        }


def _count_data_points(report: str) -> int:
    """Count $ amounts and % figures as proxy for data density."""
    import re
    dollars = len(re.findall(r'\$[\d,.]+[BMK]?', report))
    percents = len(re.findall(r'[\d.]+%', report))
    return dollars + percents


def _check_sections(report: str, expected_sections: list[str]) -> tuple[list[str], list[str]]:
    found, missing = [], []
    for section in expected_sections:
        if section.lower() in report.lower():
            found.append(section)
        else:
            missing.append(section)
    return found, missing


def evaluate_run(tc: dict, state: AgentState, latency: float) -> RunResult:
    """Score a single agent run against a test case."""
    expected_sections = tc.get("expected_sections", [])
    found, missing = _check_sections(state.report, expected_sections)

    return RunResult(
        test_id=tc["id"],
        query=tc["query"],
        success=bool(state.report and len(state.report) > 200),
        latency_s=latency,
        tool_call_count=len(state.tool_calls),
        error_count=len(state.errors),
        fallback_used=state.fallback_used,
        report_length=len(state.report),
        sections_found=found,
        sections_missing=missing,
        data_points=_count_data_points(state.report),
        notes="; ".join(state.errors[:3]) if state.errors else "",
    )


# ── Summary stats ─────────────────────────────────────────────────────────────

@dataclass
class EvalSummary:
    results: list[RunResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def completion_rate(self) -> float:
        return sum(r.success for r in self.results) / self.n if self.n else 0

    @property
    def avg_tool_calls(self) -> float:
        return sum(r.tool_call_count for r in self.results) / self.n if self.n else 0

    @property
    def avg_latency(self) -> float:
        return sum(r.latency_s for r in self.results) / self.n if self.n else 0

    @property
    def avg_section_coverage(self) -> float:
        return sum(r.section_coverage for r in self.results) / self.n if self.n else 0

    @property
    def error_rate(self) -> float:
        runs_with_errors = sum(1 for r in self.results if r.error_count > 0)
        return runs_with_errors / self.n if self.n else 0

    @property
    def fallback_rate(self) -> float:
        return sum(r.fallback_used for r in self.results) / self.n if self.n else 0

    @property
    def avg_data_points(self) -> float:
        return sum(r.data_points for r in self.results) / self.n if self.n else 0

    def print_report(self):
        sep = "─" * 60
        print(f"\n{'═' * 60}")
        print(f"  EVAL RESULTS — {self.timestamp}")
        print(f"{'═' * 60}")
        print(f"  Test cases run:     {self.n}")
        print(f"  Task completion:    {self.completion_rate:.0%}  ({sum(r.success for r in self.results)}/{self.n})")
        print(f"  Section coverage:   {self.avg_section_coverage:.0%}")
        print(f"  Avg tool calls:     {self.avg_tool_calls:.1f} per run")
        print(f"  Avg latency:        {self.avg_latency:.1f}s")
        print(f"  Error rate:         {self.error_rate:.0%}")
        print(f"  Fallback rate:      {self.fallback_rate:.0%}")
        print(f"  Avg data points:    {self.avg_data_points:.1f} ($/% figures)")
        print(f"\n{sep}")
        print("  Per-test breakdown:")
        print(sep)
        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"  {status} [{r.test_id}] {r.query[:50]}")
            print(f"      tools={r.tool_call_count} | errors={r.error_count} | "
                  f"latency={r.latency_s:.1f}s | coverage={r.section_coverage:.0%} | "
                  f"data_pts={r.data_points}")
            if r.sections_missing:
                print(f"      missing sections: {', '.join(r.sections_missing)}")
            if r.notes:
                print(f"      notes: {r.notes[:100]}")
        print(f"{'═' * 60}\n")

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "n": self.n,
                "completion_rate": round(self.completion_rate, 3),
                "avg_tool_calls": round(self.avg_tool_calls, 2),
                "avg_latency_s": round(self.avg_latency, 2),
                "avg_section_coverage": round(self.avg_section_coverage, 3),
                "error_rate": round(self.error_rate, 3),
                "fallback_rate": round(self.fallback_rate, 3),
                "avg_data_points": round(self.avg_data_points, 1),
            },
            "results": [r.to_dict() for r in self.results],
        }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_eval(test_cases: list[dict], output_path: str | None = None) -> EvalSummary:
    summary = EvalSummary()

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Running: {tc['query'][:60]}...")
        start = time.time()
        try:
            state = run_agent(tc["query"])
            latency = time.time() - start
            result = evaluate_run(tc, state, latency)
            print(f"  → {'✓ success' if result.success else '✗ failed'} | "
                  f"{result.tool_call_count} tools | {latency:.1f}s | "
                  f"{result.data_points} data points")
        except Exception as e:
            latency = time.time() - start
            result = RunResult(
                test_id=tc["id"],
                query=tc["query"],
                success=False,
                latency_s=latency,
                tool_call_count=0,
                error_count=1,
                fallback_used=False,
                report_length=0,
                sections_found=[],
                sections_missing=tc.get("expected_sections", []),
                data_points=0,
                notes=f"Agent crashed: {e}",
            )
            print(f"  → ✗ CRASHED: {e}")

        summary.results.append(result)

    summary.print_report()

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Results saved to {output_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent evaluation suite")
    parser.add_argument("--quick", action="store_true", help="Run only 2 test cases")
    parser.add_argument("--ticker", type=str, help="Run single ticker test")
    parser.add_argument("--output", type=str, default="evals/results.json")
    args = parser.parse_args()

    if args.ticker:
        cases = [{
            "id": "custom",
            "query": f"Analyse {args.ticker} and compare with its main competitors",
            "expected_ticker": args.ticker.upper(),
            "expected_sections": ["Executive Summary", "Financial Performance", "Competitive Landscape", "Analyst Verdict"],
            "tags": ["custom"],
        }]
    elif args.quick:
        cases = QUICK_TEST_CASES
    else:
        cases = FULL_TEST_CASES

    run_eval(cases, output_path=args.output)
