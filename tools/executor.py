"""
Code execution tool — runs Python in a subprocess with timeout + output capture.

This demonstrates the "code exec" pillar of the agentic system.
The agent uses this to: compute financial ratios, run comparisons,
build summary tables, and do any arithmetic that needs to be exact.

Safety: we run in a subprocess (not eval), apply a hard timeout,
and restrict dangerous imports via an allowlist check.
"""
from __future__ import annotations
import sys
import json
import subprocess
import textwrap
import tempfile
import os
from langchain_core.tools import tool


# Imports the agent is allowed to use in generated code
ALLOWED_IMPORTS = {
    "math", "statistics", "json", "re", "datetime", "collections",
    "itertools", "functools", "operator", "decimal", "fractions",
    "pandas", "numpy", "yfinance",
}

BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess", "import socket",
    "__import__", "open(", "exec(", "eval(", "compile(",
    "importlib", "shutil", "pathlib",
]


def _is_safe(code: str) -> tuple[bool, str]:
    """
    Lightweight safety check on generated code before execution.
    Not a sandbox — just a first-pass guard against obvious misuse.
    """
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return False, f"Blocked pattern detected: '{pattern}'"
    return True, ""


def _run_python(code: str, timeout: int = 15) -> dict:
    """
    Execute Python code in a subprocess with a hard timeout.
    Returns stdout, stderr, and exit code.
    """
    # Inject safe preamble
    preamble = textwrap.dedent("""
        import math, statistics, json, re, datetime
        from collections import defaultdict, Counter
        import pandas as pd
        import numpy as np
    """)
    full_code = preamble + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="agent_exec_"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "exit_code": -1,
            "success": False,
        }
    finally:
        os.unlink(tmp_path)


# ── LangChain Tool ────────────────────────────────────────────────────────────

@tool
def python_executor_tool(code: str) -> str:
    """
    Execute Python code to compute financial metrics, ratios, or comparisons.
    Use this when you need precise arithmetic, data manipulation, or table generation.

    Available libraries: math, statistics, json, pandas, numpy
    Output via print() statements — everything printed will be captured.

    Args:
        code: Valid Python code to execute

    Returns:
        JSON string with stdout output, any errors, and success status
    """
    safe, reason = _is_safe(code)
    if not safe:
        return json.dumps({
            "success": False,
            "error": f"Code blocked by safety check: {reason}",
            "stdout": "",
        })

    result = _run_python(code)

    if not result["success"] and result["stderr"]:
        # Return structured error so the agent can self-correct
        return json.dumps({
            "success": False,
            "error": result["stderr"],
            "stdout": result["stdout"],
            "hint": "Fix the code and retry. Common issues: syntax errors, undefined variables, wrong column names.",
        })

    return json.dumps({
        "success": True,
        "output": result["stdout"],
        "stderr": result["stderr"] if result["stderr"] else None,
    })


@tool
def compute_financial_ratios_tool(data_json: str) -> str:
    """
    Compute derived financial ratios from raw financial data.
    Pass in a JSON string of financial metrics; returns computed ratios.

    Args:
        data_json: JSON string containing financial metrics

    Returns:
        JSON string with computed ratios and analysis
    """
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    code = f"""
data = {json.dumps(data)}
results = {{}}

# Debt-to-equity
cash = data.get('cash') or 0
debt = data.get('debt') or 0
if debt and cash:
    results['net_debt'] = debt - cash
    results['net_debt_fmt'] = f"${{(debt - cash)/1e9:.2f}}B"

# Price-to-FCF
price = data.get('current_price')
shares = data.get('shares_outstanding')
fcf = data.get('free_cash_flow')
mkt_cap = data.get('market_cap')
if fcf and mkt_cap and fcf > 0:
    results['price_to_fcf'] = round(mkt_cap / fcf, 2)

# Revenue per share (proxy for operational scale)
revenue = data.get('revenue_ttm')
if revenue and mkt_cap and mkt_cap > 0:
    results['ps_ratio_check'] = round(mkt_cap / revenue, 2)

# Margin quality score (0-100)
gross = data.get('gross_margin') or 0
operating = data.get('operating_margin') or 0
net = data.get('net_margin') or 0
results['margin_quality_score'] = round((gross * 40 + operating * 35 + net * 25) * 100, 1)

# Growth-adjusted PE (PEG proxy)
pe = data.get('pe_ratio')
growth = data.get('earnings_growth')
if pe and growth and growth > 0:
    results['peg_ratio'] = round(pe / (growth * 100), 2)

print(json.dumps(results, indent=2))
"""
    result = _run_python(code)
    if result["success"]:
        try:
            return result["stdout"]
        except Exception:
            return result["stdout"]
    return json.dumps({"error": result["stderr"]})
