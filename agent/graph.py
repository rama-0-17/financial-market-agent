"""
Financial Market Intelligence Agent — LangGraph implementation.

Optimized architecture:
  recall_memory → planner → executor → synthesizer → store_memory → END

Major optimization:
  - Planner produces DIRECT tool calls
  - Executor NO LONGER uses an LLM
  - Tool execution is deterministic
  - Huge reduction in TPM/rate limits
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, StateGraph

from agent.llm import (
    get_planner_llm,
    get_worker_llm,
    invoke_llm,
)
from agent.state import (
    AgentState,
    Plan,
    PlanStep,
    ToolCall,
)
from memory.store import (
    Scratchpad,
    recall_memories,
    store_memory,
)
from tools.executor import (
    compute_financial_ratios_tool,
    python_executor_tool,
)
from tools.financial import (
    competitor_comparison_tool,
    financial_data_tool,
)
from tools.search import (
    news_search_tool,
    web_search_tool,
)

# ─────────────────────────────────────────────────────────────
# Robust JSON extraction
# ─────────────────────────────────────────────────────────────


def extract_json_from_text(text: str) -> dict | None:
    """
    Safely extract JSON from messy LLM output.
    """

    code_block_match = re.search(
        r"```(?:json)?\s*\n(.*?)\n```",
        text,
        re.DOTALL,
    )

    if code_block_match:
        text_to_parse = code_block_match.group(1).strip()
    else:
        text_to_parse = text

    start_idx = text_to_parse.find("{")

    if start_idx == -1:
        return None

    for attempt_idx in range(
        start_idx,
        len(text_to_parse),
    ):

        if text_to_parse[attempt_idx] != "{":
            continue

        for end_idx in range(
            len(text_to_parse),
            attempt_idx,
            -1,
        ):

            if text_to_parse[end_idx - 1] != "}":
                continue

            candidate = text_to_parse[
                attempt_idx:end_idx
            ]

            try:
                return json.loads(candidate)

            except Exception:
                continue

    return None


# ─────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────

TOOLS = {
    "financial_data": financial_data_tool,
    "competitor_comparison": competitor_comparison_tool,
    "web_search": web_search_tool,
    "news_search": news_search_tool,
    "python_executor": python_executor_tool,
    "compute_ratios": compute_financial_ratios_tool,
}

TOOL_DESCRIPTIONS = """
Available tools:

- financial_data(ticker, data_type)
  Fetch stock/company financial data.
  data_type:
    - overview
    - revenue_history
    - earnings
    - price_history

- competitor_comparison(tickers)
  Compare multiple stocks side-by-side.

- web_search(query)
  Search the web for financial/company information.

- news_search(company)
  Get recent financial news.

- python_executor(code)
  Execute Python for calculations.

- compute_ratios(data_json)
  Compute derived financial ratios.
"""

# ─────────────────────────────────────────────────────────────
# Scratchpad
# ─────────────────────────────────────────────────────────────

_scratchpad = Scratchpad()

# ─────────────────────────────────────────────────────────────
# Node: recall_memory
# ─────────────────────────────────────────────────────────────


def recall_memory_node(state: AgentState) -> dict:
    """
    Retrieve relevant memories before planning.
    """

    memories = recall_memories(
        state.user_query,
        n_results=3,
    )

    mem_ids = [m["id"] for m in memories]

    if memories:

        mem_context = "\n".join(
            f"[Past analysis] {m['content'][:300]}"
            for m in memories
        )

        _scratchpad.add(
            "RECALLED MEMORY",
            mem_context,
        )

    return {
        "memory_ids": mem_ids,
    }


# ─────────────────────────────────────────────────────────────
# Node: planner
# ─────────────────────────────────────────────────────────────


def planner_node(state: AgentState) -> dict:
    """
    Planner creates DIRECT executable tool steps.
    """

    llm = get_planner_llm()

    system = SystemMessage(content=f"""
You are a senior financial research planner.

Your job is to create a DIRECT executable plan.

Available tools:

{TOOL_DESCRIPTIONS}

You MUST return valid JSON.

Response format:

{{
  "goal": "one sentence goal",

  "primary_ticker": "AAPL",

  "competitor_tickers": [
    "MSFT",
    "GOOGL"
  ],

  "steps": [
    {{
      "tool": "financial_data",
      "args": {{
        "ticker": "AAPL",
        "data_type": "overview"
      }},
      "reason": "Fetch company overview"
    }},

    {{
      "tool": "news_search",
      "args": {{
        "company": "Apple"
      }},
      "reason": "Get recent financial news"
    }}
  ],

  "reasoning": "brief reasoning"
}}

Rules:
- ONLY use available tools
- Return JSON ONLY
- No markdown
- No explanations outside JSON
- Max 6 steps
- Prefer direct financial tools before web search
- Avoid redundant tool calls
""")

    context = ""

    if state.memory_ids:
        context = (
            "\nRelevant past research "
            "is available in memory."
        )

    user_msg = HumanMessage(
        content=f"""
Query:
{state.user_query}

{context}
"""
    )

    try:

        response = invoke_llm(
            llm,
            [system, user_msg],
        )

        content = response.content

        plan_data = extract_json_from_text(content)

        if not plan_data:
            raise ValueError(
                f"No valid JSON found:\n{content[:300]}"
            )

        steps = [
            {
                "tool": s.get("tool", ""),
                "args": s.get("args", {}),
                "reason": s.get("reason", ""),
            }
            for s in plan_data.get("steps", [])
        ]

        plan = Plan(
            goal=plan_data.get(
                "goal",
                state.user_query,
            ),
            steps=steps,
            reasoning=plan_data.get(
                "reasoning",
                "",
            ),
        )

        _scratchpad.add_plan(
            f"Goal: {plan.goal}\n"
            + "\n".join(
                f"- {s.tool}: {s.args}"
                for s in plan.steps
            )
        )

        return {
            "plan": plan,
            "primary_ticker": plan_data.get(
                "primary_ticker",
                "",
            ),
            "competitor_tickers": plan_data.get(
                "competitor_tickers",
                [],
            ),
            "messages": [
                AIMessage(
                    content=f"Plan created: {plan.goal}"
                )
            ],
        }

    except Exception as e:

        fallback_plan = Plan(
            goal=f"Analyze {state.user_query}",
            steps=[
                PlanStep(
                    tool="financial_data",
                    args={
                        "ticker": state.user_query.split()[0],
                        "data_type": "overview",
                    },
                    reason="Fallback overview fetch",
                ),
                PlanStep(
                    tool="news_search",
                    args={
                        "company": state.user_query,
                    },
                    reason="Fallback news search",
                ),
            ],
            reasoning=f"Fallback plan due to: {e}",
        )

        _scratchpad.add_error(
            "planner",
            str(e),
        )

        return {
            "plan": fallback_plan,
            "errors": state.errors + [
                f"Planner error: {e}"
            ],
            "fallback_used": True,
        }


# ─────────────────────────────────────────────────────────────
# Node: executor
# ─────────────────────────────────────────────────────────────


def executor_node(state: AgentState) -> dict:
    """
    Direct deterministic tool execution.

    NO executor LLM.
    """

    plan = state.plan

    if not plan or not plan.steps:
        return {
            "errors": state.errors
            + ["No plan to execute"],
            "done": True,
        }

    all_tool_calls = list(state.tool_calls)

    tool_results = dict(state.tool_results)

    errors = list(state.errors)

    retry_count = state.retry_count

    for i, step in enumerate(
        plan.steps[plan.current_step:],
        start=plan.current_step,
    ):

        step_key = f"step_{i}"

        if step_key in tool_results:
            continue

        tool_name = step.tool

        args = step.args

        if tool_name not in TOOLS:

            errors.append(
                f"Unknown tool: {tool_name}"
            )

            continue

        tool_call = ToolCall(
            tool=tool_name,
            input=args,
        )

        attempt = 0

        max_attempts = 3

        while attempt < max_attempts:

            attempt += 1

            try:

                tool_fn = TOOLS[tool_name]

                # result = tool_fn.invoke(args)

                # Special argument handling per tool

                if tool_name == "compute_ratios":
                    if "data_json" in args:
                        payload = {
                            "data_json": (
                                args["data_json"]
                                if isinstance(args["data_json"], str)
                                else json.dumps(args["data_json"])
                            )
                        }
                    else:
                        payload = {
                            "data_json": json.dumps(args)
                        }
                    result = tool_fn.invoke(payload)
                else:
                    result = tool_fn.invoke(args)

                tool_call.output = result

                tool_call.attempt = attempt

                tool_results[step_key] = result

                all_tool_calls.append(tool_call)

                _scratchpad.add_tool_result(
                    tool_name,
                    str(result)[:1500],
                )

                break

            except Exception as tool_err:

                if attempt < max_attempts:

                    time.sleep(2 ** attempt)

                    continue

                tool_call.error = str(tool_err)

                all_tool_calls.append(tool_call)

                error_msg = (
                    f"Tool {tool_name} failed: "
                    f"{tool_err}"
                )

                errors.append(error_msg)

                _scratchpad.add_error(
                    step_key,
                    error_msg,
                )

        time.sleep(0.5)

    return {
        "tool_calls": all_tool_calls,
        "tool_results": tool_results,
        "errors": errors,
        "retry_count": retry_count,
        "plan": Plan(
            goal=plan.goal,
            steps=plan.steps,
            current_step=len(plan.steps),
            reasoning=plan.reasoning,
        ),
    }


# ─────────────────────────────────────────────────────────────
# Node: synthesizer
# ─────────────────────────────────────────────────────────────


def synthesizer_node(state: AgentState) -> dict:
    """
    Generate final analyst report.
    """

    llm = get_worker_llm()

    scratchpad_content = _scratchpad.to_string(
        max_chars=3500
    )

    system = SystemMessage(content="""
You are a senior equity research analyst.

Write a professional financial report.

Use markdown headers.

Required sections:

## Executive Summary
## Company Overview
## Financial Performance
## Competitive Landscape
## Key Risks & Opportunities
## Analyst Verdict

Guidelines:
- Use actual numbers when available
- Be concise
- Be specific
- Mention missing data honestly
- End with Bullish/Neutral/Bearish verdict
""")

    errors_note = ""

    if state.errors:

        errors_note = (
            "\n\nErrors during analysis:\n"
            + "\n".join(
                f"- {e}"
                for e in state.errors[:5]
            )
        )

    user_msg = HumanMessage(content=f"""
Query:
{state.user_query}

Primary ticker:
{state.primary_ticker}

Competitors:
{', '.join(state.competitor_tickers)}

Working memory:
{scratchpad_content}

{errors_note}

Write the report now.
""")

    try:

        response = invoke_llm(
            llm,
            [system, user_msg],
        )

        report = response.content

        sections = {}

        section_pattern = (
            r"##\s+(.+?)\n(.*?)(?=##\s+|\Z)"
        )

        for match in re.finditer(
            section_pattern,
            report,
            re.DOTALL,
        ):

            sections[
                match.group(1).strip()
            ] = match.group(2).strip()

        return {
            "report": report,
            "report_sections": sections,
            "messages": [
                AIMessage(
                    content="Report generated successfully."
                )
            ],
        }

    except Exception as e:

        fallback_report = f"""
# Market Intelligence Report

Query:
{state.user_query}

Report synthesis failed:
{e}

## Raw Working Memory

{scratchpad_content[:2500]}
"""

        return {
            "report": fallback_report,
            "errors": state.errors
            + [f"Synthesis error: {e}"],
        }


# ─────────────────────────────────────────────────────────────
# Node: store_memory
# ─────────────────────────────────────────────────────────────


def store_memory_node(state: AgentState) -> dict:
    """
    Persist useful information into ChromaDB.
    """

    if not state.report:
        return {"done": True}

    summary = state.report_sections.get(
        "Executive Summary",
        state.report[:500],
    )

    mem_id = store_memory(
        content=(
            f"Query: {state.user_query}\n"
            f"Ticker: {state.primary_ticker}\n\n"
            f"{summary}"
        ),
        metadata={
            "ticker": state.primary_ticker,
            "query": state.user_query[:100],
            "type": "analysis_summary",
        },
    )

    if state.competitor_tickers:

        comp_summary = (
            f"Competitors of "
            f"{state.primary_ticker}: "
            f"{', '.join(state.competitor_tickers)}"
        )

        store_memory(
            content=comp_summary,
            metadata={
                "ticker": state.primary_ticker,
                "type": "competitor_map",
            },
        )

    _scratchpad.clear()

    return {
        "done": True,
        "memory_ids": state.memory_ids + [mem_id],
    }


# ─────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────


def should_replan(
    state: AgentState,
) -> Literal["planner", "synthesizer"]:

    if (
        state.replanned
        and state.retry_count >= state.max_retries
    ):
        return "synthesizer"

    if (
        len(state.errors) > 3
        and not state.replanned
    ):
        return "planner"

    return "synthesizer"


# ─────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────


def build_agent() -> StateGraph:

    graph = StateGraph(AgentState)

    graph.add_node(
        "recall_memory",
        recall_memory_node,
    )

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "executor",
        executor_node,
    )

    graph.add_node(
        "synthesizer",
        synthesizer_node,
    )

    graph.add_node(
        "store_memory",
        store_memory_node,
    )

    graph.set_entry_point("recall_memory")

    graph.add_edge(
        "recall_memory",
        "planner",
    )

    graph.add_edge(
        "planner",
        "executor",
    )

    graph.add_conditional_edges(
        "executor",
        should_replan,
    )

    graph.add_edge(
        "synthesizer",
        "store_memory",
    )

    graph.add_edge(
        "store_memory",
        END,
    )

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

_agent = None


def get_agent():

    global _agent

    if _agent is None:
        _agent = build_agent()

    return _agent


def run_agent(query: str) -> AgentState:
    """
    Run full agent pipeline.
    """

    agent = get_agent()

    initial_state = AgentState(
        user_query=query,
        messages=[
            HumanMessage(content=query)
        ],
    )

    final_state = agent.invoke(initial_state)

    return AgentState(**final_state)