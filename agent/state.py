"""
Agent state — the single source of truth flowing through every LangGraph node.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """
    Structured execution step produced directly by planner.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class Plan(BaseModel):
    """
    Structured plan produced by planner node.
    """

    goal: str
    steps: list[PlanStep]
    current_step: int = 0
    reasoning: str = ""


class ToolCall(BaseModel):
    """
    Record of a single tool invocation.
    """

    tool: str
    input: dict[str, Any]

    output: Any = None
    error: Optional[str] = None

    attempt: int = 1


class AgentState(BaseModel):
    """
    Full agent state flowing through LangGraph.
    """

    # ─────────────────────────────────────────────
    # Conversation
    # ─────────────────────────────────────────────

    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list
    )

    # ─────────────────────────────────────────────
    # Task
    # ─────────────────────────────────────────────

    user_query: str = ""

    primary_ticker: str = ""
    competitor_tickers: list[str] = Field(default_factory=list)

    # ─────────────────────────────────────────────
    # Planning
    # ─────────────────────────────────────────────

    plan: Optional[Plan] = None

    replanned: bool = False

    # ─────────────────────────────────────────────
    # Tool execution
    # ─────────────────────────────────────────────

    tool_calls: list[ToolCall] = Field(default_factory=list)

    tool_results: dict[str, Any] = Field(default_factory=dict)

    # ─────────────────────────────────────────────
    # Error recovery
    # ─────────────────────────────────────────────

    errors: list[str] = Field(default_factory=list)

    retry_count: int = 0
    max_retries: int = 3

    fallback_used: bool = False

    # ─────────────────────────────────────────────
    # Memory
    # ─────────────────────────────────────────────

    scratchpad: str = ""

    memory_ids: list[str] = Field(default_factory=list)

    # ─────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────

    report: str = ""

    report_sections: dict[str, str] = Field(default_factory=dict)

    financial_data: dict[str, Any] = Field(default_factory=dict)

    done: bool = False

    class Config:
        arbitrary_types_allowed = True