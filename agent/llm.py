"""
LLM factory — Groq free tier optimized setup.

Architecture:
- Planner → llama-3.3-70b-versatile
- Worker/Executor → llama-3.1-8b-instant
- Formatter/Synthesizer → llama-3.1-8b-instant

Why:
The planner benefits from stronger reasoning.
Executor/tool-selection/synthesis do not need 70B quality
and are much cheaper + faster on 8B.

Groq Free Tier Limits (approx):
- 30 RPM
- 12,000 TPM

Main constraint is TPM, not RPM.
"""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# ============================================================
# MODELS
# ============================================================

PLANNER_MODEL = "llama-3.3-70b-versatile"
WORKER_MODEL = "llama-3.1-8b-instant"

# ============================================================
# RATE LIMITING
# ============================================================

# Planner is expensive → slower pacing
PLANNER_DELAY = 15.0

# 8B worker is cheap/faster
WORKER_DELAY = 3.0

_last_call_times: dict[str, float] = {}


def _get_delay_for_model(model_name: str) -> float:
    """
    Different pacing for different model sizes.
    """
    if model_name == PLANNER_MODEL:
        return PLANNER_DELAY

    return WORKER_DELAY


def _rate_limit_sleep(model_name: str):
    """
    Pace requests to stay under Groq TPM/RPM limits.
    """
    now = time.time()

    last_call = _last_call_times.get(model_name, 0.0)

    delay = _get_delay_for_model(model_name)

    elapsed = now - last_call

    if elapsed < delay:
        wait = delay - elapsed

        logger.debug(
            f"Pacing {model_name}: sleeping {wait:.1f}s"
        )

        time.sleep(wait)

    _last_call_times[model_name] = time.time()


# ============================================================
# RETRY HELPERS
# ============================================================

def _extract_retry_delay(error_str: str) -> float:
    """
    Parse retry-after seconds from Groq/OpenAI style 429 messages.
    """

    match = re.search(
        r"(?:retry after|try again in|wait)\s*(\d+(?:\.\d+)?)\s*s",
        str(error_str),
        re.IGNORECASE,
    )

    if match:
        return float(match.group(1)) + 2.0

    return 30.0


# ============================================================
# LLM FACTORY
# ============================================================

@lru_cache(maxsize=4)
def _make_llm(model_name: str) -> ChatGroq:
    """
    Cached LLM factory.
    """

    logger.info(f"Initializing LLM: {model_name}")

    return ChatGroq(
        model=model_name,
        temperature=0.1,
        max_retries=0,   # handled manually
        timeout=60,
    )


# ============================================================
# INVOCATION WRAPPER
# ============================================================

def invoke_llm(
    llm: BaseChatModel,
    messages: list,
):
    """
    Centralized invoke wrapper.

    Handles:
    - pacing
    - retries
    - 429 handling
    - temporary outages
    """

    model_name = getattr(llm, "model_name", "unknown")

    for attempt in range(1, 5):

        _rate_limit_sleep(model_name)

        try:
            logger.info(f"Calling model: {model_name}")

            response = llm.invoke(messages)

            return response

        except Exception as e:
            err = str(e)

            is_last_attempt = attempt == 4

            # ------------------------------------------------
            # RATE LIMITS
            # ------------------------------------------------
            if (
                "429" in err
                or "rate_limit" in err.lower()
                or "rate limit" in err.lower()
            ):

                if is_last_attempt:
                    logger.error(
                        f"Rate limit failed after retries: {err}"
                    )
                    raise

                delay = _extract_retry_delay(err)

                logger.warning(
                    f"Rate limited ({model_name}) — "
                    f"waiting {delay:.0f}s "
                    f"(attempt {attempt}/4)"
                )

                print(
                    f"  ⏳ Rate limited ({model_name}), "
                    f"waiting {delay:.0f}s..."
                )

                time.sleep(delay)

            # ------------------------------------------------
            # TEMPORARY OUTAGES
            # ------------------------------------------------
            elif (
                "503" in err
                or "unavailable" in err.lower()
            ):

                if is_last_attempt:
                    logger.error(
                        f"Service unavailable after retries: {err}"
                    )
                    raise

                logger.warning(
                    f"Service unavailable ({model_name}) — "
                    f"waiting 20s "
                    f"(attempt {attempt}/4)"
                )

                time.sleep(20)

            # ------------------------------------------------
            # UNKNOWN ERRORS
            # ------------------------------------------------
            else:
                logger.exception(
                    f"Unhandled LLM error ({model_name})"
                )
                raise


# ============================================================
# PUBLIC HELPERS
# ============================================================

def get_planner_llm() -> ChatGroq:
    """
    Strong reasoning model for planning.
    """
    return _make_llm(PLANNER_MODEL)


def get_worker_llm() -> ChatGroq:
    """
    Fast/cheap worker model for execution.
    """
    return _make_llm(WORKER_MODEL)


def get_formatter_llm() -> ChatGroq:
    """
    Formatter/synthesizer model.
    """
    return _make_llm(WORKER_MODEL)