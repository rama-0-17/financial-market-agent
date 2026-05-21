#!/usr/bin/env python3
"""
CLI runner — test the agent without starting the Streamlit UI.

Usage:
    python run.py "Analyse Apple vs Microsoft and Google"
    python run.py --ticker NVDA
    python run.py  # interactive mode
"""
from __future__ import annotations
import sys
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Financial Market Intelligence Agent CLI")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--ticker", type=str, help="Quick analyse a single ticker")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set. Add it to .env or get a free key at console.groq.com")
        sys.exit(1)

    if args.ticker:
        query = f"Analyse {args.ticker.upper()} and compare with its main competitors"
    elif args.query:
        query = args.query
    else:
        query = input("Enter research query: ").strip()
        if not query:
            print("No query entered. Exiting.")
            sys.exit(0)

    print(f"\n🚀 Running agent for: {query}\n{'─' * 60}")

    from agent.graph import run_agent
    import time

    start = time.time()
    try:
        state = run_agent(query)
        elapsed = time.time() - start

        print(f"\n{'═' * 60}")
        print(f"✅ Done in {elapsed:.1f}s · {len(state.tool_calls)} tool calls · {len(state.errors)} errors")
        print(f"{'═' * 60}\n")
        print(state.report)

        if state.errors:
            print(f"\n{'─' * 60}")
            print("⚠️  Errors during run:")
            for err in state.errors:
                print(f"  - {err}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\n❌ Agent error: {e}")
        raise


if __name__ == "__main__":
    main()