"""

Streamlit UI for the Financial Market Intelligence Agent.

Designed to be clean and demo-ready — shows:

  - The agent running in real-time (step-by-step status)

  - The final report with formatting

  - Tool call trace (the "under the hood" view)

  - Financial charts via plotly

  - Memory panel (past analyses)

"""

from __future__ import annotations

import sys

import os

import json

import time

# Add parent dir to path so imports work

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import plotly.graph_objects as go

import plotly.express as px

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(

page_title="Market Intelligence Agent",

page_icon="📊",

layout="wide",

initial_sidebar_state="expanded",

)

st.markdown("""

<style>

    /* Load Material Symbols icon font */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined';
        font-weight: normal;
        font-style: normal;
        font-size: 18px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
        vertical-align: middle;
        margin-right: 6px;
    }

    .icon-lg { font-size: 24px; }
    .icon-sm { font-size: 16px; }

    .stApp { background-color: #0f1117; color: #e0e0e0; }

    .metric-card {

        background: #1a1d27;

        border: 1px solid #2a2d3e;

        border-radius: 10px;

        padding: 16px 20px;

        margin: 6px 0;

    }

    .tool-call {

        background: #111827;

        border-left: 3px solid #3b82f6;

        padding: 8px 14px;

        margin: 4px 0;

        border-radius: 0 6px 6px 0;

        font-family: monospace;

        font-size: 13px;

    }

    .error-call {

        border-left-color: #ef4444;

    }

    .fallback-call {

        border-left-color: #f59e0b;

    }

    .section-header {

        color: #60a5fa;

        font-size: 18px;

        font-weight: 600;

        margin-top: 24px;

        margin-bottom: 8px;

        border-bottom: 1px solid #2a2d3e;

        padding-bottom: 6px;

    }

    .status-running { color: #60a5fa; }

    .status-done { color: #34d399; }

    .status-error { color: #f87171; }

    /* Example query buttons styling */
    div[data-testid="stHorizontalBlock"] .stButton > button,
    .example-btn > button {
        text-align: left !important;
        justify-content: flex-start !important;
    }

</style>

""", unsafe_allow_html=True)

# ── Helper: render icon + text heading ───────────────────────────────────────

def icon(name: str, css_class: str = "") -> str:
    """Return an HTML snippet for a Material Symbol icon."""
    extra = f" {css_class}" if css_class else ""
    return f'<span class="material-symbols-outlined{extra}">{name}</span>'


def icon_header(icon_name: str, text: str, level: int = 2) -> None:
    """Render a heading with a leading Material Symbol icon."""
    st.markdown(
        f'<h{level} style="display:flex;align-items:center;gap:8px;">'
        f'{icon(icon_name, "icon-lg")}{text}</h{level}>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:8px;">'
        f'{icon("settings", "icon-lg")}Settings</h2>',
        unsafe_allow_html=True,
    )

    # Check for API keys from secrets or environment
    groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

    if groq_key:
        st.success("✓ Groq API key loaded")
    else:
        st.error("✗ Groq API key not found. Set GROQ_API_KEY environment variable or in .streamlit/secrets.toml")
        os.environ["GROQ_API_KEY"] = st.text_input("Groq API Key", type="password", help="Get a free key at console.groq.com")

    if tavily_key:
        st.info("✓ Tavily API key loaded (optional)")
    else:
        st.caption("Tavily API key not configured (falls back to DuckDuckGo)")
        tavily_input = st.text_input("Tavily API Key (optional)", type="password", help="Free at tavily.com — 1000 searches/month")
        if tavily_input:
            os.environ["TAVILY_API_KEY"] = tavily_input

    st.divider()

    st.markdown(
        f'{icon("alt_route")}**Model routing**',
        unsafe_allow_html=True,
    )

    st.caption("Using Groq API for all model calls")

    st.divider()

    # Memory panel

    st.markdown(
        f'{icon("memory")}**Long-term memory**',
        unsafe_allow_html=True,
    )

    try:

        from memory.store import get_memory_count

        count = get_memory_count()

        st.metric("Stored analyses", count)

    except Exception:

        st.caption("Memory not initialised yet")

    if st.button("Clear memory", type="secondary"):

        import shutil

        if os.path.exists("./chroma_db"):

            shutil.rmtree("./chroma_db")

        st.success("Memory cleared")

        st.rerun()

# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown(
    f'<h1 style="display:flex;align-items:center;gap:10px;">'
    f'<span class="material-symbols-outlined" style="font-size: 45px;">monitoring</span> Financial Market Intelligence Agent</h1>',
    unsafe_allow_html=True,
)

st.caption("Powered by LangGraph · Groq · yfinance · Multi-step agentic reasoning")

# ── Session state init ────────────────────────────────────────────────────────

# "query_text" is the key used by st.text_input below — writing to it directly
# is the only reliable way to pre-populate the widget after first render.
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

# Example queries — clicking populates the text input

with st.expander("Example queries"):

    st.markdown(
        f'<span style="font-size: 13px; color: gray">{icon("lightbulb", "icon-sm")} <i>Click any example to load it into the search box</i></span>',
        unsafe_allow_html=True,
    )

    examples = [

        "Analyse Apple (AAPL) and compare against Microsoft and Google",

        "Is NVIDIA overvalued compared to AMD and Intel?",

        "Give me a market intelligence report on Tesla vs Ford and GM",

        "How is Amazon performing vs its main cloud and e-commerce rivals?",

        "Analyse Meta Platforms and compare with Snap and Pinterest",

    ]

    for ex in examples:

        if st.button(ex, key=f"example_{ex}", width="stretch"):

            st.session_state["query_text"] = ex

            st.rerun()

query = st.text_input(

    "Enter your research query",

    placeholder="e.g. Analyse Apple and compare it against Microsoft and Google",

    key="query_text",

)

col1, col2 = st.columns([1, 5])

with col1:

    run_btn = st.button(
        "Run Agent",
        type="primary",
        width="stretch",
        icon=":material/search:",
    )

with col2:

    if not os.getenv("GROQ_API_KEY"):

        st.warning("Add your Groq API key in the sidebar to run the agent.")

# ── Agent runner ──────────────────────────────────────────────────────────────

if run_btn and query and os.getenv("GROQ_API_KEY"):

    st.divider()

    # Status area

    status_area = st.empty()

    progress_bar = st.progress(0)

    # Output tabs — using Material icon names inline via markdown label trick
    tab_report, tab_tools, tab_charts, tab_debug = st.tabs([

        "Report", "Tool Trace", "Charts", "Debug"

    ])

    with status_area.container():

        st.markdown(
            f'<p class="status-running">{icon("hourglass_top")}Agent starting...</p>',
            unsafe_allow_html=True,
        )

    try:

        # Import here so API key is set first

        from agent.graph import run_agent

        from tools.financial import get_stock_overview, get_price_history_for_chart, get_revenue_history

        start_time = time.time()

        # Run agent with progress updates

        with status_area.container():

            st.markdown(
                f'<p class="status-running">{icon("psychology")}Recalling relevant past analyses...</p>',
                unsafe_allow_html=True,
            )

            progress_bar.progress(10)

        with status_area.container():

            st.markdown(
                f'<p class="status-running">{icon("checklist")}Planning analysis steps...</p>',
                unsafe_allow_html=True,
            )

            progress_bar.progress(20)

        # Actually run the agent

        final_state = run_agent(query)

        elapsed = time.time() - start_time

        progress_bar.progress(100)

        with status_area.container():

            st.markdown(

                f'<p class="status-done">'
                f'{icon("check_circle")}Complete in {elapsed:.1f}s · '

                f'{len(final_state.tool_calls)} tool calls · '

                f'{len(final_state.errors)} errors</p>',

                unsafe_allow_html=True,

            )

        # ── Report tab ────────────────────────────────────────────────────────

        with tab_report:

            if final_state.report:

                st.markdown(final_state.report)

            else:

                st.error("No report generated. Check the Debug tab for errors.")

            # Quick metrics row

            if final_state.primary_ticker:

                st.divider()

                st.markdown(
                    f'<h3 style="display:flex;align-items:center;gap:8px;">'
                    f'{icon("analytics")}Quick metrics</h3>',
                    unsafe_allow_html=True,
                )

                try:

                    overview = get_stock_overview(final_state.primary_ticker)

                    cols = st.columns(5)

                    metrics = [

                        ("Market Cap", overview.get("market_cap_fmt", "N/A")),

                        ("P/E Ratio", f"{overview.get('pe_ratio', 'N/A')}"),

                        ("Revenue (TTM)", overview.get("revenue_ttm_fmt", "N/A")),

                        ("Net Margin", overview.get("net_margin_fmt", "N/A")),

                        ("YTD Return", overview.get("ytd_return_fmt", "N/A")),

                    ]

                    for col, (label, value) in zip(cols, metrics):

                        col.metric(label, value)

                except Exception:

                    pass

        # ── Tool trace tab ────────────────────────────────────────────────────

        with tab_tools:

            st.markdown(
                f'<h3>{icon("build")}Tool call trace ({len(final_state.tool_calls)} calls)</h3>',
                unsafe_allow_html=True,
            )

            if final_state.plan:

                with st.expander("Execution plan", expanded=True):

                    st.markdown(
                        f'{icon("flag")}**Goal:** {final_state.plan.goal}',
                        unsafe_allow_html=True,
                    )

                    for i, step in enumerate(final_state.plan.steps, 1):

                        st.markdown(

                            f"{i}. "

                            f"**{step.tool}** "

                            f"→ `{step.args}`"

                        )

            for tc in final_state.tool_calls:

                css_class = "tool-call"

                if tc.error:

                    css_class += " error-call"

                icon_name = "check" if not tc.error else "close"

                attempt_note = f" (attempt {tc.attempt})" if tc.attempt > 1 else ""

                st.markdown(

                    f'<div class="{css_class}">'

                    f'<span class="material-symbols-outlined icon-sm">{icon_name}</span>'

                    f'<strong>{tc.tool}</strong>{attempt_note} · '

                    f'input: <code>{json.dumps(tc.input)[:80]}</code>'

                    f'{"<br><span class=\'material-symbols-outlined icon-sm\'>warning</span> Error: " + tc.error if tc.error else ""}'

                    f'</div>',

                    unsafe_allow_html=True,

                )

            if final_state.errors:

                st.markdown(
                    f'<h4>{icon("error")}Errors &amp; recovery</h4>',
                    unsafe_allow_html=True,
                )

                for err in final_state.errors:

                    st.error(err)

            if final_state.fallback_used:

                st.warning("Fallback tools were used for some steps")

            if final_state.replanned:

                st.info("Agent replanned mid-execution due to errors")

        # ── Charts tab ────────────────────────────────────────────────────────

        with tab_charts:

            ticker = final_state.primary_ticker

            all_tickers = [ticker] + final_state.competitor_tickers if ticker else []

            if ticker:

                st.markdown(
                    f'<h3>{icon("show_chart")}Price history — {ticker}</h3>',
                    unsafe_allow_html=True,
                )

                try:

                    price_data = get_price_history_for_chart(ticker)

                    if "error" not in price_data:

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(

                            x=price_data["dates"],

                            y=price_data["close"],

                            name=ticker,

                            line=dict(color="#3b82f6", width=2),

                            fill="tozeroy",

                            fillcolor="rgba(59,130,246,0.1)",

                        ))

                        fig.update_layout(

                            template="plotly_dark",

                            paper_bgcolor="rgba(0,0,0,0)",

                            plot_bgcolor="rgba(0,0,0,0)",

                            margin=dict(l=0, r=0, t=20, b=0),

                            height=300,

                        )

                        st.plotly_chart(fig, width="stretch")

                except Exception as e:

                    st.caption(f"Chart unavailable: {e}")

            # Comparison bar chart

            if len(all_tickers) > 1:

                st.markdown(
                    f'<h3>{icon("bar_chart")}Margin comparison</h3>',
                    unsafe_allow_html=True,
                )

                comp_data = []

                for t in all_tickers[:5]:

                    try:

                        ov = get_stock_overview(t)

                        if "error" not in ov:

                            comp_data.append({

                                "Ticker": t,

                                "Gross Margin": (ov.get("gross_margin") or 0) * 100,

                                "Operating Margin": (ov.get("operating_margin") or 0) * 100,

                                "Net Margin": (ov.get("net_margin") or 0) * 100,

                            })

                    except Exception:

                        pass

                if comp_data:

                    df = pd.DataFrame(comp_data)

                    fig2 = px.bar(

                        df.melt(id_vars="Ticker", var_name="Metric", value_name="Margin %"),

                        x="Ticker", y="Margin %", color="Metric", barmode="group",

                        template="plotly_dark",

                        color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],

                    )

                    fig2.update_layout(

                        paper_bgcolor="rgba(0,0,0,0)",

                        plot_bgcolor="rgba(0,0,0,0)",

                        margin=dict(l=0, r=0, t=20, b=0),

                        height=320,

                    )

                    st.plotly_chart(fig2, width="stretch")

        # ── Debug tab ─────────────────────────────────────────────────────────

        with tab_debug:

            st.json({

                "primary_ticker": final_state.primary_ticker,

                "competitor_tickers": final_state.competitor_tickers,

                "plan_steps": final_state.plan.steps if final_state.plan else [],

                "tool_call_count": len(final_state.tool_calls),

                "error_count": len(final_state.errors),

                "errors": final_state.errors,

                "fallback_used": final_state.fallback_used,

                "replanned": final_state.replanned,

                "retry_count": final_state.retry_count,

                "memory_ids_recalled": final_state.memory_ids,

                "report_sections": list(final_state.report_sections.keys()),

                "elapsed_s": round(elapsed, 2),

            })

    except Exception as e:

        progress_bar.progress(0)

        with status_area.container():

            st.markdown(
                f'<p class="status-error">{icon("error")}Agent failed: {e}</p>',
                unsafe_allow_html=True,
            )

        st.exception(e)

elif run_btn and not query:

    st.warning("Please enter a query first.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()

st.caption(

    "Built with LangGraph · Groq · yfinance · ChromaDB · Streamlit\n\n"

    "Financial data via Yahoo Finance. Not financial advice."

)