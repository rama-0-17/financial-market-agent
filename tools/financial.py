"""
Financial data tool — wraps yfinance for completely free market data.

Provides:
  - Stock price + momentum metrics
  - Income statement (revenue, margins, growth)
  - Balance sheet (cash, debt, ratios)
  - Earnings history + surprises
  - Competitor snapshot comparison
"""
from __future__ import annotations
import json
import time
import yfinance as yf
import pandas as pd
from typing import Any
import logging

# Configure logging for yfinance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_get(obj: Any, *keys, default=None) -> Any:
    """Safely chain dict/attr lookups."""
    val = obj
    for k in keys:
        try:
            val = val[k] if isinstance(val, dict) else getattr(val, k, None)
            if val is None:
                return default
        except (KeyError, TypeError, AttributeError):
            return default
    return val if val is not None else default


def _fmt(val: Any, as_pct: bool = False, decimals: int = 2) -> str:
    """Format a number for display, handling None/NaN gracefully."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    if as_pct:
        return f"{val * 100:.{decimals}f}%"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.{decimals}f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:.{decimals}f}M"
    return f"{val:.{decimals}f}"


# ── Core data functions ───────────────────────────────────────────────────────

def get_stock_overview(ticker: str) -> dict[str, Any]:
    """
    Pull a full company overview: price action, valuation, margins, growth.
    Returns a structured dict ready for the agent scratchpad.
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker.upper())
            
            # Try to get info with explicit error handling
            try:
                info = t.info
                if not info or info.get("regularMarketPrice") is None:
                    logger.warning(f"Attempt {attempt + 1}: Empty or invalid info for {ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
            except Exception as info_err:
                logger.warning(f"Attempt {attempt + 1}: Error fetching info for {ticker}: {info_err}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                info = {}

            # Price data
            try:
                hist = t.history(period="1y")
            except Exception as hist_err:
                logger.warning(f"Error fetching history for {ticker}: {hist_err}")
                hist = pd.DataFrame()

            price_now = _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice")
            price_52w_low = _safe_get(info, "fiftyTwoWeekLow")
            price_52w_high = _safe_get(info, "fiftyTwoWeekHigh")
            ytd_return = None
            if not hist.empty and price_now:
                try:
                    price_start = hist["Close"].iloc[0]
                    ytd_return = (price_now - price_start) / price_start
                except Exception:
                    pass

            result = {
                "ticker": ticker.upper(),
                "company_name": _safe_get(info, "longName", default=ticker),
                "sector": _safe_get(info, "sector", default="N/A"),
                "industry": _safe_get(info, "industry", default="N/A"),
                "description": (_safe_get(info, "longBusinessSummary") or "")[:500],

                # Valuation
                "market_cap": _safe_get(info, "marketCap"),
                "market_cap_fmt": _fmt(_safe_get(info, "marketCap")),
                "pe_ratio": _safe_get(info, "trailingPE"),
                "forward_pe": _safe_get(info, "forwardPE"),
                "ps_ratio": _safe_get(info, "priceToSalesTrailing12Months"),
                "pb_ratio": _safe_get(info, "priceToBook"),
                "ev_ebitda": _safe_get(info, "enterpriseToEbitda"),

                # Price
                "current_price": price_now,
                "52w_low": price_52w_low,
                "52w_high": price_52w_high,
                "ytd_return": ytd_return,
                "ytd_return_fmt": _fmt(ytd_return, as_pct=True) if ytd_return is not None else "N/A",
                "beta": _safe_get(info, "beta"),

                # Financials
                "revenue_ttm": _safe_get(info, "totalRevenue"),
                "revenue_ttm_fmt": _fmt(_safe_get(info, "totalRevenue")),
                "gross_margin": _safe_get(info, "grossMargins"),
                "gross_margin_fmt": _fmt(_safe_get(info, "grossMargins"), as_pct=True),
                "operating_margin": _safe_get(info, "operatingMargins"),
                "operating_margin_fmt": _fmt(_safe_get(info, "operatingMargins"), as_pct=True),
                "net_margin": _safe_get(info, "profitMargins"),
                "net_margin_fmt": _fmt(_safe_get(info, "profitMargins"), as_pct=True),
                "revenue_growth": _safe_get(info, "revenueGrowth"),
                "revenue_growth_fmt": _fmt(_safe_get(info, "revenueGrowth"), as_pct=True),
                "earnings_growth": _safe_get(info, "earningsGrowth"),

                # Balance sheet
                "cash": _safe_get(info, "totalCash"),
                "cash_fmt": _fmt(_safe_get(info, "totalCash")),
                "debt": _safe_get(info, "totalDebt"),
                "debt_fmt": _fmt(_safe_get(info, "totalDebt")),
                "free_cash_flow": _safe_get(info, "freeCashflow"),
                "fcf_fmt": _fmt(_safe_get(info, "freeCashflow")),
                "roe": _safe_get(info, "returnOnEquity"),
                "roe_fmt": _fmt(_safe_get(info, "returnOnEquity"), as_pct=True),

                # Analyst
                "analyst_rating": _safe_get(info, "recommendationKey", default="N/A"),
                "target_price": _safe_get(info, "targetMeanPrice"),
                "analyst_count": _safe_get(info, "numberOfAnalystOpinions"),
            }
            
            # If we got here, return the result
            logger.info(f"Successfully fetched data for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                # Final fallback
                return {
                    "ticker": ticker, 
                    "error": str(e),
                    "company_name": ticker,
                    "market_cap_fmt": "N/A",
                    "pe_ratio": None,
                    "current_price": None,
                    "revenue_ttm_fmt": "N/A",
                }


def get_revenue_history(ticker: str, periods: int = 8) -> dict[str, Any]:
    """Pull quarterly revenue history to show growth trend."""
    try:
        t = yf.Ticker(ticker.upper())
        financials = t.quarterly_income_stmt

        if financials is None or financials.empty:
            return {"ticker": ticker, "error": "No income statement data available"}

        revenue_row = None
        for row_name in ["Total Revenue", "Revenue"]:
            if row_name in financials.index:
                revenue_row = financials.loc[row_name]
                break

        if revenue_row is None:
            return {"ticker": ticker, "error": "Revenue row not found"}

        revenue_row = revenue_row.sort_index()
        recent = revenue_row.tail(periods)

        quarters = []
        for date, val in recent.items():
            quarters.append({
                "quarter": date.strftime("%Y-Q%q") if hasattr(date, 'strftime') else str(date),
                "revenue": float(val) if not pd.isna(val) else None,
                "revenue_fmt": _fmt(float(val)) if not pd.isna(val) else "N/A",
            })

        # Compute YoY growth for latest quarter
        yoy_growth = None
        if len(recent) >= 5:
            latest = recent.iloc[-1]
            year_ago = recent.iloc[-5]
            if not pd.isna(latest) and not pd.isna(year_ago) and year_ago != 0:
                yoy_growth = (latest - year_ago) / abs(year_ago)

        return {
            "ticker": ticker.upper(),
            "quarters": quarters,
            "yoy_growth_latest": yoy_growth,
            "yoy_growth_fmt": _fmt(yoy_growth, as_pct=True) if yoy_growth is not None else "N/A",
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_competitor_comparison(tickers: list[str]) -> dict[str, Any]:
    """
    Fetch key metrics for multiple tickers side-by-side.
    Rate-limits with a small sleep to avoid Yahoo throttling.
    """
    results = {}
    metrics = [
        "market_cap_fmt", "pe_ratio", "forward_pe", "ps_ratio",
        "gross_margin_fmt", "operating_margin_fmt", "net_margin_fmt",
        "revenue_ttm_fmt", "revenue_growth_fmt", "ytd_return_fmt",
        "analyst_rating", "target_price", "beta",
    ]

    for ticker in tickers:
        data = get_stock_overview(ticker)
        results[ticker] = {m: data.get(m, "N/A") for m in metrics}
        results[ticker]["company_name"] = data.get("company_name", ticker)
        time.sleep(0.3)  # gentle throttle

    # Compute rankings for key metrics
    rankings = {}
    for metric in ["pe_ratio", "forward_pe", "gross_margin", "net_margin", "revenue_growth"]:
        vals = {}
        for ticker in tickers:
            t_data = get_stock_overview(ticker) if ticker not in results else None
            # Re-fetch raw (non-fmt) values for ranking
        # Rankings computed separately via raw overview data already fetched
        pass

    return {
        "tickers": tickers,
        "comparison": results,
    }


def get_earnings_history(ticker: str) -> dict[str, Any]:
    """Pull earnings surprise history — shows execution consistency."""
    try:
        t = yf.Ticker(ticker.upper())
        earnings = t.earnings_dates

        if earnings is None or earnings.empty:
            return {"ticker": ticker, "error": "No earnings data"}

        recent = earnings.head(8)
        history = []
        for date, row in recent.iterrows():
            eps_est = row.get("EPS Estimate")
            eps_act = row.get("Reported EPS")
            surprise = row.get("Surprise(%)")
            history.append({
                "date": str(date.date()) if hasattr(date, 'date') else str(date),
                "eps_estimate": float(eps_est) if eps_est and not pd.isna(eps_est) else None,
                "eps_actual": float(eps_act) if eps_act and not pd.isna(eps_act) else None,
                "surprise_pct": float(surprise) if surprise and not pd.isna(surprise) else None,
            })

        beats = sum(1 for h in history if h["surprise_pct"] and h["surprise_pct"] > 0)
        return {
            "ticker": ticker.upper(),
            "history": history,
            "beat_rate": beats / len(history) if history else None,
            "beat_rate_fmt": f"{beats}/{len(history)} quarters beat estimates",
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_price_history_for_chart(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Return OHLCV data suitable for charting."""
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period=period)
        if hist.empty:
            return {"ticker": ticker, "error": "No price history"}

        return {
            "ticker": ticker.upper(),
            "dates": [str(d.date()) for d in hist.index],
            "close": hist["Close"].round(2).tolist(),
            "volume": hist["Volume"].tolist(),
            "high": hist["High"].round(2).tolist(),
            "low": hist["Low"].round(2).tolist(),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ── LangChain Tool wrapper ────────────────────────────────────────────────────

from langchain_core.tools import tool


@tool
def financial_data_tool(ticker: str, data_type: str = "overview") -> str:
    """
    Fetch financial data for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'MSFT'
        data_type: One of 'overview', 'revenue_history', 'earnings', 'price_history'

    Returns:
        JSON string with requested financial data
    """
    ticker = ticker.upper().strip()

    if data_type == "overview":
        result = get_stock_overview(ticker)
    elif data_type == "revenue_history":
        result = get_revenue_history(ticker)
    elif data_type == "earnings":
        result = get_earnings_history(ticker)
    elif data_type == "price_history":
        result = get_price_history_for_chart(ticker)
    else:
        result = {"error": f"Unknown data_type: {data_type}. Use: overview, revenue_history, earnings, price_history"}

    return json.dumps(result, default=str, indent=2)


@tool
def competitor_comparison_tool(tickers: list[str]) -> str:
    """
    Compare multiple stocks side-by-side on key financial metrics.

    Args:
        tickers: List of ticker symbols, e.g. ['AAPL', 'MSFT', 'GOOGL']

    Returns:
        JSON string with comparison table
    """
    tickers = [t.upper().strip() for t in tickers]
    result = get_competitor_comparison(tickers)
    return json.dumps(result, default=str, indent=2)
