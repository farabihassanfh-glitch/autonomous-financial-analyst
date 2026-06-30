"""Market-data and analysis tools the agent can call (its "actuators").

Each tool returns a structured dict and degrades gracefully on failure, returning
an error payload instead of raising — this is what lets the agent stay reactive
when a single data source is unavailable.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List

import yfinance as yf
from langchain_core.tools import tool

# Leveraged/inverse funds (2x/3x daily, "Inverse", "Bull"/"Bear", "Daily Target")
# decay over time by design and are short-term trading instruments, not
# buy-and-hold positions. A standard Buy/Hold/Sell framing doesn't apply to them.
_LEVERAGED_RE = re.compile(
    r"\b(\d(\.\d)?x|ultra|inverse|bear|bull|daily target|2x|3x)\b", re.IGNORECASE
)
LOW_LIQUIDITY_AVG_VOLUME = 100_000  # shares/day below which pricing is unreliable


@tool
def get_stock_price(ticker: str) -> Dict:
    """Return the current price and key metrics for a stock ticker.

    Use this for real-time pricing questions. Args: ticker e.g. 'AAPL', 'MSFT'.
    """
    try:
        info = yf.Ticker(ticker.upper()).info
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if price is None:
            return {"ticker": ticker.upper(), "status": "error",
                    "error": f"No price data for {ticker}; ticker may be invalid."}
        name = info.get("longName", info.get("shortName", ""))
        result = {
            "ticker": ticker.upper(),
            "company_name": name,
            "current_price": round(float(price), 2),
            "currency": info.get("currency", "USD"),
            "day_high": info.get("dayHigh", info.get("regularMarketDayHigh")),
            "day_low": info.get("dayLow", info.get("regularMarketDayLow")),
            "volume": info.get("volume", info.get("regularMarketVolume")),
            "market_cap": info.get("marketCap"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "success",
        }
        if name and _LEVERAGED_RE.search(name):
            result["product_type_warning"] = (
                f"LEVERAGED/INVERSE PRODUCT: '{name}' appears to be a leveraged, "
                f"inverse, or daily-reset fund. These decay over time by design and "
                f"are short-term trading instruments, not buy-and-hold positions. "
                f"A standard long-term Buy/Hold/Sell recommendation does not apply."
            )
        return result
    except Exception as e:  # noqa: BLE001 - tools must fail soft for the agent
        return {"ticker": ticker.upper(), "status": "error", "error": str(e)}


@tool
def get_stock_history(ticker: str, period: str = "3y") -> Dict:
    """Return historical performance and total return over a period.

    Args: ticker; period one of '1mo','3mo','6mo','1y','2y','3y','5y','10y'.
    """
    try:
        hist = yf.Ticker(ticker.upper()).history(period=period)
        hist = hist.dropna(subset=["Close"])
        if hist.empty:
            return {"ticker": ticker.upper(), "status": "error",
                    "error": f"No history for {ticker} over {period}."}
        start, end = hist["Close"].iloc[0], hist["Close"].iloc[-1]
        # Actual span available may be far shorter than requested (e.g. a recent
        # IPO). Report the true span and warn so the agent doesn't read a few days
        # of data as if it were a multi-year track record.
        span_days = (hist.index[-1] - hist.index[0]).days
        approx_years = round(span_days / 365.25, 2)
        result = {
            "ticker": ticker.upper(),
            "requested_period": period,
            "actual_span_days": span_days,
            "actual_span_years": approx_years,
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(float(start), 2),
            "end_price": round(float(end), 2),
            "return_pct": round(float((end - start) / start * 100), 2),
            "return_note": f"Return is over the actual {span_days} days available, "
                           f"not the requested {period}.",
            "high": round(float(hist["High"].max()), 2),
            "low": round(float(hist["Low"].min()), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "data_points": len(hist),
            "status": "success",
        }
        if span_days < 200:
            result["warning"] = (
                f"LIMITED HISTORY: only {len(hist)} trading days "
                f"(~{span_days} days) available, likely a recent IPO or new "
                f"listing. Do NOT treat this as long-term performance; judge it as "
                f"a newly listed security with a short track record."
            )
        if result["avg_volume"] < LOW_LIQUIDITY_AVG_VOLUME:
            result["liquidity_warning"] = (
                f"LOW LIQUIDITY: average volume is only ~{result['avg_volume']:,} "
                f"shares/day. Thinly traded securities can have unreliable pricing "
                f"and wide spreads; treat any return figure with caution."
            )
        return result
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker.upper(), "status": "error", "error": str(e)}


@tool
def search_financial_news(query: str) -> List[Dict]:
    """Search recent financial news for a query using the Tavily API.

    Requires TAVILY_API_KEY. Returns a list of articles (title, url, content).
    """
    if not os.getenv("TAVILY_API_KEY"):
        return [{"status": "error",
                 "error": "TAVILY_API_KEY not set; news search unavailable."}]
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        results = TavilySearchResults(max_results=5).invoke(query)
        return results
    except Exception as e:  # noqa: BLE001
        return [{"status": "error", "error": str(e)}]
