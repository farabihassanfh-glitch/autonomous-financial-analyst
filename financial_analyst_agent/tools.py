"""Market-data and analysis tools the agent can call (its "actuators").

Each tool returns a structured dict and degrades gracefully on failure, returning
an error payload instead of raising — this is what lets the agent stay reactive
when a single data source is unavailable.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import yfinance as yf
from langchain_core.tools import tool


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
        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", info.get("shortName")),
            "current_price": round(price, 2),
            "currency": info.get("currency", "USD"),
            "day_high": info.get("dayHigh", info.get("regularMarketDayHigh")),
            "day_low": info.get("dayLow", info.get("regularMarketDayLow")),
            "volume": info.get("volume", info.get("regularMarketVolume")),
            "market_cap": info.get("marketCap"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "success",
        }
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
        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(start, 2),
            "end_price": round(end, 2),
            "return_pct": round((end - start) / start * 100, 2),
            "high": round(hist["High"].max(), 2),
            "low": round(hist["Low"].min(), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "data_points": len(hist),
            "status": "success",
        }
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
