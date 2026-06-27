"""Recommendation logging and backtesting.

Every recommendation is appended to a JSONL log. Later, ``score_recommendations``
pulls the actual forward stock return from yfinance and measures whether the
agent's Buy/Hold/Sell calls were directionally correct — a real accuracy metric
over time, which most agent demos lack.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

LOG_PATH = "data/recommendations.jsonl"

_ACTION_RE = re.compile(r"\b(STRONG BUY|BUY|HOLD|SELL|STRONG SELL)\b", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence[^0-9]{0,15}(\d{1,3})\s*%", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def extract_recommendation(answer: str, ticker_hint: str | None = None) -> dict:
    """Parse the action (Buy/Hold/Sell) and confidence from a briefing."""
    action_match = _ACTION_RE.search(answer)
    conf_match = _CONF_RE.search(answer)
    return {
        "action": action_match.group(1).upper() if action_match else "UNKNOWN",
        "confidence_pct": int(conf_match.group(1)) if conf_match else None,
        "ticker": ticker_hint,
    }


def log_recommendation(ticker: str, action: str, query: str,
                       confidence_pct: int | None = None, path: str = LOG_PATH) -> dict:
    """Append a recommendation record to the JSONL log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker.upper(),
        "action": action.upper(),
        "confidence_pct": confidence_pct,
        "query": query,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _forward_return(ticker: str, since_iso: str, horizon_days: int) -> float | None:
    """Return the % change from the record date to ~horizon_days later."""
    try:
        start = datetime.fromisoformat(since_iso).date()
        hist = yf.Ticker(ticker).history(start=str(start)).dropna(subset=["Close"])
        if len(hist) < 2:
            return None
        entry = hist["Close"].iloc[0]
        idx = min(horizon_days, len(hist) - 1)
        later = hist["Close"].iloc[idx]
        return round((later - entry) / entry * 100, 2)
    except Exception:  # noqa: BLE001
        return None


def score_recommendations(path: str = LOG_PATH, horizon_days: int = 30) -> dict:
    """Score logged recommendations against actual forward returns.

    A Buy is "correct" if the forward return is positive; a Sell if negative;
    a Hold if the move stayed within +/-5%. Records too recent to evaluate
    (no horizon data yet) are skipped.
    """
    p = Path(path)
    if not p.exists():
        return {"status": "empty", "message": f"No log at {path}."}

    scored, correct, pending = [], 0, 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        fwd = _forward_return(rec["ticker"], rec["timestamp"], horizon_days)
        if fwd is None:
            pending += 1
            continue
        action = rec["action"]
        if "BUY" in action:
            hit = fwd > 0
        elif "SELL" in action:
            hit = fwd < 0
        else:  # HOLD / UNKNOWN
            hit = abs(fwd) <= 5
        correct += int(hit)
        scored.append({**rec, "forward_return_pct": fwd, "correct": hit})

    n = len(scored)
    return {
        "status": "ok",
        "evaluated": n,
        "pending": pending,
        "accuracy_pct": round(correct / n * 100, 1) if n else None,
        "horizon_days": horizon_days,
        "details": scored,
    }
