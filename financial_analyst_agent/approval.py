"""Human-in-the-loop sign-off for high-stakes recommendations.

In a regulated setting a Buy/Sell call shouldn't ship without a human analyst's
approval. This module decides when sign-off is required and records the decision
(approver + timestamp) to an audit log. It is framework-agnostic: the CLI calls
``request_signoff_cli`` (interactive), while a web UI can call ``record_decision``
directly from an Approve/Reject button.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

AUDIT_PATH = "data/signoff_audit.jsonl"

_HIGH_STAKES_RE = re.compile(r"\b(STRONG BUY|BUY|SELL|STRONG SELL)\b", re.IGNORECASE)


def requires_signoff(answer: str) -> bool:
    """True if the briefing contains an actionable Buy/Sell recommendation."""
    return bool(_HIGH_STAKES_RE.search(answer))


def record_decision(ticker: str, action: str, approved: bool, approver: str,
                    note: str = "", path: str = AUDIT_PATH) -> dict:
    """Append a sign-off decision to the audit log and return the record."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker.upper() if ticker else None,
        "action": action.upper() if action else None,
        "decision": "APPROVED" if approved else "REJECTED",
        "approver": approver,
        "note": note,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def request_signoff_cli(answer: str, ticker: str, action: str) -> dict:
    """Interactively ask a human to approve/reject a recommendation (CLI)."""
    print("\n" + "=" * 70)
    print(f"⚖️  HUMAN SIGN-OFF REQUIRED — recommendation: {action} {ticker}")
    print("=" * 70)
    approver = input("Approver name: ").strip() or "anonymous"
    choice = input("Approve this recommendation? [y/N]: ").strip().lower()
    approved = choice in ("y", "yes")
    note = "" if approved else (input("Reason for rejection (optional): ").strip())
    rec = record_decision(ticker, action, approved, approver, note)
    print(f"\nLogged: {rec['decision']} by {approver} at {rec['timestamp']}")
    if not approved:
        print("⛔ Recommendation NOT finalized — flagged for revision.")
    return rec
