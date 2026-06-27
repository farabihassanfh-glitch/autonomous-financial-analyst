"""Citation-verification guardrail.

After the agent produces a briefing, a separate "compliance reviewer" Claude call
audits whether every factual claim is backed by a cited tool source. This catches
unsourced assertions and hallucinated figures before a briefing is trusted.
"""

from __future__ import annotations

import json
import re

from .config import get_chat_model

_REVIEWER_PROMPT = """You are a compliance reviewer auditing a financial briefing.

You are given (A) the raw outputs the analyst's tools returned, and (B) the
briefing the analyst wrote. Identify every factual/quantitative CLAIM in the
briefing (prices, returns, market caps, dated facts). For each, decide whether it
is supported by the tool outputs AND carries a source citation in the text.

Respond with ONLY a JSON object, no prose. "issues" must list ONLY claims that
are unsourced OR unsupported by the tool outputs — never list a claim that is
correctly sourced and supported. If every claim is fine, "issues" is an empty list.
{{
  "total_claims": <int>,
  "sourced_claims": <int how many are BOTH supported and cited>,
  "issues": ["<problem claim 1>", "<problem claim 2>"]
}}

### Tool outputs
{tool_outputs}

### Briefing
{briefing}
"""


def verify_citations(briefing: str, tool_outputs: list[str]) -> dict:
    """Audit a briefing for sourced claims.

    Returns {status, total_claims, sourced_claims, issues, citation_tags}.
    status is "pass" if every claim is sourced, else "review".
    """
    # Deterministic signal: count explicit [Source: ...] tags in the text.
    citation_tags = len(re.findall(r"\[Source[:\s]", briefing, flags=re.IGNORECASE))

    prompt = _REVIEWER_PROMPT.format(
        tool_outputs="\n".join(tool_outputs) or "(no tool outputs)",
        briefing=briefing,
    )
    try:
        raw = get_chat_model().invoke(prompt).content
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "citation_tags": citation_tags}

    total = int(data.get("total_claims", 0))
    sourced = int(data.get("sourced_claims", 0))
    issues = data.get("issues", []) or []
    return {
        "status": "pass" if total and sourced >= total and not issues else "review",
        "total_claims": total,
        "sourced_claims": sourced,
        "issues": issues,
        "citation_tags": citation_tags,
    }


def format_badge(result: dict) -> str:
    """Render a one-line human-readable badge from a verify_citations result."""
    if result.get("status") == "error":
        return f"⚠️  Verification failed: {result.get('error')}"
    total, sourced = result.get("total_claims", 0), result.get("sourced_claims", 0)
    if result.get("status") == "pass":
        return f"✅ Citations verified — {sourced}/{total} claims sourced"
    issues = result.get("issues", [])
    head = f"⚠️  {sourced}/{total} claims sourced — {len(issues)} flagged"
    return head + ("\n   - " + "\n   - ".join(issues) if issues else "")
