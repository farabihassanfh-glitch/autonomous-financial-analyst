"""Citation-verification guardrail.

After the agent produces a briefing, a separate "compliance reviewer" Claude call
audits whether every factual claim is backed by a cited tool source. This catches
unsourced assertions and hallucinated figures before a briefing is trusted.
"""

from __future__ import annotations

import ast
import json
import re

from .config import get_chat_model


def _parse_tool_output(out: str):
    """Best-effort parse of a tool's stringified return value (dict or list).

    Tool results are dicts that get stringified into LangChain messages. A
    numpy/pandas scalar (e.g. np.float64(1.23)) breaks both json.loads and
    ast.literal_eval, silently turning a real warning into "nothing parsed" —
    so we sanitize that repr away before parsing as a defense-in-depth measure
    even though the tools now return plain Python floats at the source.
    """
    cleaned = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", out)
    for candidate in (out, cleaned):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:  # noqa: BLE001
                continue
    return None


def extract_data_caveats(tool_outputs: list[str]) -> list[str]:
    """Pull any data-quality warnings/errors the tools reported.

    Tools self-report limitations (e.g. a recent IPO's short history, a missing
    news key, null fields). This surfaces those to the user so a weak data input
    is never silently trusted — the systemic guard against "garbage in" answers.
    """
    caveats: list[str] = []
    for out in tool_outputs:
        data = _parse_tool_output(out)
        if isinstance(data, dict):
            for key in ("warning", "liquidity_warning", "product_type_warning"):
                if data.get(key):
                    caveats.append(str(data[key]))
            if data.get("status") == "error" and data.get("error"):
                caveats.append(f"Tool error: {data['error']}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("status") == "error":
                    caveats.append(f"Tool error: {item.get('error')}")
        else:  # unparseable string — regex fallback for the key signals
            m = re.search(r"LIMITED HISTORY[^\"'}]*", out)
            if m:
                caveats.append(m.group(0).strip())

    seen, unique = set(), []
    for c in caveats:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def assess_verdict_reliability(tool_outputs: list[str]) -> dict:
    """Decide whether the data actually supports an actionable recommendation.

    A confident Buy/Hold/Sell call is irresponsible when: the asset has too
    little trading history (recent IPO), core price/history data couldn't be
    retrieved at all (bad ticker or delisted security), liquidity is too thin
    for the price to be reliable, or the asset is a leveraged/inverse product
    the long-term framing doesn't fit. A failed *news* search alone does NOT
    block a verdict — that's a supplementary source, not core market data.

    Returns {"reliable": bool, "blockers": [...]}; when not reliable, the caller
    should show "Insufficient data for a recommendation" instead of the model's
    own verdict — enforced in code so a model that wants to be "helpful" can't
    talk around it.
    """
    blockers: list[str] = []
    for out in tool_outputs:
        data = _parse_tool_output(out)
        if not isinstance(data, dict):
            continue  # list-type outputs (e.g. news search) are supplementary
        if data.get("status") == "error" and data.get("error"):
            blockers.append(f"Could not retrieve core market data: {data['error']}")
        for key in ("warning", "liquidity_warning", "product_type_warning"):
            if data.get(key):
                blockers.append(str(data[key]))

    seen, unique = set(), []
    for b in blockers:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return {"reliable": len(unique) == 0, "blockers": unique}

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
