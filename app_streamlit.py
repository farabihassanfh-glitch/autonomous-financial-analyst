"""Streamlit web UI for the Autonomous Financial Research Analyst.

Run locally:   streamlit run app_streamlit.py
Then open:     http://localhost:8501

Optional: set APP_PASSWORD in the environment to gate the app before anyone can
run a (paid) query — useful for a public Railway deployment.
"""

from __future__ import annotations

import importlib.util
import os

import streamlit as st

# On Streamlit Community Cloud, secrets are provided via st.secrets. Bridge them
# into environment variables so the (framework-agnostic) config layer picks them
# up the same way it reads a local .env file.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # no secrets file (e.g. local dev with .env) — fine

# RAG needs heavy optional deps (torch, chromadb). Detect whether they're
# installed so the public/slim deployment can disable the toggle gracefully.
RAG_AVAILABLE = importlib.util.find_spec("langchain_chroma") is not None

from financial_analyst_agent.agent import analyze
from financial_analyst_agent.approval import record_decision, requires_signoff
from financial_analyst_agent.backtest import (
    extract_recommendation,
    log_recommendation,
    score_recommendations,
)
from financial_analyst_agent.config import CHAT_MODEL
from financial_analyst_agent.verify import (
    assess_verdict_reliability,
    extract_data_caveats,
    verify_citations,
)

st.set_page_config(page_title="Autonomous Financial Analyst", page_icon="📈",
                   layout="wide")


# --------------------------------------------------------------------------- #
# Optional password gate (set APP_PASSWORD to enable)
# --------------------------------------------------------------------------- #
def check_password() -> bool:
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        return True  # no gate configured
    if st.session_state.get("authed"):
        return True
    st.title("📈 Autonomous Financial Analyst")
    pw = st.text_input("Enter access password", type="password")
    if pw and pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Model: `{CHAT_MODEL}`")
    use_verify = st.toggle("Citation verification", value=True,
                           help="Audit every claim for a cited tool source.")
    use_rag = st.toggle("Private RAG (PDFs in data/reports/)", value=False,
                        disabled=not RAG_AVAILABLE,
                        help="Retrieve from your private analyst PDFs."
                        if RAG_AVAILABLE
                        else "RAG deps not installed in this deployment — "
                             "available when running locally with the full "
                             "requirements.txt.")
    require_signoff = st.toggle("Require Buy/Sell sign-off", value=True,
                                help="Pause for human approval on actionable calls.")
    ticker_hint = st.text_input("Ticker (for logging)", value="",
                                placeholder="e.g. NVDA").strip().upper() or None
    st.divider()
    st.caption("Not financial advice. Educational portfolio project.")


st.title("📈 Autonomous Financial Research Analyst")
st.caption("A goal-oriented agent (Claude + LangGraph) that researches any ticker "
           "— stocks, ETFs, indices, crypto — end-to-end across price, history, "
           "news, and private documents, with citation guardrails and human sign-off.")

tab_analyze, tab_backtest, tab_about = st.tabs(
    ["🔍 Analyze", "📊 Backtest", "ℹ️ About & FAQ"])


# --------------------------------------------------------------------------- #
# Analyze tab
# --------------------------------------------------------------------------- #
with tab_analyze:
    examples = {
        "Analyze NVIDIA": "Analyze NVIDIA stock and tell me if it's a good investment.",
        "Compare MSFT vs GOOGL": "Compare Microsoft and Google as AI investments.",
        "Risks in Tesla": "What are the key risks of investing in Tesla right now?",
        "Should I buy AMD?": "Should I buy AMD? Give a Buy/Hold/Sell with confidence %.",
    }
    cols = st.columns(len(examples))
    for col, (label, q) in zip(cols, examples.items()):
        if col.button(label, use_container_width=True):
            st.session_state["query"] = q

    st.warning("⚠️ **Not financial advice.** This is an AI research demo. It can be "
               "wrong or work from incomplete data — verify everything before acting.")

    query = st.text_area("Your research question",
                         value=st.session_state.get("query", ""),
                         height=90, placeholder="e.g. Analyze NVIDIA stock...")

    if st.button("Run analysis", type="primary", disabled=not query.strip()):
        with st.spinner("Agent is researching — calling tools and synthesizing..."):
            result = analyze(query, with_rag=use_rag)
        rec = extract_recommendation(result["answer"], ticker_hint=ticker_hint)
        reliability = assess_verdict_reliability(result["tool_outputs"])
        # No verdict is shown if EITHER the tools flagged the data as too weak,
        # OR the model itself explicitly declined to give one (e.g. an asset with
        # no fundamentals, where the call depends on the user's own risk
        # tolerance rather than anything the tools could determine).
        no_verdict = (not reliability["reliable"]) or rec["action"] == "NO RECOMMENDATION"
        verification = (verify_citations(result["answer"], result["tool_outputs"])
                        if use_verify else None)
        # Don't log a recommendation to the backtest if there wasn't a real one —
        # that would silently grade a refusal as if it were an actual call.
        if not no_verdict and rec["action"] != "UNKNOWN" and rec["ticker"]:
            log_recommendation(rec["ticker"], rec["action"], query,
                               rec["confidence_pct"])
        # persist across reruns (so Approve/Reject buttons work)
        st.session_state["result"] = result
        st.session_state["rec"] = rec
        st.session_state["reliability"] = reliability
        st.session_state["no_verdict"] = no_verdict
        st.session_state["verification"] = verification
        st.session_state.pop("signoff", None)

    # render last result (survives button reruns)
    if "result" in st.session_state:
        result = st.session_state["result"]
        rec = st.session_state["rec"]
        reliability = st.session_state.get("reliability", {"reliable": True, "blockers": []})
        no_verdict = st.session_state.get("no_verdict", False)
        verification = st.session_state.get("verification")

        m1, m2, m3, m4 = st.columns(4)
        # Hard override: if there's no real verdict, the UI shows that —
        # regardless of what a naive keyword scan might have matched — so it
        # can never contradict the agent's own stated conclusion.
        if no_verdict:
            m1.metric("Recommendation", "NO RECOMMENDATION")
            m2.metric("Confidence", "N/A")
        else:
            m1.metric("Recommendation", rec["action"])
            conf_display = (f"{rec['confidence_pct']}%" if rec.get("confidence_pct")
                            else rec.get("confidence_label") or "—")
            m2.metric("Confidence", conf_display)
        m3.metric("Tool calls", len(result["tool_calls"]))
        if verification and verification.get("status") != "error":
            m4.metric("Claims sourced",
                      f"{verification['sourced_claims']}/{verification['total_claims']}")
        else:
            m4.metric("Claims sourced", "—")

        if no_verdict:
            if reliability["blockers"]:
                st.error("🚫 **No recommendation is shown.** The data does not "
                         "support a reliable Buy/Hold/Sell call:\n" +
                         "\n".join(f"- {b}" for b in reliability["blockers"]))
            else:
                st.info("ℹ️ **No Buy/Hold/Sell recommendation is shown.** The "
                        "agent determined this call depends on factors only you "
                        "can supply (e.g. personal risk tolerance or conviction "
                        "about an asset with no fundamentals to analyze) — see "
                        "its reasoning in the briefing below.")

        caveats = extract_data_caveats(result["tool_outputs"])
        if caveats:
            st.warning("**Data caveats the agent worked with** (weak inputs the "
                       "tools flagged):\n" + "\n".join(f"- {c}" for c in caveats))

        if verification:
            if verification.get("status") == "pass":
                st.success(f"✅ Citations verified — "
                           f"{verification['sourced_claims']}/"
                           f"{verification['total_claims']} claims sourced")
            elif verification.get("status") == "error":
                st.warning(f"Verification failed: {verification.get('error')}")
            else:
                st.warning(f"⚠️ {verification['sourced_claims']}/"
                           f"{verification['total_claims']} claims sourced — "
                           f"{len(verification['issues'])} flagged")
                for issue in verification["issues"]:
                    st.write(f"- {issue}")

        st.markdown(result["answer"])

        with st.expander("🔧 Tools the agent called"):
            for tc in result["tool_calls"]:
                st.write(f"- `{tc['name']}` {tc['args']}")

        # Human-in-the-loop sign-off (only meaningful if there's a real verdict)
        if not no_verdict and require_signoff and requires_signoff(result["answer"]):
            st.divider()
            st.subheader("⚖️ Human sign-off required")
            st.caption(f"Actionable call detected: **{rec['action']} "
                       f"{rec['ticker'] or '?'}** — approve before finalizing.")
            if "signoff" in st.session_state:
                d = st.session_state["signoff"]
                (st.success if d["decision"] == "APPROVED" else st.error)(
                    f"{d['decision']} by {d['approver']} at {d['timestamp']}")
            else:
                approver = st.text_input("Approver name", value="")
                c1, c2 = st.columns(2)
                if c1.button("✅ Approve", use_container_width=True):
                    st.session_state["signoff"] = record_decision(
                        rec["ticker"] or "?", rec["action"], True,
                        approver or "anonymous")
                    st.rerun()
                if c2.button("⛔ Reject", use_container_width=True):
                    st.session_state["signoff"] = record_decision(
                        rec["ticker"] or "?", rec["action"], False,
                        approver or "anonymous", note="rejected via web UI")
                    st.rerun()


# --------------------------------------------------------------------------- #
# Backtest tab
# --------------------------------------------------------------------------- #
with tab_backtest:
    st.caption("Score logged recommendations against actual forward stock returns.")
    horizon = st.slider("Forward horizon (trading days)", 5, 90, 30, step=5)
    if st.button("Run backtest"):
        r = score_recommendations(horizon_days=horizon)
        if r["status"] != "ok":
            st.info(r.get("message", "No recommendations logged yet — run some analyses first."))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Directional accuracy",
                      f"{r['accuracy_pct']}%" if r["accuracy_pct"] is not None else "—")
            c2.metric("Evaluated", r["evaluated"])
            c3.metric("Pending (too recent)", r["pending"])
            if r["details"]:
                st.dataframe(
                    [{"ticker": d["ticker"], "action": d["action"],
                      "forward_return_%": d["forward_return_pct"],
                      "correct": "✅" if d["correct"] else "❌"}
                     for d in r["details"]],
                    use_container_width=True,
                )


# --------------------------------------------------------------------------- #
# About & FAQ tab
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("What is this?")
    st.markdown(
        "An **autonomous AI research agent** that investigates a tradable asset "
        "end-to-end and produces an evidence-backed investment briefing. It works "
        "for **any Yahoo Finance ticker** — stocks, ETFs (e.g. IBIT, SPY), indices "
        "(^GSPC), and crypto (BTC-USD). You ask one question (\"Analyze IBIT\"); the "
        "agent decides on its own which tools to call — live price, multi-year "
        "history, recent news, and private documents — then synthesizes a "
        "recommendation with cited sources.\n\n"
        "It's built with **Claude** (Anthropic) for reasoning and **LangGraph** "
        "for the agent loop. This is a portfolio project — *not financial advice*."
    )

    st.subheader("How it works")
    st.markdown(
        "1. **You** ask a research question.\n"
        "2. **Claude** plans and calls tools as needed (it isn't told the steps).\n"
        "3. **Tools** fetch data: `get_stock_price`, `get_stock_history`, "
        "`search_financial_news`, and `query_private_database` (RAG over PDFs).\n"
        "4. The agent **loops** — tool results feed back in — until it has enough "
        "to write the briefing.\n"
        "5. **Guardrails** run: a citation check audits every claim, and any "
        "Buy/Sell call pauses for human sign-off."
    )

    st.subheader("The difference: agent vs. chatbot")
    st.markdown(
        "A chatbot answers the literal question. An **agent pursues a goal** — "
        "given \"analyze NVIDIA,\" it independently gathers price, trend, news, and "
        "research before recommending, looping through tools until done."
    )

    st.subheader("FAQ")
    with st.expander("What can it research?"):
        st.write("Any ticker Yahoo Finance recognizes — individual stocks, ETFs "
                 "(IBIT, SPY, QQQ), market indices (^GSPC, ^IXIC), and crypto "
                 "pairs (BTC-USD, ETH-USD). The market tools (price, history, "
                 "news) work for all of them. Private RAG is most useful when your "
                 "PDFs actually cover the asset you're asking about.")
    with st.expander("Is this financial advice?"):
        st.write("No. It's an educational/portfolio demonstration of agentic AI. "
                 "Do not make investment decisions based on it.")
    with st.expander("Where does the data come from?"):
        st.write("Live prices and history from Yahoo Finance (`yfinance`), recent "
                 "news from the Tavily search API, and — when Private RAG is on — "
                 "your own PDF documents in `data/reports/`.")
    with st.expander("What does 'Claims sourced' mean?"):
        st.write("After the agent answers, a separate Claude 'compliance reviewer' "
                 "audits every factual claim to confirm it's backed by a tool "
                 "output and carries a citation. The badge shows how many passed — "
                 "a guardrail against hallucinated numbers.")
    with st.expander("Why does it sometimes ask for sign-off?"):
        st.write("In a regulated setting, an actionable Buy/Sell shouldn't ship "
                 "without a human analyst's approval. When the agent makes such a "
                 "call, it pauses for Approve/Reject and logs the decision with an "
                 "approver and timestamp for audit.")
    with st.expander("What is the Backtest tab?"):
        st.write("Every recommendation is logged, then scored against the actual "
                 "forward stock return to measure the agent's directional accuracy "
                 "over time — a real evaluation metric, not just a demo.")
    with st.expander("What is Private RAG?"):
        st.write("Retrieval-Augmented Generation. Drop PDFs into `data/reports/`, "
                 "toggle it on, and the agent can answer from and cite those "
                 "private documents instead of relying on public knowledge.")
    with st.expander("Does it cost money to run?"):
        st.write("Each analysis makes a few Claude API calls (a few cents). The "
                 "embedding model used for RAG runs locally and is free.")

    st.divider()
    st.caption("Built by Syed Farabi Hassan · Claude + LangGraph · MIT licensed. "
               "Not financial advice.")
