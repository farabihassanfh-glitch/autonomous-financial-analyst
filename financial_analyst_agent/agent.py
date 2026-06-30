"""The autonomous agent: a LangGraph state machine that loops between a Claude
reasoning node and a tool-execution node until the research goal is met."""

from __future__ import annotations

import logging
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import get_chat_model
from .tools import get_stock_history, get_stock_price, search_financial_news

logger = logging.getLogger(__name__)

CHARTER = """You are an autonomous Financial Research Analyst specializing in AI-focused companies.

PRIMARY GOAL: Produce a comprehensive, evidence-backed investment briefing for the
requested company, covering:
  1. Financial health — current price and multi-year performance trend
  2. Market sentiment — recent news and how to read it
  3. AI research activity — strategic initiatives (use the private database tool)
  4. Key risks and opportunities
  5. A clear recommendation with a confidence level

BEHAVIOR:
- Take initiative: gather all the data needed to meet the goal without being asked.
- Be reactive: if a tool returns an error, note the gap and continue with what you
  have rather than stopping.
- Be transparent: cite the tool behind every factual claim and state any data gaps.
- Never invent numbers — only report what the tools return.

WHEN TO REFUSE A VERDICT:
A Buy/Hold/Sell recommendation is a real-world decision-relevant statement. It is
irresponsible to issue one when the underlying data cannot support it. If a tool
result includes a "warning", "liquidity_warning", or "product_type_warning" field,
or if get_stock_price/get_stock_history returned an error, you MUST NOT give a
Buy/Hold/Sell verdict or a confidence percentage. Instead say plainly:
"INSUFFICIENT DATA FOR A RECOMMENDATION" and explain exactly what is missing
(e.g. too little trading history, unreliable/illiquid pricing, a leveraged or
inverse product the long-term framing doesn't fit, or the ticker/data could not
be retrieved at all) and what would need to be true for a recommendation to be
possible. Saying "I don't have enough information" is a complete, valid answer —
do not manufacture a verdict to seem more helpful than the data allows.
"""


class AgentState(TypedDict):
    """Conversation state shared across graph nodes; messages accumulate."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_agent(with_rag: bool = False, with_memory: bool = True):
    """Compile and return the LangGraph agent.

    Args:
        with_rag: include the private-database RAG tool (requires a built retriever).
        with_memory: enable conversation checkpointing per thread_id.
    """
    tools = [get_stock_price, get_stock_history, search_financial_news]
    if with_rag:
        from .rag import query_private_database

        tools.append(query_private_database)

    model = get_chat_model().bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=CHARTER)] + list(state["messages"])
        response = model.invoke(messages)
        if getattr(response, "tool_calls", None):
            logger.info("Agent requested %d tool call(s)", len(response.tool_calls))
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue,
                                   {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver() if with_memory else None
    return workflow.compile(checkpointer=checkpointer)


def analyze(query: str, with_rag: bool = False, thread_id: str = "session-1") -> dict:
    """Run one query end-to-end and return the answer plus execution detail.

    Returns a dict with:
        answer        — the final briefing text
        tool_calls    — list of {"name", "args"} the agent invoked
        tool_outputs  — list of raw tool-result strings (for verification/audit)
        messages      — the full LangGraph message list
    """
    agent = build_agent(with_rag=with_rag, with_memory=True)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    messages = result["messages"]

    tool_calls, tool_outputs = [], []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            tool_calls.append({"name": tc["name"], "args": tc["args"]})
        if m.__class__.__name__ == "ToolMessage":
            tool_outputs.append(str(m.content))

    return {
        "answer": messages[-1].content,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "messages": messages,
    }


def run_query(query: str, with_rag: bool = False, thread_id: str = "session-1") -> str:
    """Convenience helper: run one query and return just the final answer text."""
    return analyze(query, with_rag=with_rag, thread_id=thread_id)["answer"]
