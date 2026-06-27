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


def run_query(query: str, with_rag: bool = False, thread_id: str = "session-1") -> str:
    """Convenience helper: run one query end-to-end and return the final answer."""
    agent = build_agent(with_rag=with_rag, with_memory=True)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    return result["messages"][-1].content
