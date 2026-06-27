"""Autonomous Financial Research Analyst — an agentic research system built on
Claude + LangGraph with tool use and retrieval-augmented generation."""

from dotenv import load_dotenv

# Load .env on package import so env-var-dependent tools work no matter which
# submodule is imported first.
load_dotenv()

__version__ = "0.1.0"
