import json
import logging
from typing import List, Dict, Any, AsyncGenerator

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware

from app.schemas.chat_session import ChatSession
from app.schemas.memory import MemoryEntry
from app.agents.research_agent.tools.tools import (
    arxiv_search,
    semantic_scholar_search,
    pubmed_search,
    duckduckgo_search,
    crossref_search,
)

logger = logging.getLogger("ResearchAgent")


def _safe(fn):
    """Ensure tool failures never break the agent."""
    def wrapper(**kwargs):
        try:
            return json.dumps(fn(**kwargs), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    return wrapper


# --------------------- TOOL DEFINITIONS ---------------------

@tool(name_or_callable="arxiv_search")
def arxiv_tool(query: str, max_results: int = 5, **_):
    """Search scientific papers on ArXiv."""
    return _safe(arxiv_search)(query=query, max_results=max_results)


@tool(name_or_callable="semantic_scholar_search")
def semantic_scholar_tool(query: str, max_results: int = 5, **_):
    """Search Semantic Scholar database."""
    return _safe(semantic_scholar_search)(query=query, max_results=max_results)


@tool(name_or_callable="pubmed_search")
def pubmed_tool(query: str, max_results: int = 5, **_):
    """Search PubMed scientific library."""
    return _safe(pubmed_search)(query=query, max_results=max_results)


@tool(name_or_callable="duckduckgo_search")
def duckduckgo_tool(query: str, max_results: int = 5, **_):
    """Search general content from DuckDuckGo."""
    return _safe(duckduckgo_search)(query=query, max_results=max_results)


@tool(name_or_callable="crossref_search")
def crossref_tool(query: str, max_results: int = 5, **_):
    """Search DOI + Research metadata via CrossRef."""
    return _safe(crossref_search)(query=query, max_results=max_results)


TOOLS = [arxiv_tool, semantic_scholar_tool, pubmed_tool, duckduckgo_tool, crossref_tool]


# ------------------------- AGENT --------------------------

class ResearchAgent:

    def __init__(self, llm, max_results: int = 5, store_memory: bool = True):
        self.llm = llm
        self.max_results = max_results
        self.store_memory = store_memory
        self.last_agent_query = None

        self.tools = self._configure_tools(max_results)

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._system_prompt(),
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    max_tokens_before_summary=4000,
                    messages_to_keep=15
                )
            ]
        )

    def _configure_tools(self, max_results: int):

        for tool in TOOLS:
            original_func = tool.func

            def patched(func=original_func):
                def wrapper(**kwargs):
                    # Force tools to ALWAYS use the final LLM reformulated query
                    if self.last_agent_query:
                        kwargs["query"] = self.last_agent_query
                    kwargs.setdefault("max_results", max_results)
                    return func(**kwargs)

                return wrapper

            tool.func = patched()

        return TOOLS

    def _system_prompt(self):
        return """
You are a **Scientific Research Assistant**.
You MUST reformulate the query for academic databases before using tools.

Always generate a structured Markdown research report.

---

# 📄 Research Summary: {query}

## 🔍 Top Papers (Titles + Authors)
List concise entries.

## 🧠 Key Insights
2–4 sentence synthesis.

## 📎 Sources
One URL per line.

---
"""

    # ---------------------------------------------------------
    # 🆕 STREAMING VERSION — ADDED WITHOUT REMOVING ANY CODE
    # ---------------------------------------------------------
    async def stream(self, user_id: str, chat_id: str, query: str) -> AsyncGenerator[str, None]:
        """Live streaming version of the research agent for frontend SSE."""
        logger.info(f"[ResearchAgent STREAM] {query}")

        yield "🔍 Reformulating query...\n"

        full_markdown = ""

        async for event in self.agent.astream_events({"messages": [{"role": "user", "content": query}]}):
            kind = event.get("event")
            data = event.get("data")
            name = event.get("name")

            if kind == "on_chat_model_stream":
                chunk = getattr(data, "content", None)
                if chunk:
                    full_markdown += chunk
                    yield chunk

            elif kind == "on_tool_start":
                yield f"\n🛠 Running tool: `{name}`...\n"

            elif kind == "on_tool_end":
                yield f"✔️ `{name}` complete.\n"

        # parse finished result
        cleaned = self._extract_response_text(full_markdown)
        self.last_agent_query = cleaned.split("\n")[0].replace("# 📄 Research Summary:", "").strip()

        results = await self._extract_full_results()

        yield "\n📁 Finalizing results...\n"

        # store memory
        if self.store_memory:
            await MemoryEntry(
                user_id=user_id, chat_id=chat_id, key="research_history",
                value={"query": query, "final_query": self.last_agent_query, "markdown": cleaned, "results": results}
            ).insert()

        yield "\n🎉 Research complete!\n"

    # ---------------------------------------------------------
    # ORIGINAL RUN (STAYS UNTOUCHED)
    # ---------------------------------------------------------
    async def run(self, user_id: str, chat_id: str, query: str) -> Dict[str, Any]:
        logger.info(f"[ResearchAgent] Running: {query}")

        chat = await ChatSession.find_one({"chat_id": chat_id}) or ChatSession(user_id=user_id, chat_id=chat_id)
        await chat.save()

        try:
            response = await self.agent.ainvoke({"messages": [{"role": "user", "content": query}]})
        except Exception as e:
            return {"error": str(e), "markdown": "", "results": []}

        self.last_agent_query = self._extract_response_query(response)  # store refined query
        markdown = self._extract_response_text(response)
        results = await self._extract_full_results()

        if self.store_memory:
            await MemoryEntry(
                user_id=user_id, chat_id=chat_id, key="research_history",
                value={"query": query, "final_query": self.last_agent_query, "markdown": markdown, "results": results}
            ).insert()

        return {"markdown": markdown, "results": results}

    # -----------------------------------------------------------

    def _extract_response_query(self, response):
        """Extract final agent query rewrite."""
        content = self._extract_response_text(response)
        return content.split("\n")[0].replace("# 📄 Research Summary:", "").strip()

    def _extract_response_text(self, response):
        if isinstance(response, str): return response
        if isinstance(response, dict) and "messages" in response:
            return response["messages"][-1].content
        if hasattr(response, "content"): return response.content
        return str(response)

    async def _extract_full_results(self) -> List[dict]:
        """Returns full metadata including title, authors, summary, url, doi, etc."""
        results = []

        for tool in self.tools:
            try:
                raw = tool.func(query=self.last_agent_query, max_results=self.max_results)
                data = json.loads(raw)

                if isinstance(data, list):
                    results.extend([entry for entry in data if isinstance(entry, dict)])

            except Exception:
                continue

        return results
