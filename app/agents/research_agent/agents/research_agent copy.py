import json
import logging
from typing import List, Dict, Any, Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware
from datetime import datetime
from app.schemas.chat_session import ChatSession
from app.schemas.memory import MemoryEntry
from app.agents.research_agent.tools.tools import (
    arxiv_search,
    semantic_scholar_search,
    pubmed_search,
    duckduckgo_search,
    crossref_search,
)

from app.services.chat_history_service import ChatHistoryService
from app.config.weaviate_service import WeaviateService
from app.services.embedding_service import EmbeddingService  # <-- assumed embedding generator exists

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

        self.weaviate = WeaviateService()
        self.chat_history = ChatHistoryService()
        self.embedding_service = EmbeddingService()

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

    # -------------------------
    # 🔥 MEMORY + DATABASE SAVE
    # -------------------------
    async def _save_to_memory(self, user_id: str, chat_id: str, role: str, text: str, agent_name="research_agent"):
        """
        Saves message into MongoDB + Weaviate + embeddings automatically.
        """
        # Generate embeddings
        embedding = await self.embedding_service.embed(text)

        # Store in MongoDB chat history
        await self.chat_history.add_message(
            user_id=user_id,
            chat_id=chat_id,
            role=role,
            content=text,
            source=agent_name,
            embedding=embedding,
            metadata={"stored_at": datetime.utcnow().isoformat()}
        )

        # Store in Weaviate vector DB
        await self.weaviate.save_chat_turn(
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            embedding=embedding,
            role=role,
            agent=agent_name,
            metadata={"context_type": role}
        )

    # -------------------------
    # 🔹 FINAL RESPONSE FLOW
    # -------------------------
    async def run(self, user_id: str, chat_id: str, query: str) -> Dict[str, Any]:
        logger.info(f"[ResearchAgent] Running: {query}")

        # Save user query first
        await self._save_to_memory(user_id, chat_id, "user", query)

        chat = await ChatSession.find_one({"chat_id": chat_id}) or ChatSession(user_id=user_id, chat_id=chat_id)
        await chat.save()

        try:
            response = await self.agent.ainvoke({"messages": [{"role": "user", "content": query}]})
        except Exception as e:
            return {"error": str(e), "markdown": "", "results": []}

        self.last_agent_query = self._extract_response_query(response)
        markdown = self._extract_response_text(response)
        results = await self._extract_full_results()

        # Save assistant output
        await self._save_to_memory(user_id, chat_id, "assistant", markdown)

        if self.store_memory:
            await MemoryEntry(
                user_id=user_id, chat_id=chat_id, key="research_history",
                value={"query": query, "final_query": self.last_agent_query, "markdown": markdown, "results": results}
            ).insert()

        return {"markdown": markdown, "results": results}

    # -------------------------
    # STREAMING RESPONSE WITH MEMORY
    # -------------------------
    async def stream(self, user_id: str, chat_id: str, query: str):
        """
        Streaming version — memory stored after completion.
        """

        logger.info(f"[ResearchAgent] Streaming: {query}")

        # Save user query immediately
        await self._save_to_memory(user_id, chat_id, "user", query)

        try:
            full_response_text = ""

            async for chunk in self.agent.astream({"messages": [{"role": "user", "content": query}]}):
                text = ""

                if isinstance(chunk, dict) and "messages" in chunk:
                    msg = chunk["messages"][-1]
                    if hasattr(msg, "content"):
                        text = msg.content
                elif hasattr(chunk, "content"):
                    text = chunk.content

                full_response_text += text
                yield text  # stream chunk

            # After streaming — store assistant response
            await self._save_to_memory(user_id, chat_id, "assistant", full_response_text)

        except Exception as e:
            yield f"\n\n❌ ERROR: {str(e)}"

    # -------------------------
    # HELPER FUNCTIONS
    # -------------------------
    def _extract_response_query(self, response):
        content = self._extract_response_text(response)
        return content.split("\n")[0].replace("# 📄 Research Summary:", "").strip()

    def _extract_response_text(self, response):
        if isinstance(response, str): return response
        if isinstance(response, dict) and "messages" in response:
            return response["messages"][-1].content
        if hasattr(response, "content"): return response.content
        return str(response)

    async def _extract_full_results(self) -> List[dict]:
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


# ------------------------- TEST MODE -------------------------
if __name__ == "__main__":
    import asyncio, os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from app.config.mongo import init_db, close_db

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0,
        model_kwargs={"tool_choice": "auto"}
    )

    agent = ResearchAgent(llm)

    async def test():
        await init_db()
        result = await agent.run("usr", "test-id", "Top advancements in AI autonomous agents 2023-2025")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        await close_db()

    asyncio.run(test())
