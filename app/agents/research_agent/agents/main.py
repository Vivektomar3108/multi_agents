import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

from app.schemas.chat_session import ChatSession
from app.schemas.chat_history import ChatHistory
from app.schemas.memory import MemoryEntry

from app.services.pdf_service import FileServiceBatch
from app.config.aws_s3 import S3Service
from app.config.weaviate_service import WeaviateService

from app.agents.research_agent.agents.research_agent import ResearchAgent
from app.agents.research_agent.agents.writter_agent import WriterAgent


# =========================================================
# ENV + LOGGER
# =========================================================

load_dotenv()
logger = logging.getLogger("SupervisorAgent")
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY")


# =========================================================
# BUILT-IN WEAVIATE CONTEXT (NOT A TOOL)
# =========================================================

async def fetch_context_from_weaviate(
    query: str,
    user_id: str,
    chat_id: str,
    top_k: int = 6,
) -> List[str]:
    """
    Fetch relevant chunks from Weaviate for this user/chat.
    This always runs BEFORE the supervisor decides on tools.
    """
    weaviate_service: Optional[WeaviateService] = None
    try:
        s3 = S3Service()
        weaviate_service = WeaviateService()
        file_service = FileServiceBatch(s3, weaviate_service)

        results = await file_service.query(
            text=query,
            top_k=top_k,
            user_id=user_id,
            chat_id=chat_id,
        )

        chunks: List[str] = []
        for r in results or []:
            props = r.get("properties", {}) if isinstance(r, dict) else getattr(r, "properties", {})
            text = props.get("text") or props.get("content") or ""
            if text:
                chunks.append(text[:2000])
        return chunks

    except Exception as e:
        logger.error(f"⚠ Retrieval Error: {e}")
        return []

    finally:
        # best-effort cleanup for Weaviate client
        try:
            if weaviate_service and hasattr(weaviate_service, "client"):
                client = getattr(weaviate_service, "client")
                close_fn = getattr(client, "close", None) or getattr(client, "aclose", None)
                if close_fn:
                    maybe = close_fn()
                    if asyncio.iscoroutine(maybe):
                        await maybe
        except Exception:
            pass


async def fetch_recent_history(
    user_id: str,
    chat_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch last N messages for conversational continuity.
    Returns a simple [ {role, content}, ... ] list.
    """
    records = (
        await ChatHistory
        .find(ChatHistory.user_id == user_id, ChatHistory.chat_id == chat_id)
        .sort("-created_at")
        .limit(limit)
        .to_list()
    )

    return [
        {"role": rec.role, "content": rec.content}
        for rec in reversed(records)
    ]


# =========================================================
# SUPERVISOR SYSTEM PROMPT
# =========================================================

SUPERVISOR_PROMPT = """
You are **ELEN Supervisor Agent**, an intelligent autonomous orchestrator.

You DO NOT directly browse the web or write long papers yourself.
Instead, you:

1. Read:
   - user query
   - provided chat history
   - retrieved Weaviate context

2. Decide whether you need to use tools:
   - `research_web` → external scientific research via sub-agent
   - `write_paper` → autonomous writer agent to generate formatted research papers

3. Combine:
   - Weaviate context
   - Conversation history
   - ResearchAgent output
   - WriterAgent output

4. Produce a final, clean Markdown answer.

---

### Tool Usage Rules

- Use **research_web** when:
  - Stored context is insufficient, outdated, or missing.
  - The user explicitly wants latest papers, citations, or scientific evidence.

- Use **write_paper** when:
  - The user explicitly wants a research paper or long-form structured academic output
    (e.g., "Write an IEEE paper on X", "Draft an ACM-style paper", etc.).

---

### Constraints

- NEVER hallucinate facts.
- ALWAYS use retrieved context when available.
- If critical information is missing, say so explicitly.
- Prefer minimal tool calls; do not overuse them.

---

### Output Format (Markdown)

# 🧠 Final Answer
Direct, concise answer to the user query.

## 📚 Retrieved Knowledge
Short bullet list summarizing relevant context from Weaviate + history.

## 🔍 New Research (only if `research_web` was used)
Key takeaways + citations from new research.

## 📝 Written Output (only if `write_paper` was used)
The structured paper or section produced by the writer agent.

## 🔗 Sources
List of URLs, DOIs, or references, if available.

---

Think step-by-step and decide carefully which tools (if any) are needed.
"""


# =========================================================
# TOOL WRAPPERS
# =========================================================

def wrap_research_tool(research_agent: ResearchAgent):
    @tool(
        name_or_callable="research_web",
        description=(
            "Use this tool to perform external academic research (arXiv, "
            "Semantic Scholar, PubMed, CrossRef, etc.) when stored knowledge is not enough."
        ),
    )
    async def research_tool(query: str, user_id: str, chat_id: str) -> str:
        """
        Perform external academic research on the given query.
        Returns JSON with fields such as markdown and links.
        """
        result = await research_agent.run(
            user_id=user_id,
            chat_id=chat_id,
            query=query,
        )
        return json.dumps(result, ensure_ascii=False)

    return research_tool


def wrap_writer_tool(writer_agent: WriterAgent):
    @tool(
        name_or_callable="write_paper",
        description=(
            "Use this tool to generate a structured research paper in a given format "
            "(e.g., IEEE, ACM, APA) using the available context."
        ),
    )
    async def writer_tool(
        user_id: str,
        chat_id: str,
        query: str,
        format: str = "IEEE",
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a research paper or long-form academic document.

        Args:
            user_id: User identifier for persistence.
            chat_id: Chat identifier for linking history.
            query: High-level topic or instructions for the paper.
            format: Name of the template (e.g., 'IEEE', 'ACM', 'APA').
            title: Optional custom title of the paper.
            context: Optional dict with:
                - history: list of prior messages
                - retrieved_context: list of text chunks
                - research_results: any prior research_web output
        """
        if context is None:
            context = {}

        # We pass the full merged context + all instructions to the writer agent
        full_context: Dict[str, Any] = {
            "query": query,
            **context,
        }

        result = await writer_agent.run(
            user_id=user_id,
            chat_id=chat_id,
            context=full_context,
            format_name=format,
            title=title,
        )
        return json.dumps(result, ensure_ascii=False)

    return writer_tool


# =========================================================
# SUPERVISOR IMPLEMENTATION
# =========================================================

class SupervisorAgent:
    def __init__(self, llm: ChatOpenAI, research_agent: ResearchAgent, writer_agent: WriterAgent):
        self.llm = llm

        self.tools = [
            wrap_research_tool(research_agent),
            wrap_writer_tool(writer_agent),
        ]

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SUPERVISOR_PROMPT,
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    trigger=('tokens', 4500),
                    keep=('messages', 25),
                )
            ],
        )

    async def run(self, user_id: str, chat_id: str, user_query: str) -> Dict[str, Any]:
        logger.info(f"[SupervisorAgent] Processing: {user_query}")

        # =========================================================
        # SAFE SESSION ENSURE — BEANIE COMPATIBLE
        # =========================================================
        from pymongo.errors import DuplicateKeyError

        chat = await ChatSession.find_one(ChatSession.chat_id == chat_id)

        if chat is None:
            chat = ChatSession(user_id=user_id, chat_id=chat_id)
            try:
                await chat.insert()
            except DuplicateKeyError:
                chat = await ChatSession.find_one(ChatSession.chat_id == chat_id)


        # Log the user message into history
        await ChatHistory(
            user_id=user_id,
            chat_id=chat_id,
            role="user",
            content=user_query,
            source="user",
        ).insert()

        # Collect context
        history = await fetch_recent_history(user_id, chat_id)
        weaviate_context = await fetch_context_from_weaviate(user_query, user_id, chat_id)

        context_payload = {
            "history": history,
            "retrieved_context": weaviate_context,
        }

        # Invoke supervisor agent with context + user query
        result = await self.agent.ainvoke({
            "messages": [
                {"role": "system", "content": json.dumps(context_payload)},
                {"role": "user", "content": user_query},
            ],
            "user_id": user_id,
            "chat_id": chat_id,
        })

        print(result)

        # Extract final message
        try:
            final_text = result["messages"][-1].content
        except Exception:
            final_text = str(result)

        # Store assistant message
        await ChatHistory(
            user_id=user_id,
            chat_id=chat_id,
            role="assistant",
            content=final_text,
            source="SupervisorAgent",
        ).insert()

        # Optional: store high-level memory if you want
        await MemoryEntry(
            user_id=user_id,
            chat_id=chat_id,
            key="supervisor_output",
            value={"query": user_query, "response": final_text},
        ).insert()

        return result


# =========================================================
# TEST HARNESS
# =========================================================

if __name__ == "__main__":
    from app.config.mongo import init_db, close_db

    async def test():
        await init_db()

        llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.1,
        )

        research = ResearchAgent(llm)
        writer = WriterAgent(llm)

        supervisor = SupervisorAgent(llm, research_agent=research, writer_agent=writer)

        result = await supervisor.run(
            user_id="demo",
            chat_id="session-001",
            user_query="write research paper this.",
        )

        print("\n=========== FINAL OUTPUT ===========\n")
        print(result["markdown"])

        print("\n=========== FINAL OUTPUT ===========\n")
        print(result)

        await close_db()

    asyncio.run(test())
