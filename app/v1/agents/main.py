import os
import json
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator  # keep AsyncGenerator
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
load_dotenv()
logger = logging.getLogger("SupervisorAgent")
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY")


# =========================================================
async def fetch_context_from_weaviate(
    query: str, user_id: str, chat_id: str, top_k: int = 6
) -> List[str]:
    weaviate_service = None
    try:
        s3 = S3Service()
        weaviate_service = WeaviateService()
        file_service = FileServiceBatch(s3, weaviate_service)
        results = await file_service.query(
            text=query, top_k=top_k, user_id=user_id, chat_id=chat_id
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


async def fetch_recent_history(
    user_id: str, chat_id: str, limit: int = 10
) -> List[Dict[str, Any]]:
    records = (
        await ChatHistory.find(
            ChatHistory.user_id == user_id, ChatHistory.chat_id == chat_id
        )
        .sort("-created_at").limit(limit).to_list()
    )
    return [{"role": rec.role, "content": rec.content} for rec in reversed(records)]


SUPERVISOR_PROMPT = """
You are **ELEN Supervisor Agent**, an intelligent autonomous orchestrator.
... (same as before) ...
"""


# =========================================================
def wrap_research_tool(research_agent: ResearchAgent):
    @tool(
        name_or_callable="research_web",
        description=("Use this tool to perform external academic research (arXiv, "
            "Semantic Scholar, PubMed, CrossRef, etc.) when stored knowledge is not enough."),
    )
    async def research_tool(query: str, user_id: str, chat_id: str) -> str:
        result = await research_agent.run(user_id=user_id, chat_id=chat_id, query=query)
        return json.dumps(result, ensure_ascii=False)

    return research_tool


def wrap_writer_tool(writer_agent: WriterAgent):
    @tool(
        name_or_callable="write_paper",
        description=(
            "Generate a structured long-form research paper in IEEE/ACM format "
            "based on context and query."
        ),
    )
    async def writer_tool(
        *,
        user_id: str,
        chat_id: str,
        query: str,
        format: str = "IEEE",
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Safe paper generation tool with guaranteed return format."""

        if context and isinstance(context, str):
            try:
                context = json.loads(context)
            except Exception:
                context = {}

        full_context = {"query": query, **(context or {})}

        response = await writer_agent.run(
            user_id=user_id,
            chat_id=chat_id,
            context=full_context,
            format_name=format,
            title=title,
        )

        return json.dumps(response, ensure_ascii=False)

    return writer_tool


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
                    max_tokens_before_summary=4500,
                    messages_to_keep=25,
                )
            ],
        )

    def _format_stream(self, content: str, type: str = "assistant_chunk", extra: Dict[str, Any] = None) -> str:
        """Convert raw token text into structured JSON for UI streaming."""
        payload = {
            "type": type,
            "sender": "supervisor",
            "timestamp": int(time.time()),
            "content": content,
            "internal": True  # ⛔ Prevent model from treating stream as tool call input
        }
        if extra:
            payload.update(extra)

        return json.dumps(payload, ensure_ascii=False)

    async def run(self, user_id: str, chat_id: str, user_query: str) -> AsyncGenerator[str, None]:
        logger.info(f"[SupervisorAgent] Processing: {user_query}")

        chat = await ChatSession.find_one(ChatSession.chat_id == chat_id) or ChatSession(
            user_id=user_id, chat_id=chat_id
        )
        await chat.save()

        await ChatHistory(
            user_id=user_id,
            chat_id=chat_id,
            role="user",
            content=user_query,
            source="user",
        ).insert()

        history = await fetch_recent_history(user_id, chat_id)
        weaviate_context = await fetch_context_from_weaviate(user_query, user_id, chat_id)

        context_payload = {"history": history, "retrieved_context": weaviate_context}

        full_output = ""
        async for chunk in self.agent.astream(
            {
                "messages": [
                    {"role": "system", "content": SUPERVISOR_PROMPT},
                    {"role": "system", "content": f"Context:\nHistory:\n{json.dumps(history, indent=2)}\n\nRetrieved:\n{json.dumps(weaviate_context, indent=2)}"},
                    {"role": "user", "content": f"Query: {user_query}"}
                ],
                "user_id": user_id,
                "chat_id": chat_id
            },
            stream_mode="messages",
        ):

            text = chunk.content if hasattr(chunk, "content") else str(chunk)

            if text:
                full_output += text
                yield self._format_stream(text, type="assistant_chunk")

        if full_output.strip():
            await ChatHistory(
                user_id=user_id,
                chat_id=chat_id,
                role="assistant",
                content=full_output,
                source="SupervisorAgent",
            ).insert()

            await MemoryEntry(
                user_id=user_id,
                chat_id=chat_id,
                key="supervisor_output",
                value={"query": user_query, "response": full_output},
            ).insert()

            yield self._format_stream(full_output, type="complete")

        return


# =========================================================
# TEST HARNESS (STREAMING)
if __name__ == "__main__":
    from app.config.mongo import init_db, close_db

    async def test():
        await init_db()

        llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            streaming=True,
        )

        research = ResearchAgent(llm)
        writer = WriterAgent(llm)

        supervisor = SupervisorAgent(llm, research_agent=research, writer_agent=writer)

        print("\n=========== STREAMING OUTPUT ===========\n")
        async for chunk in supervisor.run(
            user_id="demo",
            chat_id="session-001",
            user_query="latest research on quantum computing",
        ):
            print(chunk, flush=True)

        print("\n\n=========== END STREAM ===========\n")

        await close_db()

    asyncio.run(test())
