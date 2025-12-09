import os
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.agents.core.agent_runtime import AgentRuntime
from app.agents.research_agent.agents.search_agent_test import SearchAgent


class SearchController:
    """
    Controller that wraps the SearchAgent inside AgentRuntime for
    automatic memory, summarization, forgetting and semantic linking.
    """

    def __init__(self):
        load_dotenv()

        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise RuntimeError("❌ Missing GROQ_API_KEY in environment variables!")
        
        # Single LLM for both agent and runtime (no JSON binding)
        self.llm = ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            temperature=0.3,
            model_kwargs={"tool_choice": "auto"}
        )

        # Initialize agent and runtime
        self.search_agent = SearchAgent(self.llm)
        self.runtime = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy initialization of the SearchAgent and Runtime."""
        if not self._initialized:
            await self.search_agent.initialize()
            self.runtime = AgentRuntime(agent=self.search_agent, llm=self.llm)
            self._initialized = True

    async def run_search(self, user_id: str, chat_id: str, query: str) -> dict:
        """Execute search with full memory context."""
        if not query:
            raise HTTPException(400, "Query cannot be empty")

        await self._ensure_initialized()

        result = await self.runtime.run(
            user_id=user_id,
            chat_id=chat_id,
            query=query
        )

        return {
            "status": "success",
            "mode": "full",
            "chat_id": chat_id,
            "data": {
                "response": result.get("response", ""),
                "urls": result.get("urls", []),
                "results": result.get("results", [])
            }
        }

    async def stream_research(self, user_id: str, chat_id: str, query: str):
        """Stream search results with memory context."""
        if not query:
            raise HTTPException(400, "Query cannot be empty")

        await self._ensure_initialized()

        async for chunk in self.runtime.stream(
            user_id=user_id,
            chat_id=chat_id,
            query=query
        ):
            yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"

