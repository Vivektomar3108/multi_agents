import os
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.agents.core.agent_runtime import AgentRuntime
from app.agents.research_agent.agents.research_agent import ResearchAgent


class ResearchController:
    """
    Controller that wraps the ResearchAgent inside AgentRuntime so
    memory, summarization, forgetting and semantic linking happen automatically.
    """

    def __init__(self):
        load_dotenv()

        # -----------------------------
        # Initialize LLM
        # -----------------------------
        self.llm = ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            temperature=0,
            model_kwargs={"tool_choice": "auto"}
        )

        # -----------------------------
        # Attach ResearchAgent to Runtime
        # -----------------------------
        research_agent = ResearchAgent(self.llm)
        self.runtime = AgentRuntime(agent=research_agent, llm=self.llm)

    # -----------------------------
    # NON-STREAM RESPONSE
    # -----------------------------
    async def run_research(self, user_id: str, chat_id: str, query: str) -> dict:
        if not query:
            raise HTTPException(400, "Query cannot be empty")

        result = await self.runtime.run(
            user_id=user_id,
            chat_id=chat_id,
            query=query
        )

        return {
            "status": "success",
            "mode": "full",
            "chat_id": chat_id,
            "data": result
        }

    # -----------------------------
    # STREAMING RESPONSE
    # -----------------------------
    async def stream_research(self, user_id: str, chat_id: str, query: str):
        if not query:
            raise HTTPException(400, "Query cannot be empty")

        async for chunk in self.runtime.stream(
            user_id=user_id,
            chat_id=chat_id,
            query=query
        ):
            yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"
