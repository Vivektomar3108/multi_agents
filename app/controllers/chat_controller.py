import logging
import json
from datetime import datetime

from langchain_openai import ChatOpenAI

from app.agents.research_agent.agents.research_agent import ResearchAgent
from app.agents.research_agent.agents.writter_agent import WriterAgent
from app.agents.research_agent.agents.main import SupervisorAgent
from app.config.setting import settings


logger = logging.getLogger("AIChatController")


# =========================================================
# NON-STREAMING CONTROLLER
# =========================================================

class AIChatController:
    """
    Non-streaming version of the Supervisor controller.
    Always returns a full final response in one call.
    """

    def __init__(self):
        logger.info("🚀 Initializing Non-Streaming Supervisor Controller...")

        # Normal LLM (NO streaming)
        self.llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            streaming=False,     # << IMPORTANT
        )

        # Sub-agents
        research = ResearchAgent(self.llm)
        writer = WriterAgent(self.llm)

        # Supervisor orchestrator
        self.supervisor = SupervisorAgent(
            llm=self.llm,
            research_agent=research,
            writer_agent=writer
        )


    # =====================================================
    # HANDLE NON-STREAM REQUEST
    # =====================================================
    async def chat(
        self,
        user_id: str,
        chat_id: str,
        query: str
    ) -> dict:
        """
        Main single-shot non-streaming chat endpoint.
        Calls supervisor.run() and returns final Markdown.
        """

        logger.info(f"🧠 Non-Streaming Chat Request: {query}")

        try:
            result = await self.supervisor.run(
                user_id=user_id,
                chat_id=chat_id,
                user_query=query
            )

            # result = {"markdown": "..."} from supervisor
            response_text = result.get("markdown", "")

            return {
                "status": "success",
                "user_id": user_id,
                "chat_id": chat_id,
                "query": query,
                "response": result,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.exception("❌ Supervisor non-streaming request failed.")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
