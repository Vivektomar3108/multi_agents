import os
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.agents.core.agent_runtime import AgentRuntime
from app.agents.research_agent.agents.writter_agent import WriterAgent


class WriterController:
    """
    Controller wrapping WriterAgent inside AgentRuntime.

    Features automatically handled by AgentRuntime:
    - Memory retrieval + contextual stitching
    - Relevance filtering
    - Summarization when history grows too large
    - Forgetting obsolete low-score memory
    - Smooth chaining after ResearchAgent work
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
        # Attach WriterAgent to AgentRuntime
        # -----------------------------
        writer_agent = WriterAgent(self.llm)
        self.runtime = AgentRuntime(agent=writer_agent, llm=self.llm)


    # -----------------------------
    # NON-STREAM MODE
    # -----------------------------
    async def write(self, user_id: str, chat_id: str, instruction: str) -> dict:
        """
        Expected input:
        instruction: str — plain natural language request

        Example:
        - "Write an IEEE-style research paper on EMG using our previous research"
        - "Improve and expand the methodology section"
        - "Rewrite abstract for clarity and formal tone"
        - "Generate a full thesis structure from our notes"
        """

        if not instruction or not isinstance(instruction, str):
            raise HTTPException(400, "`instruction` must be a non-empty string.")

        result = await self.runtime.run(
            user_id=user_id,
            chat_id=chat_id,
            query=instruction
        )

        return {
            "status": "success",
            "mode": "writer_full_response",
            "chat_id": chat_id,
            "query": instruction,
            "result": result
        }


    # -----------------------------
    # STREAMING MODE
    # -----------------------------
    async def stream_write(self, user_id: str, chat_id: str, instruction: str):
        """
        Same as `write()` but streamed token-by-token.

        This enables realtime UI writing (like ChatGPT typing).
        """

        if not instruction or not isinstance(instruction, str):
            raise HTTPException(400, "`instruction` must be a non-empty string.")

        async for chunk in self.runtime.stream(
            user_id=user_id,
            chat_id=chat_id,
            query=instruction
        ):
            yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"
