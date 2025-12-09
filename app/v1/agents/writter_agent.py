import os
import json
import logging
from typing import Dict, Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

from app.schemas.chat_session import ChatSession
from app.schemas.memory import MemoryEntry

logger = logging.getLogger("WriterAgent")


class WriterAgent:
    """
    WriterAgent writes structured research papers based on:
    - Retrieved context from the SupervisorAgent
    - Template files under /templates
    """

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

    def __init__(self, llm):
        self.llm = llm

        self.agent = create_agent(
            model=self.llm,
            tools=[],  # Writer doesn't call tools
            system_prompt=self._system_prompt(),
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    max_tokens_before_summary=4000,
                    messages_to_keep=15
                )
            ]
        )

    def _system_prompt(self):
        return """
You are ELEN Writer Agent — a professional research writing model.

Rules:
- Use ONLY given context.
- Follow formatting in the template.
- If information is missing, write:

   [⚠ Missing data - further research needed]

- Final output MUST be clean, formatted Markdown unless template specifies otherwise.
"""

    def _load_template(self, name: str) -> Optional[str]:
        if not name:
            return None

        filename = f"{name.strip().upper()}.txt"
        filepath = os.path.join(self.TEMPLATE_DIR, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    async def run(
        self,
        user_id: str,
        chat_id: str,
        context: Dict[str, Any],
        format_name: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:

        logger.info(f"[WriterAgent] Generating paper in format: {format_name}")

        # Ensure chat session exists
        chat = await ChatSession.find_one({"chat_id": chat_id}) or ChatSession(
            user_id=user_id, chat_id=chat_id
        )
        await chat.save()

        template = self._load_template(format_name)

        if not template:
            logger.warning(f"No template found: {format_name}")
            return {
                "error": "template_not_found",
                "available_templates": os.listdir(self.TEMPLATE_DIR)
            }

        request_payload = {
            "title": title,
            "template": template,
            "context": context
        }

        try:
            response = await self.agent.ainvoke({
                "messages": [
                    {"role": "user", "content": json.dumps(request_payload)}
                ]
            })

        except Exception as e:
            logger.error(f"[WriterAgent] LLM failure: {e}")
            return {"error": "llm_failure", "details": str(e)}

        # Extract clean text
        paper = response["messages"][-1].content if "messages" in response else str(response)

        # Store memory
        await MemoryEntry(
            user_id=user_id,
            chat_id=chat_id,
            key=f"paper_{format_name}",
            value={"title": title, "paper": paper},
        ).insert()

        return {"format": format_name, "title": title, "paper": paper}


# -------------------------------------------------------------------
# Streaming-Safe LangChain Tool
# -------------------------------------------------------------------

def build_writer_tool(writer_agent: WriterAgent):
    """
    This version supports streaming agents (astream).
    Arguments arrive structured (not raw JSON), so normalize safely.
    """

    @tool(
        name="write_paper",
        description="Generate a long-form research paper using context and templates."
    )
    async def writer_tool(
        *,
        user_id: str,
        chat_id: str,
        query: Optional[str] = None,
        context: Any = None,
        format: str = "IEEE",
        title: Optional[str] = None
    ) -> str:

        # Normalize context if model passes it as JSON string
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except Exception:
                context = {"raw": context}

        result = await writer_agent.run(
            user_id=user_id,
            chat_id=chat_id,
            context=context or {},
            format_name=format,
            title=title or query,
        )

        # MUST return clean JSON (not message objects)
        return json.dumps(result, ensure_ascii=False)

    return writer_tool
