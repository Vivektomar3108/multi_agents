import os
import json
import logging
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

from app.schemas.chat_session import ChatSession
from app.schemas.memory import MemoryEntry

logger = logging.getLogger("WriterAgent")


class WriterAgent:
    """
    WriterAgent writes a full research paper based on:
    - Retrieved context (from supervisor)
    - ResearchAgent results (if provided)
    - A predefined paper format template stored in `/templates/*.txt`
    """

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

    def __init__(self, llm):
        self.llm = llm

        self.agent = create_agent(
            model=self.llm,
            tools=[],  # no external tools needed
            system_prompt=self._system_prompt(),
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    max_tokens_before_summary=4000,
                    messages_to_keep=15
                )
            ]
        )

    # --------------------------
    # System Prompt
    # --------------------------
    def _system_prompt(self):
        return """
You are **ELEN Writer Agent**, a professional academic writing system.

STRICT RULES:
- DO NOT hallucinate facts, numbers, or papers.
- Use ONLY the context provided.
- Follow the given research format EXACTLY.
- If context is insufficient, clearly mark missing sections with:

    `[⚠️ Missing data — further research required]`

YOU CAN WRITE:
- Full research papers
- Abstracts
- Literature reviews
- Academic structured content

Final output MUST be clean, formatted Markdown unless the template enforces a different formatting style.
"""

    # --------------------------
    # Load Template from Disk
    # --------------------------
    def _load_template(self, template_name: str) -> str | None:
        if not template_name:
            return None

        filename = f"{template_name.strip().upper()}.txt"
        path = os.path.join(self.TEMPLATE_DIR, filename)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # --------------------------
    # Run the Writer Agent
    # --------------------------
    async def run(
        self,
        user_id: str,
        chat_id: str,
        context: Dict[str, Any],
        format_name: str,
        title: str | None = None
    ) -> Dict[str, Any]:

        logger.info(f"[WriterAgent] Generating paper with format: {format_name}")

        # Ensure chat session exists
        chat = await ChatSession.find_one({"chat_id": chat_id}) or ChatSession(user_id=user_id, chat_id=chat_id)
        await chat.save()

        # Load template
        template_text = self._load_template(format_name)

        if not template_text:
            logger.warning(f"[WriterAgent] Template not found: {format_name}")
            return {"error": "template_not_found", "available_templates": os.listdir(self.TEMPLATE_DIR)}

        # Build input message for the LLM
        user_input = {
            "role": "user",
            "content": json.dumps({
                "title": title,
                "template": template_text,
                "context": context
            }, ensure_ascii=False)
        }

        # Generate response
        try:
            response = await self.agent.ainvoke({"messages": [user_input]})
        except Exception as e:
            logger.error(f"[WriterAgent] LLM Failed: {e}")
            return {"error": str(e), "paper": ""}

        # Get final content
        try:
            paper = response["messages"][-1].content
        except:
            paper = str(response)

        # Store final paper memory
        await MemoryEntry(
            user_id=user_id,
            chat_id=chat_id,
            key=f"paper_{format_name}",
            value={"title": title, "paper": paper}
        ).insert()

        return {"format": format_name, "title": title, "paper": paper}


# -------------------------------------------------------------------
# Register as a TOOL so the Supervisor can call it
# -------------------------------------------------------------------

def build_writer_tool(writer_agent: WriterAgent):

    @tool("write_paper")
    async def writer_tool(payload: str):
        """
        Expected payload JSON:
        {
            "user_id": "...",
            "chat_id": "...",
            "context": {...},
            "format": "IEEE",
            "title": "Optional title"
        }
        """

        data = json.loads(payload)

        result = await writer_agent.run(
            user_id=data["user_id"],
            chat_id=data["chat_id"],
            context=data.get("context", {}),
            format_name=data.get("format") or "",
            title=data.get("title"),
        )

        return json.dumps(result, ensure_ascii=False)

    return writer_tool
