# app/agents/research_agent/agents/writter_agent.py  (keep your existing path/name)

import json
import logging
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from app.schemas.chat_session import ChatSession
from app.schemas.memory import MemoryEntry

logger = logging.getLogger("WriterAgent")


import os
from langchain.tools import tool


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


@tool("load_template")
def load_template_tool(template_name: str = "IEEE") -> str:
    """
    Load a research paper template by name from /templates directory.
    """

    if not template_name or not isinstance(template_name, str):
        return "[ERROR] Template name must be a non-empty string."

    filename = f"{template_name.strip().upper()}.txt"
    path = os.path.join(TEMPLATE_DIR, filename)

    if not os.path.exists(path):
        available = [
            f.replace(".txt", "") 
            for f in os.listdir(TEMPLATE_DIR) 
            if f.endswith(".txt")
        ]
        return f"[ERROR] Template '{template_name}' not found. Available: {available}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] Failed to load template: {e}"


class WriterAgent:
    """
    WriterAgent is a research-paper-focused writing & editing agent.

    It takes natural language instructions (like a query) and, using the
    conversation context + any embedded structured data (paper drafts,
    templates, research summaries), it can:

    - Draft a full research paper
    - Rewrite or polish existing sections
    - Extend sections (e.g., "add more to Related Work")
    - Change tone/format (e.g., "convert to IEEE-style paper")
    - Apply edits (e.g., "shorten the abstract", "make it more formal")

    Memory, retrieval, and additional context are handled by AgentRuntime.
    This agent only sees the `content` string passed to `run()` / `stream()`.
    """

    def __init__(self, llm):
        self.llm = llm

        self.agent = create_agent(
            model=self.llm,
            tools=[],  # purely generative, no external tools for writing
            system_prompt=self._system_prompt(),
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    trigger=('tokens', 4000),
                    keep=('messages', 15),
                )
            ],
        )

    # -------------------------------------------------
    # System Prompt — Research Paper Writer / Editor
    # -------------------------------------------------
    def _system_prompt(self):
        return r"""You are ELEN — a Professional Research Paper Writing & Editing Agent, functioning as a senior academic co-author.
Your job: write, extend, edit, or format research papers directly, not plan them (unless explicitly asked).

1. INPUT FORMAT

You will receive a single content string that may include:

User instructions (e.g., “write a research paper on …”)

Conversation snippets

Blocks such as:

CURRENT_PAPER

TEMPLATE

RESEARCH_CONTEXT

Optional runtime flag:

USER_REPEATED_REQUEST=TRUE


Interpret this flag as:
🔥 The user wants execution immediately, not planning.

2. INTERNAL INTENT CLASSIFICATION

(Never reveal this classification to the user.)

Classify the user’s request as one of:

create_paper

extend_paper

edit_section

revise_quality

reformat

summarize

If ambiguous → default to create_paper.

3. EXECUTION PRIORITY RULES
A. If USER_REPEATED_REQUEST=TRUE appears

Do NOT explain what you will do

Do NOT produce meta text

Do NOT output headings like:

“Proposed solution”

“Research structure”

“Plan”

Immediately write the paper or section requested.

B. If user says “write”, “draft”, “generate”, “give me the paper”

→ Produce the full written content, not an outline.

C. Only produce an outline IF the user explicitly asks

Examples: “give me an outline”, “plan it”, “show structure only”.

4. CONTEXT & FACTUAL SAFETY

You must use ONLY information found in:

RESEARCH_CONTEXT

CURRENT_PAPER

user instructions

templates (if given)

Do not invent:

Authors

Paper titles

DOIs

Datasets

Experimental numbers

Metrics

Statistical results

If the required data is missing, write:

[⚠️ Missing data — additional details needed to complete this section.]

5. OUTPUT FORMAT (Styled Markdown)

Unless the template overrides it, use this clean academic structure:

# Title

## Abstract

## 1. Introduction

## 2. Related Work / Literature Review

## 3. Methodology

## 4. Experiments / Results

## 5. Discussion

## 6. Conclusion

## References

Formatting Rules

Output clean, professional, formal Markdown

No emojis in the final paper

Smooth logical transitions

Bullet lists allowed where helpful

If editing/extending one section → output ONLY that section

If writing the whole paper → output the full structure

6. HONESTY & TRANSPARENCY

Never fabricate facts or citations

If context is insufficient → add transparent placeholders

Do NOT claim to have run experiments

Err on the side of accuracy and caution

7. CORE PRINCIPLE

Unless the user explicitly asks for an outline or plan,
➡️ Your job is to WRITE the research content itself.
"""

    # -------------------------------------------------
    # Helper: Extract Assistant Text from Response
    # -------------------------------------------------
    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        Normalize LangChain agent response into pure text.
        """
        if isinstance(response, str):
            return response

        if isinstance(response, dict) and "messages" in response:
            try:
                return response["messages"][-1].content
            except Exception:
                return str(response)

        if hasattr(response, "content"):
            return response.content

        return str(response)

    # -------------------------------------------------
    # NON-STREAM RUN (called by AgentRuntime)
    # -------------------------------------------------
    async def run(
        self,
        user_id: str,
        chat_id: str,
        content: str,   # this is now a "query-like" instruction + context
    ) -> Dict[str, Any]:

        logger.info(f"[WriterAgent] Running writer with instruction for chat_id={chat_id}")

        # Ensure chat session exists (keeps parity with other agents)
        chat = await ChatSession.find_one({"chat_id": chat_id}) or ChatSession(
            user_id=user_id, chat_id=chat_id
        )
        await chat.save()

        # Call LLM agent with the raw content
        try:
            response = await self.agent.ainvoke({"messages": [{"role": "user", "content": content}]})
            paper = self._extract_text(response)
        except Exception as e:
            logger.error(f"[WriterAgent] LLM invocation failed: {e}")
            return {"error": str(e), "response": ""}



        # Store final/last paper output into MemoryEntry (optional long-term store)
        try:
            await MemoryEntry(
                user_id=user_id,
                chat_id=chat_id,
                key="writer_last_output",
                value={"paper": paper},
            ).insert()
        except Exception as e:
            logger.error(f"[WriterAgent] Failed to store writer output in memory: {e}")

        return {"response": paper}

    # -------------------------------------------------
    # STREAMING VERSION (for SSE/WebSockets)
    # -------------------------------------------------
    async def stream(
        self,
        user_id: str,
        chat_id: str,
        content: str,
    ):
        """
        Streaming version — AgentRuntime handles when to call this.
        Yields text chunks progressively.
        """
        logger.info(f"[WriterAgent] Streaming writer output for chat_id={chat_id}")

        buffer = ""

        try:
            async for chunk in self.agent.astream({"messages": [{"role": "user", "content": content}]}):
                text = getattr(chunk, "content", "") or (
                    chunk.get("messages", [{}])[-1].content if isinstance(chunk, dict) else ""
                )
                buffer += text
                yield text
        except Exception as e:
            err = f"\n\n❌ ERROR: {str(e)}"
            logger.error(f"[WriterAgent] Streaming failed: {e}")
            yield err
            return

        # Optionally store full streamed result
        try:
            await MemoryEntry(
                user_id=user_id,
                chat_id=chat_id,
                key="writer_last_output",
                value={"paper": buffer},
            ).insert()
        except Exception as e:
            logger.error(f"[WriterAgent] Failed to store streamed writer output in memory: {e}")
