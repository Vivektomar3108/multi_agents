import json
import logging
from typing import Dict, Any, List, Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware

from app.services.chat_history_service import ChatHistoryService
from app.services.embedding_service import EmbeddingService
from app.config.weaviate_service import WeaviateService

from app.agents.research_agent.tools.tools import (
    arxiv_search,
    semantic_scholar_search,
    pubmed_search,
    duckduckgo_search,
    crossref_search,
)

logger = logging.getLogger("ResearchAgent")


# ---------------- SAFE TOOL WRAPPER ----------------
def _safe(fn):
    """Prevents tool failure from breaking execution."""
    def wrapper(**kwargs):
        try:
            return json.dumps(fn(**kwargs), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    return wrapper


# ---------------- TOOL DEFINITIONS ----------------
@tool("arxiv_search")
def arxiv_tool(query: str, max_results: int = 5, **_):
    """Search scientific papers from ArXiv."""
    return _safe(arxiv_search)(query=query, max_results=max_results)

@tool("semantic_scholar_search")
def semantic_scholar_tool(query: str, max_results: int = 5, **_):
    """Search Semantic Scholar."""
    return _safe(semantic_scholar_search)(query=query, max_results=max_results)

@tool("pubmed_search")
def pubmed_tool(query: str, max_results: int = 5, **_):
    """Search PubMed research."""
    return _safe(pubmed_search)(query=query, max_results=max_results)

@tool("duckduckgo_search")
def duckduckgo_tool(query: str, max_results: int = 5, **_):
    """Search general knowledge from DuckDuckGo."""
    return _safe(duckduckgo_search)(query=query, max_results=max_results)

@tool("crossref_search")
def crossref_tool(query: str, max_results: int = 5, **_):
    """Search DOI metadata from CrossRef."""
    return _safe(crossref_search)(query=query, max_results=max_results)

TOOLS = [arxiv_tool, semantic_scholar_tool, pubmed_tool, duckduckgo_tool, crossref_tool]


# ---------------- RESEARCH AGENT ----------------
class ResearchAgent:
    """
    Smart scientific assistant that adapts to users,
    switches modes (definition → research → implementation),
    remembers preference, and performs retrieval only when helpful.
    """

    def __init__(self, llm, max_results: int = 5):
        self.llm = llm
        self.max_results = max_results
        self.last_agent_query = None

        # Runtime services
        self.embedder = EmbeddingService()
        self.chat_store = ChatHistoryService()
        self.vector_store = WeaviateService()

        # Wrapped tools
        self.tools = self._configure_tools(max_results)

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._system_prompt(),
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    trigger=('tokens', 4000),
                    keep=('messages', 15)
                )
            ]
        )

    # ---------------- SYSTEM PROMPT ----------------
    def _system_prompt(self):
        return """
You are EVE — a Scientific Research Agent.
Your job: understand intent, decide if research tools are needed, gather evidence, and return clear, structured scientific answers.

--------------------------------------------------
1. INTENT DETECTION
--------------------------------------------------
Classify the user request:

{definition, research, implementation, comparison, follow_up, unknown}

If unknown → ask a clarification.

--------------------------------------------------
2. EXPERTISE LEVEL
--------------------------------------------------
Choose: beginner / intermediate / expert.
Adapt explanation accordingly.

--------------------------------------------------
3. TOOL USAGE RULES
--------------------------------------------------
Use research tools only when the user needs:
- papers
- citations
- SOTA methods
- factual verification

Do NOT use tools for simple explanations.

Sanitize tool queries:
- keep only first meaningful sentence
- max 200 chars
- remove markdown, logs, PII, chat history
- collapse whitespace

Never send the conversation content to tools.

--------------------------------------------------
4. ReAct LOOP (INTERNAL ONLY — NEVER SHOWN)
--------------------------------------------------
Thought: decide if tools are needed  
Action: <tool>  
Action Input: "<sanitized_query>"  
Observation: <result>  
Thought: interpret result  

Max 2 calls per service.

--------------------------------------------------
5. ANSWER FORMAT (VISIBLE)
--------------------------------------------------
🧠 Intent: <intent>  
🎓 Mode: <expertise>  

📄 Answer:  
<structured scientific explanation>  

If research/comparison:  
🔍 Sources:  
- <DOI / PubMed / arXiv> — <1-line summary>  

Max 5 sources.  
No hallucinated citations.

--------------------------------------------------
6. FOLLOW-UP QUESTION (OPTIONAL)
--------------------------------------------------
Ask one helpful research-oriented follow-up only when it adds value.

Example:
❓ Want deeper methodology, more papers, comparison, or implementation?

--------------------------------------------------
7. SAFETY
--------------------------------------------------
- No invented data  
- No fabricated citations  
- Keep reasoning hidden  
- State uncertainty if evidence is weak  

--------------------------------------------------
8. EXECUTION FLOW
--------------------------------------------------
1. Detect intent & expertise  
2. Decide tool usage  
3. If using tools → sanitize query + ReAct loop  
4. Produce structured answer + citations (if any)  
5. Optionally ask follow-up question
"""
    # ---------------- TOOL PATCHING ----------------
    def _configure_tools(self, max_results: int):
        for tool in TOOLS:
            original = tool.func
            def patched(func=original):
                def wrapper(**kwargs):
                    if self.last_agent_query:
                        kwargs["query"] = self.last_agent_query
                    kwargs.setdefault("max_results", max_results)
                    return func(**kwargs)
                return wrapper
            tool.func = patched()
        return TOOLS


    # ---------------- INTENT + USER MODE DETECTION ----------------
    def _extract_intent(self, text: str) -> str:
        if "Intent:" in text:
            return text.split("Intent:")[1].split("\n")[0].strip().lower()
        return "unknown"

    def _extract_user_mode(self, text: str) -> str:
        if "Mode:" in text:
            return text.split("Mode:")[1].split("\n")[0].strip().lower()
        return "intermediate"


    # ---------------- LEARNING ENGINE ----------------
    async def _update_learning(self, user_id, chat_id, intent, mode):
        await self.chat_store.add_message(
            user_id=user_id,
            chat_id=chat_id,
            role="assistant",
            content=f"intent:{intent}, mode:{mode}",
            metadata={"intent": intent, "user_mode": mode}
        )


    async def _infer_previous_intent(self, chat_id):
        msgs = await self.chat_store.get_last_messages(chat_id, 10)
        for m in reversed(msgs):
            intent = getattr(m.metadata, "intent", None)
            if intent:
                return intent
        return "definition"

    async def _infer_user_preference(self, chat_id):
        msgs = await self.chat_store.get_last_messages(chat_id, 25)
        score = 0
        for msg in msgs:
            if msg.role != "user": continue
            text = msg.content.lower()
            if "explain simply" in text or "easy" in text:
                score -= 1
            if "citations" in text or "compare models" in text:
                score += 1

        if score >= 2: return "expert"
        if score <= -2: return "beginner"
        return "intermediate"


    # ---------------- RUN MODE ----------------
    async def run(self, user_id: str, chat_id: str, query: str) -> Dict[str, Any]:

        # Build intelligent context
        previous_intent = await self._infer_previous_intent(chat_id)
        preference = await self._infer_user_preference(chat_id)

        enhanced_query = (
            f"Previous intent: {previous_intent}\n"
            f"User level: {preference}\n"
            f"User query: {query}"
        )

        response = await self.agent.ainvoke({"messages": [{"role": "user", "content": enhanced_query}]})
        reply = self._extract_text(response)

        intent = self._extract_intent(reply)
        mode = self._extract_user_mode(reply)

        # Detect whether search tools should run
        if intent in ("research", "follow_up"):
            self.last_agent_query = query

        await self._update_learning(user_id, chat_id, intent, mode)

        return {"response": reply, "intent": intent, "mode": mode}

    # ---------------- STREAM MODE ----------------
    async def stream(self, user_id: str, chat_id: str, query: str):
        previous_intent = await self._infer_previous_intent(chat_id)
        preference = await self._infer_user_preference(chat_id)

        enhanced_query = (
            f"Previous intent: {previous_intent}\n"
            f"User level: {preference}\n"
            f"User query: {query}"
        )

        buffer = ""
        try:
            async for chunk in self.agent.astream({"messages": [{"role": "user", "content": enhanced_query}]}):
                text = getattr(chunk, "content", "") or chunk.get("messages", [{}])[-1].content
                buffer += text
                yield text
        except Exception as e:
            yield f"\n\n❌ ERROR: {str(e)}"

        intent = self._extract_intent(buffer)
        mode = self._extract_user_mode(buffer)

        await self._update_learning(user_id, chat_id, intent, mode)


    # ---------------- HELPERS ----------------
    def _extract_text(self, resp):
        if isinstance(resp, str): return resp
        if isinstance(resp, dict) and "messages" in resp:
            return resp["messages"][-1].content
        if hasattr(resp, "content"): return resp.content
        return str(resp)
