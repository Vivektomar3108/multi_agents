# app/agents/research_multiagent.py
import os
import asyncio
import logging
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_groq import ChatGroq

from app.agents.tools_v1 import (
    arxiv_search,
    load_pdf_and_chunk,
    load_url_and_chunk,
)
from app.schemas.memory import MemoryEntry
from app.config.chroma import get_vector_store
from langchain_anthropic import ChatAnthropic
import os

load_dotenv()
logger = logging.getLogger(__name__)

# Configure logger if not already configured by the application
if not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
logger.setLevel(logging.DEBUG)


# --------------------------------------------------------------
#  LLM (ChatGroq)
# --------------------------------------------------------------
logger.info("Initializing ChatGroq LLM...")

llm = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.05")),
    streaming=False,
)

logger.info("ChatGroq initialized with model=%s", os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"))


# ---------------------------
# Utility: safe LLM call
# ---------------------------
def _call_llm_sync(prompt: str) -> str:
    """
    Call the LLM in a defensive way. Some LLM wrappers expose .invoke(), .generate()
    or are callable. Try a few and return string output.
    """
    logger.debug("[LLM] _call_llm_sync called. prompt_len=%d", len(prompt) if prompt else 0)
    try:
        # prefer .invoke (used by DeepAgents examples)
        resp = llm.invoke(prompt)
        logger.debug("[LLM] invoke returned type=%s resp=%s", type(resp), str(resp)[:500])
        # If resp is dict-like with "output" or "content"
        if isinstance(resp, dict):
            return str(resp.get("output") or resp.get("content") or resp)
        if hasattr(resp, "content"):
            return str(getattr(resp, "content"))
        return str(resp)
    except Exception as e_invoke:
        logger.exception("[LLM] .invoke() failed: %s", e_invoke)
        try:
            gen = llm.generate(prompt)  # some wrappers
            logger.debug("[LLM] generate returned type=%s gen=%s", type(gen), str(gen)[:500])
            # generate() may return object; try to extract text
            if hasattr(gen, "generations"):
                return str(gen.generations)
            return str(gen)
        except Exception as e_gen:
            logger.exception("[LLM] .generate() failed: %s", e_gen)
            # last resort: callable
            try:
                resp_call = llm(prompt)
                logger.debug("[LLM] direct call returned type=%s resp=%s", type(resp_call), str(resp_call)[:500])
                return str(resp_call)
            except Exception as e_call:
                logger.exception("LLM call failed completely: %s", e_call)
                return ""


# --------------------------------------------------------------
# HYBRID MEMORY COMPONENTS
# --------------------------------------------------------------
class HybridMemory:
    """
    Hybrid memory combining:
      - Vector memory (Chroma via get_vector_store)
      - Short window (stored in Mongo MemoryEntry as JSON)
      - Summaries (LLM-generated, also stored in vector DB)
    """

    WINDOW_SIZE = 8

    @staticmethod
    async def store_message(user_id: str, chat_id: str, role: str, text: str) -> None:
        """
        Store a message into the vector DB (for semantic search) and into the short window.
        role: "user" | "agent" | "system"
        """
        logger.info("[HybridMemory] store_message called user=%s chat=%s role=%s text_len=%s",
                    user_id, chat_id, role, len(text) if text else 0)
        try:
            vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
            logger.debug("[HybridMemory] obtained vector store for collection=%s", f"user_{user_id}_chat_{chat_id}")
            # Chroma / LangChain wrapper expects plain strings
            vs.add_texts([f"{role.upper()}: {text}"])
            logger.debug("[HybridMemory] added text to vector store")
        except Exception as e:
            logger.exception("[HybridMemory] Failed to add to vector store for user=%s chat=%s: %s", user_id, chat_id, e)

        # persist in a MongoDB short-window entry
        await HybridMemory._store_window(user_id, chat_id, role, text)

    @staticmethod
    async def retrieve_relevant(user_id: str, chat_id: str, query: str, k: int = 4):
        """
        Perform semantic recall from Chroma vector store.
        Returns a list of Documents (LangChain Document objects) or raw items.
        """
        logger.info("[HybridMemory] retrieve_relevant called user=%s chat=%s query_len=%s k=%s",
                    user_id, chat_id, len(query) if query else 0, k)
        try:
            vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
            logger.debug("[HybridMemory] similarity_search on collection=%s", f"user_{user_id}_chat_{chat_id}")
            results = vs.similarity_search(query, k=k)
            logger.debug("[HybridMemory] similarity_search returned %s results", len(results) if results else 0)
            return results
        except Exception as e:
            logger.exception("[HybridMemory] Vector lookup failed for user=%s chat=%s: %s", user_id, chat_id, e)
            return []

    @staticmethod
    async def _store_window(user_id: str, chat_id: str, role: str, text: str):
        """
        Store a rolling window of recent messages in MongoDB (MemoryEntry).
        Keeps only last WINDOW_SIZE messages.
        """
        key = "short_window_messages"
        logger.debug("[HybridMemory] _store_window called user=%s chat=%s role=%s", user_id, chat_id, role)
        try:
            existing = await MemoryEntry.find_one(
                MemoryEntry.user_id == user_id,
                MemoryEntry.chat_id == chat_id,
                MemoryEntry.key == key
            )

            if existing:
                arr = existing.value.get("messages", [])
                logger.debug("[HybridMemory] existing short-window length=%s", len(arr))
                arr.append({"role": role, "text": text})
                arr = arr[-HybridMemory.WINDOW_SIZE:]
                existing.value = {"messages": arr}
                await existing.save()
                logger.debug("[HybridMemory] appended and saved short-window (now length=%s)", len(arr))
            else:
                logger.debug("[HybridMemory] creating new short-window record")
                await MemoryEntry(
                    user_id=user_id,
                    chat_id=chat_id,
                    key=key,
                    value={"messages": [{"role": role, "text": text}]}
                ).insert()
                logger.debug("[HybridMemory] created new short-window record")

        except Exception as e:
            logger.exception("[HybridMemory] Failed to store window message for user=%s chat=%s: %s", user_id, chat_id, e)

    @staticmethod
    async def get_window(user_id: str, chat_id: str) -> List[Dict[str, str]]:
        """
        Return the short-window messages (list of dicts with 'role' and 'text').
        """
        logger.debug("[HybridMemory] get_window called user=%s chat=%s", user_id, chat_id)
        try:
            entry = await MemoryEntry.find_one(
                MemoryEntry.user_id == user_id,
                MemoryEntry.chat_id == chat_id,
                MemoryEntry.key == "short_window_messages"
            )
            if not entry:
                logger.debug("[HybridMemory] no short-window entry found")
                return []
            messages = entry.value.get("messages", [])
            logger.debug("[HybridMemory] returning %s messages from window", len(messages))
            return messages
        except Exception as e:
            logger.exception("[HybridMemory] Failed to load window messages for user=%s chat=%s: %s", user_id, chat_id, e)
            return []

    @staticmethod
    async def summarize(user_id: str, chat_id: str) -> str:
        """
        Summarize the short-window messages via the LLM and store the summary in vector DB.
        Returns the summary string.
        """
        logger.info("[HybridMemory] summarize called user=%s chat=%s", user_id, chat_id)
        try:
            window = await HybridMemory.get_window(user_id, chat_id)
            if not window:
                logger.debug("[HybridMemory] summarize: empty window")
                return ""

            text = "\n".join([f"{m['role'].upper()}: {m['text']}" for m in window])
            logger.debug("[HybridMemory] summarize: window_text_len=%s", len(text))
            prompt = f"Summarize the following conversation in 6-10 concise lines with an academic tone:\n\n{text}"

            summary = _call_llm_sync(prompt) or ""
            logger.info("[HybridMemory] summarize produced len=%s", len(summary))

            # persist summary into vector store
            try:
                vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
                vs.add_texts([f"SUMMARY: {summary}"])
                logger.debug("[HybridMemory] summary added to vector DB")
            except Exception as e:
                logger.exception("[HybridMemory] Failed to persist summary to vector DB for user=%s chat=%s: %s", user_id, chat_id, e)

            return summary
        except Exception as e:
            logger.exception("[HybridMemory] Summarization failed for user=%s chat=%s: %s", user_id, chat_id, e)
            return ""


# --------------------------------------------------------------
# TOOL WRAPPERS (DOCUMENT LOADERS & ARXIV) - must have docstrings
# --------------------------------------------------------------
def arxiv_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv for a given query and return a list of metadata dicts.
    Args:
        query: search string
        max_results: number of results to fetch
    Returns:
        list of dicts: {title, authors, summary, published, url}
    """
    logger.info("[Tool] arxiv_tool called query=%s max_results=%s", query, max_results)
    try:
        result = asyncio.run(arxiv_search(query, max_results))
        logger.debug("[Tool] arxiv_tool result type=%s len=%s", type(result), len(result) if hasattr(result, "__len__") else "N/A")
        return result
    except Exception as e:
        logger.exception("[Tool] arxiv_tool failed for query=%s: %s", query, e)
        return []


def load_url_tool(url: str) -> List[Any]:
    """
    Load a web page (via Playwright) and return chunked LangChain Documents.
    Args:
        url: page URL
    Returns:
        list of Document objects (or their dict representation)
    """
    logger.info("[Tool] load_url_tool called url=%s", url)
    try:
        res = asyncio.run(load_url_and_chunk(url))
        logger.debug("[Tool] load_url_tool returned %s chunks", len(res) if hasattr(res, "__len__") else "N/A")
        return res
    except Exception as e:
        logger.exception("[Tool] load_url_tool failed for url=%s: %s", url, e)
        return []


def load_pdf_tool(file_path: str) -> List[Any]:
    """
    Load and chunk a local PDF using UnstructuredPDFLoader and return Documents.
    Args:
        file_path: local filesystem path to PDF
    Returns:
        list of Document objects (or their dict representation)
    """
    logger.info("[Tool] load_pdf_tool called file_path=%s", file_path)
    try:
        res = asyncio.run(load_pdf_and_chunk(file_path))
        logger.debug("[Tool] load_pdf_tool returned %s chunks", len(res) if hasattr(res, "__len__") else "N/A")
        return res
    except Exception as e:
        logger.exception("[Tool] load_pdf_tool failed for file_path=%s: %s", file_path, e)
        return []


def save_memory_tool(user_id: str, chat_id: str, message: str) -> Dict[str, Any]:
    """
    Save a raw message text into hybrid memory (vector + window).
    Returns status dict.
    """
    logger.info("[Tool] save_memory_tool called user=%s chat=%s message_len=%s", user_id, chat_id, len(message) if message else 0)
    try:
        asyncio.run(HybridMemory.store_message(user_id, chat_id, "system", message))
        logger.debug("[Tool] save_memory_tool completed")
        return {"status": "saved"}
    except Exception as e:
        logger.exception("[Tool] save_memory_tool failed: %s", e)
        return {"status": "error", "error": str(e)}


# --------------------------------------------------------------
# SEARCH AGENT
# --------------------------------------------------------------
search_prompt = """
You are SearchAgent.
Use your tools (arxiv_tool, load_url_tool, load_pdf_tool) to gather research information.
Return a JSON-like structure:
{
  "papers": [
    {"title": "", "summary": "", "url": ""}
  ]
}
Be factual and concise.
"""
logger.info("[Agent Init] Creating SearchAgent")


search_agent = create_deep_agent(
    model=llm,
    tools=[arxiv_tool, load_url_tool, load_pdf_tool],
    system_prompt=search_prompt,
)
logger.info("[Agent Init] SearchAgent created")


writer_prompt = """
You are WritingAgent.
Given structured search results, produce a draft of the research paper sections.
Return JSON:
{
  "sections": [
     {"name": "", "text": ""}
  ],
  "references": []
}
"""

def search_agent_tool(query: str) -> Any:
    """
    Tool wrapper that calls the SearchAgent and returns its output.
    Input: query string
    Output: search agent result (structured)
    """
    logger.info("[WriterTool] search_agent_tool called with query=%s", query)
    try:
        resp = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
        logger.debug("[WriterTool] search_agent_tool raw resp=%s", str(resp)[:500])
        # DeepAgents typically returns dict-like result; handle both dict and object
        if isinstance(resp, dict) and "output" in resp:
            logger.debug("[WriterTool] extracted output from resp")
            return resp["output"]
        return resp
    except Exception as e:
        logger.exception("[WriterTool] search_agent_tool invocation failed: %s", e)
        return {"error": str(e)}

logger.info("[Agent Init] Creating WriterAgent")
writer_agent = create_deep_agent(
    model=llm,
    tools=[search_agent_tool],
    system_prompt=writer_prompt,
)
logger.info("[Agent Init] WriterAgent created")


# --------------------------------------------------------------
# ROOT SUPERVISOR AGENT
# --------------------------------------------------------------
root_prompt = """
You are ResearchSupervisor.

You have access to:
1) SearchAgent (via search_agent_tool_root)
2) WriterAgent (via writer_agent_tool_root)
3) Vector memory + summaries

Pipeline:
1. Recall relevant memories.
2. Use SearchAgent to gather sources.
3. Use WriterAgent to create the draft.
Return a single final JSON containing draft sections and references.
"""

def search_agent_tool_root(query: str) -> Any:
    """
    Root-level tool to call the SearchAgent.
    Input: query string
    Output: search results (structured)
    """
    logger.info("[RootTool] search_agent_tool_root called query=%s", query)
    try:
        resp = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
        logger.debug("[RootTool] raw resp=%s", str(resp)[:500])
        return resp.get("output", resp) if isinstance(resp, dict) else resp
    except Exception as e:
        logger.exception("[RootTool] search_agent_tool_root failed: %s", e)
        return {"error": str(e)}

def writer_agent_tool_root(data: dict) -> Any:
    """
    Root-level tool to call the WriterAgent with structured data.
    Input: data dict
    Output: draft JSON (structured)
    """
    logger.info("[RootTool] writer_agent_tool_root called data_keys=%s", list(data.keys()) if isinstance(data, dict) else "non-dict")
    try:
        resp = writer_agent.invoke({
            "messages": [
                {"role": "user", "content": f"Write draft using:\n{data}"}
            ]
        })
        logger.debug("[RootTool] raw resp=%s", str(resp)[:500])
        return resp.get("output", resp) if isinstance(resp, dict) else resp
    except Exception as e:
        logger.exception("[RootTool] writer_agent_tool_root failed: %s", e)
        return {"error": str(e)}

logger.info("[Agent Init] Creating ResearchSupervisor agent")
research_agent = create_deep_agent(
    model=llm,
    tools=[search_agent_tool_root, writer_agent_tool_root, save_memory_tool],
    system_prompt=root_prompt,
)
logger.info("[Agent Init] ResearchSupervisor agent created")


# --------------------------------------------------------------
# ORCHESTRATOR (CALLED BY FASTAPI)
# --------------------------------------------------------------
class ResearchMultiAgent:
    """
    High-level orchestrator used by the FastAPI service.
    """
    async def run(self, query: str, user_id: str, chat_id: str, stream: bool = False):

        logger.info("[Run] run called user=%s chat=%s stream=%s query_len=%s", user_id, chat_id, stream, len(query) if query else 0)

        # -----------------------------------------
        # Utility to sanitize DeepAgents output
        # -----------------------------------------
        from langchain_core.messages import BaseMessage

        def _serialize(obj):
            """Convert DeepAgents / LangChain objects into JSON-safe values."""
            logger.debug("[Serialize] serializing type=%s", type(obj))
            if isinstance(obj, BaseMessage):
                try:
                    logger.debug("[Serialize] BaseMessage with type=%s", obj.type)
                    return {"role": obj.type, "content": obj.content}
                except Exception as e:
                    logger.exception("[Serialize] Failed to serialize BaseMessage: %s", e)
                    return str(obj)

            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    out[k] = _serialize(v)
                return out

            if isinstance(obj, list):
                return [_serialize(x) for x in obj]

            # Documents, Any other objects
            try:
                if hasattr(obj, "page_content"):
                    return obj.page_content
            except Exception as e:
                logger.debug("[Serialize] attribute page_content access failed: %s", e)

            return str(obj)

        # ----------------------------------------------------
        # 1) Save user message
        # ----------------------------------------------------
        logger.info("[Run] Saving user message to hybrid memory")
        await HybridMemory.store_message(user_id, chat_id, "user", query)

        # ----------------------------------------------------
        # 2) Relevant long-term memory recall
        # ----------------------------------------------------
        logger.info("[Run] Retrieving relevant memories")
        recall = await HybridMemory.retrieve_relevant(user_id, chat_id, query)
        logger.debug("[Run] recall raw=%s", str(recall)[:500])

        try:
            recall_text = "\n".join([getattr(d, "page_content", str(d)) for d in recall]) if recall else ""
        except Exception as e:
            logger.exception("[Run] Building recall_text failed: %s", e)
            recall_text = str(recall)

        logger.debug("[Run] recall_text_len=%s", len(recall_text))

        # ----------------------------------------------------
        # 3) Summaries (short term memory)
        # ----------------------------------------------------
        logger.info("[Run] Generating summary from short window")
        summary = await HybridMemory.summarize(user_id, chat_id)
        logger.debug("[Run] summary_len=%s", len(summary))

        # ----------------------------------------------------
        # 4) Build memory context
        # ----------------------------------------------------
        memory_context = (
            f"### MEMORY_RECALL:\n{recall_text}\n\n"
            f"### SUMMARY:\n{summary}\n"
        )
        logger.debug("[Run] memory_context_len=%s", len(memory_context))

        # ----------------------------------------------------
        # 5) Run Research Supervisor Agent (DeepAgents)
        # ----------------------------------------------------
        try:
            logger.info("[Run] Invoking research_agent with memory context and query")
            agent_res = research_agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"### MEMORY\n{memory_context}\n\n"
                            f"### QUERY\n{query}"
                        )
                    }
                ]
            })

            logger.debug("[Run] research_agent.invoke returned type=%s raw=%s", type(agent_res), str(agent_res)[:1000])
        except Exception as e:
            logger.exception("[Run] DeepAgent invocation failed, falling back to LLM only: %s", e)
            agent_res = {"output": f"Fallback LLM response:\n{query}"}

        # Extract output safely
        final_output_raw = (
            agent_res.get("output", agent_res)
            if isinstance(agent_res, dict)
            else agent_res
        )

        # Convert to JSON-safe structure
        logger.info("[Run] Serializing final output")
        try:
            final_output = _serialize(final_output_raw)
            logger.debug("[Run] final_output_serialized_preview=%s", str(final_output)[:1000])
        except Exception as e:
            logger.exception("[Run] Serialization of final output failed: %s", e)
            final_output = str(final_output_raw)

        # ----------------------------------------------------
        # 6) Save agent response into memory
        # ----------------------------------------------------
        logger.info("[Run] Saving agent response to hybrid memory")
        await HybridMemory.store_message(user_id, chat_id, "agent", str(final_output))

        # ----------------------------------------------------
        # 7) Store checkpoint into MongoDB (safe dict only)
        # ----------------------------------------------------
        try:
            logger.info("[Run] Persisting checkpoint to MongoDB")
            await MemoryEntry(
                user_id=user_id,
                chat_id=chat_id,
                key="research_result",
                value=final_output  # already JSON-safe
            ).insert()
            logger.debug("[Run] Checkpoint persisted successfully")
        except Exception as e:
            logger.exception("[Run] Failed to write checkpoint to MongoDB: %s", e)

        # ----------------------------------------------------
        # 8) Streaming mode
        # ----------------------------------------------------
        if stream:
            logger.info("[Run] Streaming enabled, preparing generator")
            text_out = str(final_output)

            async def gen():
                CHUNK = 400
                logger.debug("[Stream] streaming text length=%s", len(text_out))
                for i in range(0, len(text_out), CHUNK):
                    chunk = text_out[i:i+CHUNK]
                    logger.debug("[Stream] yielding chunk start=%s len=%s", i, len(chunk))
                    yield chunk
                    await asyncio.sleep(0.01)

            return gen()

        logger.info("[Run] Returning final_output (non-stream)")
        return final_output
