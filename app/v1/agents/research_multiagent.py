import os
import json
import re
import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except Exception as e:
    raise ImportError(
        "Missing 'openai' package. Install it via `pip install openai` "
        "(OpenAI SDK compatible with OpenRouter)."
    ) from e

try:
    import deepagents
except Exception as e:
    raise ImportError("Missing 'deepagents' package. Install it (`pip install deepagents`).") from e

try:
    from app.agents.tools_v1 import arxiv_search, load_pdf_and_chunk, load_url_and_chunk
    from app.schemas.memory import MemoryEntry
    from app.config.chroma import get_vector_store
except Exception as e:
    # Fail fast with a clear message if local modules are missing
    raise ImportError("Missing local project modules (app.agents.tools_v1, app.schemas.memory, or app.config.chroma).") from e

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
logger.setLevel(logging.DEBUG)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
LLM_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.05"))

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set in environment. LLM calls may fail locally.")

llm_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": os.getenv("SITE_URL", "http://localhost"),
        "X-Title": os.getenv("SITE_NAME", "Research-Agent"),
    },
)

try:
    deepagents.DEFAULT_TOOLS = []

    deepagents.DEFAULT_MIDDLEWARE = []

    if hasattr(deepagents, "tool_builders"):
        deepagents.tool_builders.clear()

    if hasattr(deepagents, "TodoListMiddleware"):
        deepagents.TodoListMiddleware = lambda *a, **k: None

    if hasattr(deepagents, "FileMiddleware"):
        deepagents.FileMiddleware = lambda *a, **k: None

    if hasattr(deepagents, "ExecutionMiddleware"):
        deepagents.ExecutionMiddleware = lambda *a, **k: None

    if hasattr(deepagents, "TaskMiddleware"):
        deepagents.TaskMiddleware = lambda *a, **k: None

    print("DeepAgents default tools disabled.")
except Exception as e:
    print("Failed to disable DeepAgents defaults:", e)


def _call_llm_sync(prompt: str) -> str:
    """
    Call the OpenRouter/OpenAI-compatible client synchronously.
    Returns assistant text or empty string on failure.
    """
    logger.debug("[LLM] _call_llm_sync called. prompt_len=%d", len(prompt) if prompt else 0)
    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
        )
        choice = resp.choices[0]
        msg = choice.message
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
        return getattr(msg, "content", "") or str(msg)
    except Exception as e:
        logger.exception("[LLM] call failed: %s", e)
        return ""

class OpenRouterModelAdapter:


    def __init__(self, call_llm_fn, name: str = "openrouter-adapter"):
        self.name = name
        self.call_llm = call_llm_fn
        self.tools: List[Any] = []
        self.bind_kwargs: Dict[str, Any] = {}

    def bind_tools(self, tools: Iterable[Any] = None, *args, **kwargs):
        self.tools = list(tools) if tools else []
        self.bind_kwargs.update(kwargs or {})
        if args:
            self.bind_kwargs["_positional_args"] = list(args)

        logger.debug(
            "[Adapter:%s] bind_tools -> %s", 
            self.name,
            [t if isinstance(t, str) else getattr(t, "__name__", str(t)) for t in self.tools]
        )
        return self

    def _extract_prompt_from_payload(self, payload: Any) -> str:
        try:
            if isinstance(payload, dict) and "messages" in payload:
                lines = []
                for m in payload["messages"]:
                    if isinstance(m, dict):
                        lines.append(f"[{m.get('role','')}] {m.get('content','')}")
                    else:
                        lines.append(str(m))
                return "\n".join(lines)
            return str(payload)
        except Exception:
            return str(payload)

    def _find_json_object_in_text(self, text: str) -> Optional[dict]:
        if not text:
            return None

        for m in re.finditer(r"\{.*?\}", text, flags=re.S):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "name" in obj:
                    return obj
            except Exception:
                continue
        return None
    
    def _parse_openai_tool_call(self, text: str) -> Optional[dict]:
        """
        Detect patterns like:
        <|assistant|><|commentary to=tool_name code|>{"arg":1}<|call|>
        """
        if not text:
            return None

        m = re.search(
            r"to\s*=\s*([A-Za-z0-9_]+)\s*code\|>\s*(\{.*?\})",
            text,
            flags=re.S
        )
        if not m:
            return None

        tool = m.group(1)
        blob = m.group(2)
        try:
            args = json.loads(blob)
        except Exception:
            args = {}

        return {"name": tool, "arguments": args}

    def _resolve_and_call_tool(self, name: str, args: dict):
        """Return raw result from tool (will be stringified later)."""
        if not self.tools:
            logger.debug("[Adapter:%s] No tools bound.", self.name)
            return None

        for t in self.tools:
            try:
                # ("name", fn)
                if isinstance(t, (tuple, list)) and len(t) >= 2:
                    if t[0] == name:
                        return t[1](**args)

                # {"name":..., "func": ...}
                if isinstance(t, dict) and t.get("name") == name:
                    fn = t.get("func")
                    if callable(fn):
                        return fn(**args)

                # direct function
                if callable(t) and getattr(t, "__name__", None) == name:
                    return t(**args)

                # object with .name or .func
                if hasattr(t, "name") and t.name == name:
                    if callable(t):
                        return t(**args)
                    if hasattr(t, "func") and callable(t.func):
                        return t.func(**args)

            except Exception as e:
                logger.exception("[Adapter:%s] Tool '%s' failed: %s", self.name, name, e)
                return {"error": str(e)}

        logger.debug("[Adapter:%s] Tool '%s' not found.", self.name, name)
        return None

    # ----------------------------------------------------------
    # ALWAYS RETURN STRING TOOL OUTPUT
    # ----------------------------------------------------------
    def _stringify(self, x: Any) -> str:
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, indent=2, default=str)
        except Exception:
            return repr(x)

    def invoke(self, payload: Any):
        prompt = self._extract_prompt_from_payload(payload)
        raw = self.call_llm(prompt)
        text = (raw or "").strip()

        messages = [{"role": "assistant", "content": text}]

        parsed = self._parse_openai_tool_call(text)
        if parsed:
            tool_name = parsed["name"]
            tool_args = parsed.get("arguments", {}) or {}
            result = self._resolve_and_call_tool(tool_name, tool_args)
            messages.append({
                "role": "tool",
                "content": self._stringify(result)
            })

        else:
            payload_obj = self._find_json_object_in_text(text)
            if payload_obj and payload_obj.get("name"):
                tool_name = payload_obj["name"]
                tool_args = payload_obj.get("arguments", {}) or {}
                result = self._resolve_and_call_tool(tool_name, tool_args)
                messages.append({
                    "role": "tool",
                    "content": self._stringify(result)
                })

        return messages

    def generate(self, payload: Any):
        return self.invoke(payload)

    def __call__(self, payload: Any):
        return self.invoke(payload)


# --------------------------------------------------------------
# HYBRID MEMORY
# --------------------------------------------------------------
class HybridMemory:
    """
    Hybrid memory combining vector store (Chroma) + short window stored in MongoDB via MemoryEntry.
    Adapted from your previous implementation; kept async methods.
    """
    WINDOW_SIZE = 8

    @staticmethod
    async def store_message(user_id: str, chat_id: str, role: str, text: str) -> None:
        logger.info("[HybridMemory] store_message user=%s chat=%s role=%s text_len=%s", user_id, chat_id, role, len(text) if text else 0)
        try:
            vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
            vs.add_texts([f"{role.upper()}: {text}"])
            logger.debug("[HybridMemory] added text to vector store collection=%s", f"user_{user_id}_chat_{chat_id}")
        except Exception as e:
            logger.exception("[HybridMemory] Failed to add to vector store: %s", e)

        await HybridMemory._store_window(user_id, chat_id, role, text)

    @staticmethod
    async def retrieve_relevant(user_id: str, chat_id: str, query: str, k: int = 4):
        logger.info("[HybridMemory] retrieve_relevant user=%s chat=%s query_len=%s k=%s", user_id, chat_id, len(query) if query else 0, k)
        try:
            vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
            results = vs.similarity_search(query, k=k)
            logger.debug("[HybridMemory] similarity_search returned %s results", len(results) if results else 0)
            return results
        except Exception as e:
            logger.exception("[HybridMemory] Vector lookup failed: %s", e)
            return []

    @staticmethod
    async def _store_window(user_id: str, chat_id: str, role: str, text: str):
        key = "short_window_messages"
        logger.debug("[HybridMemory] _store_window user=%s chat=%s", user_id, chat_id)
        try:
            existing = await MemoryEntry.find_one(
                MemoryEntry.user_id == user_id,
                MemoryEntry.chat_id == chat_id,
                MemoryEntry.key == key,
            )

            if existing:
                arr = existing.value.get("messages", [])
                arr.append({"role": role, "text": text})
                arr = arr[-HybridMemory.WINDOW_SIZE:]
                existing.value = {"messages": arr}
                await existing.save()
                logger.debug("[HybridMemory] appended and saved short-window (len=%s)", len(arr))
            else:
                await MemoryEntry(
                    user_id=user_id,
                    chat_id=chat_id,
                    key=key,
                    value={"messages": [{"role": role, "text": text}]},
                ).insert()
                logger.debug("[HybridMemory] created new short-window record")
        except Exception as e:
            logger.exception("[HybridMemory] Failed to store window message: %s", e)

    @staticmethod
    async def get_window(user_id: str, chat_id: str):
        logger.debug("[HybridMemory] get_window user=%s chat=%s", user_id, chat_id)
        try:
            entry = await MemoryEntry.find_one(
                MemoryEntry.user_id == user_id,
                MemoryEntry.chat_id == chat_id,
                MemoryEntry.key == "short_window_messages",
            )
            if not entry:
                return []
            messages = entry.value.get("messages", [])
            return messages
        except Exception as e:
            logger.exception("[HybridMemory] Failed to load window messages: %s", e)
            return []

    @staticmethod
    async def summarize(user_id: str, chat_id: str) -> str:
        logger.info("[HybridMemory] summarize user=%s chat=%s", user_id, chat_id)
        try:
            window = await HybridMemory.get_window(user_id, chat_id)
            if not window:
                return ""
            text = "\n".join([f"{m['role'].upper()}: {m['text']}" for m in window])
            prompt = f"Summarize the following conversation in 6-10 concise lines with an academic tone:\n\n{text}"
            summary = _call_llm_sync(prompt) or ""
            # persist summary
            try:
                vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
                vs.add_texts([f"SUMMARY: {summary}"])
            except Exception as e:
                logger.exception("[HybridMemory] Failed to persist summary: %s", e)
            return summary
        except Exception as e:
            logger.exception("[HybridMemory] Summarization failed: %s", e)
            return ""

def arxiv_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv for a given query and return a list of metadata dicts.
    Args:
        query (str): search string
        max_results (int): number of results to fetch
    Returns:
        List[dict]: list of results with keys like title, summary, url
    """
    logger.info("[Tool] arxiv_tool called query=%s max_results=%s", query, max_results)
    try:
        result = asyncio.run(arxiv_search(query, max_results))
        return list(result or [])
    except Exception as e:
        logger.exception("[Tool] arxiv_tool failed: %s", e)
        return []


def load_url_tool(url: str) -> List[Any]:
    """
    Load a web page (via Playwright) and return chunked LangChain Documents as serializable dicts.
    Args:
        url (str): page URL
    Returns:
        list: list of document dicts or strings
    """
    logger.info("[Tool] load_url_tool called url=%s", url)
    try:
        res = asyncio.run(load_url_and_chunk(url))
        serialized = []
        for d in (res or []):
            try:
                if hasattr(d, "page_content"):
                    serialized.append({"page_content": d.page_content, **(getattr(d, "metadata", {}) or {})})
                else:
                    serialized.append(d)
            except Exception:
                serialized.append(str(d))
        return serialized
    except Exception as e:
        logger.exception("[Tool] load_url_tool failed: %s", e)
        return []


def load_pdf_tool(file_path: str) -> List[Any]:
    """
    Load and chunk a local PDF using UnstructuredPDFLoader and return Documents as serializable dicts.
    Args:
        file_path (str): local path to PDF
    Returns:
        list: list of document dicts or strings
    """
    logger.info("[Tool] load_pdf_tool called file_path=%s", file_path)
    try:
        res = asyncio.run(load_pdf_and_chunk(file_path))
        serialized = []
        for d in (res or []):
            try:
                if hasattr(d, "page_content"):
                    serialized.append({"page_content": d.page_content, **(getattr(d, "metadata", {}) or {})})
                else:
                    serialized.append(d)
            except Exception:
                serialized.append(str(d))
        return serialized
    except Exception as e:
        logger.exception("[Tool] load_pdf_tool failed: %s", e)
        return []


def save_memory_tool(user_id: str, chat_id: str, message: str) -> Dict[str, Any]:
    """
    Save a raw message text into hybrid memory (vector + window). Returns status dict.
    Args:
        user_id (str), chat_id (str), message (str)
    """
    logger.info("[Tool] save_memory_tool called user=%s chat=%s message_len=%s", user_id, chat_id, len(message) if message else 0)
    try:
        # call the async memory store safely
        asyncio.run(HybridMemory.store_message(user_id, chat_id, "system", message))
        return {"status": "saved"}
    except Exception as e:
        logger.exception("[Tool] save_memory_tool failed: %s", e)
        return {"status": "error", "error": str(e)}

try:
    deepagents.config.enable_default_tools = False  # type: ignore
    logger.debug("Disabled deepagents default tools via config.enable_default_tools = False")
except Exception:
    try:
        deepagents.DEFAULT_TOOLS = []  # fallback hacky approach
        logger.debug("Cleared deepagents.DEFAULT_TOOLS (fallback)")
    except Exception:
        logger.warning("Could not disable deepagents default tools programmatically. Ensure you pass explicit tools to create_deep_agent.")

# Create adapter instances (each agent gets its own adapter)
search_adapter = OpenRouterModelAdapter(_call_llm_sync)
search_adapter.bind_tools(tools=[("arxiv_tool", arxiv_tool), ("load_url_tool", load_url_tool), ("load_pdf_tool", load_pdf_tool)])

# create SearchAgent
search_system_prompt = """
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
search_agent = deepagents.create_deep_agent(
    model=search_adapter,
    tools=[arxiv_tool, load_url_tool, load_pdf_tool],  # explicit tools override defaults
    system_prompt=search_system_prompt,
    name="SearchAgent",
)


logger.info("[Agent Init] SearchAgent created")

writer_adapter = OpenRouterModelAdapter(_call_llm_sync)
writer_adapter.bind_tools(tools=[("search_agent_tool", lambda q: search_agent.invoke({"messages": [{"role": "user", "content": q}]}))])

def search_agent_tool_wrapper(query: str) -> Any:
    """
    Tool wrapper used by WriterAgent to call SearchAgent.
    Args:
        query (str): search query text
    Returns:
        Any: structured results from SearchAgent
    """
    logger.info("[WriterTool] calling search_agent with query=%s", query)
    try:
        resp = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return resp.get("output", resp) if isinstance(resp, dict) else resp
    except Exception as e:
        logger.exception("[WriterTool] search_agent invocation failed: %s", e)
        return {"error": str(e)}

writer_system_prompt = """
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
logger.info("[Agent Init] Creating WriterAgent")
writer_agent = deepagents.create_deep_agent(
    model=writer_adapter,
    tools=[search_agent_tool_wrapper],  # explicit
    system_prompt=writer_system_prompt,
    name="WriterAgent",
)
logger.info("[Agent Init] WriterAgent created")

root_adapter = OpenRouterModelAdapter(_call_llm_sync)
root_adapter.bind_tools(tools=[
    ("search_agent_tool_root", lambda q: search_agent.invoke({"messages": [{"role": "user", "content": q}]})),
    ("writer_agent_tool_root", lambda data: writer_agent.invoke({"messages": [{"role": "user", "content": f"Write draft using:\\n{data}"}]})),
    ("save_memory_tool", save_memory_tool),
])

def search_agent_tool_root(query: str) -> Any:
    """
    Root-level tool to call SearchAgent.
    """
    logger.info("[RootTool] search_agent_tool_root called query=%s", query)
    try:
        resp = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return resp.get("output", resp) if isinstance(resp, dict) else resp
    except Exception as e:
        logger.exception("[RootTool] search_agent_tool_root failed: %s", e)
        return {"error": str(e)}

def writer_agent_tool_root(data: dict) -> Any:
    """
    Root-level tool to call WriterAgent with structured data.
    """
    logger.info("[RootTool] writer_agent_tool_root called")
    try:
        resp = writer_agent.invoke({"messages": [{"role": "user", "content": f"Write draft using:\n{data}"}]})
        return resp.get("output", resp) if isinstance(resp, dict) else resp
    except Exception as e:
        logger.exception("[RootTool] writer_agent_tool_root failed: %s", e)
        return {"error": str(e)}

logger.info("[Agent Init] Creating ResearchSupervisor agent")
research_agent = deepagents.create_deep_agent(
    model=root_adapter,
    tools=[search_agent_tool_root, writer_agent_tool_root, save_memory_tool],  # explicit tools
    system_prompt="""
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
IMPORTANT:
You ONLY have access to the following tools:
- search_agent_tool_root
- writer_agent_tool_root
- save_memory_tool

DO NOT use any file, todo, task, or system-management tools such as:
ls, read_file, write_file, edit_file, glob, grep, write_todos, task, execute.
They DO NOT exist.

""",
    name="ResearchSupervisor",
)
logger.info("[Agent Init] ResearchSupervisor created")


# --------------------------------------------------------------
# ORCHESTRATOR CLASS (exposed to FastAPI)
# --------------------------------------------------------------
class ResearchMultiAgent:
    """
    High-level orchestrator for FastAPI endpoints to call.
    """

    async def run(self, query: str, user_id: str, chat_id: str, stream: bool = False):
        logger.info("[Run] run called user=%s chat=%s stream=%s query_len=%s", user_id, chat_id, stream, len(query) if query else 0)

        try:
            await HybridMemory.store_message(user_id, chat_id, "user", query)
        except Exception:
            logger.exception("[Run] Failed to save user message to HybridMemory")

        try:
            recall = await HybridMemory.retrieve_relevant(user_id, chat_id, query)
            recall_text = "\n".join([getattr(d, "page_content", str(d)) for d in recall]) if recall else ""
        except Exception as e:
            logger.exception("[Run] recall retrieval failed: %s", e)
            recall_text = ""

        try:
            summary = await HybridMemory.summarize(user_id, chat_id)
        except Exception:
            logger.exception("[Run] summarize failed")
            summary = ""

        memory_context = f"### MEMORY_RECALL:\n{recall_text}\n\n### SUMMARY:\n{summary}\n"
        logger.debug("[Run] memory_context_len=%s", len(memory_context))

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{memory_context}\n\nUser Query:\n{query}"
                }
            ]
        }


        try:
            agent_res = research_agent.invoke(payload)
            logger.debug("[Run] research_agent.invoke returned type=%s", type(agent_res))
        except Exception as e:
            logger.exception("[Run] DeepAgent invocation failed, falling back to LLM only: %s", e)
            fallback_prompt = f"{memory_context}\n\nUser query:\n{query}"
            fallback_output = _call_llm_sync(fallback_prompt)
            agent_res = [{"role": "assistant", "content": fallback_output}]

        final_output_raw = agent_res

        from langchain_core.messages import BaseMessage  # local import to avoid heavy global dep at module import

        def _serialize(obj):
            logger.debug("[Serialize] serializing type=%s", type(obj))
            if isinstance(obj, BaseMessage):
                try:
                    return {"role": obj.type, "content": obj.content}
                except Exception:
                    return str(obj)
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    out[k] = _serialize(v)
                return out
            if isinstance(obj, list):
                return [_serialize(x) for x in obj]
            try:
                if hasattr(obj, "page_content"):
                    return obj.page_content
            except Exception:
                pass
            return str(obj)

        try:
            final_output = _serialize(final_output_raw)
            logger.debug("[Run] final_output_serialized_preview=%s", str(final_output)[:1000])
        except Exception as e:
            logger.exception("[Run] Serialization failed: %s", e)
            final_output = str(final_output_raw)

        try:
            agent_text = final_output if isinstance(final_output, str) else repr(final_output)
            await HybridMemory.store_message(user_id, chat_id, "agent", agent_text)
        except Exception:
            logger.exception("[Run] Failed to save agent response to HybridMemory")

        try:
            checkpoint_value = final_output if isinstance(final_output, dict) else {"text": final_output}
            await MemoryEntry(
                user_id=user_id,
                chat_id=chat_id,
                key="research_result",
                value=checkpoint_value,
            ).insert()
            logger.debug("[Run] Checkpoint persisted successfully")
        except Exception:
            logger.exception("[Run] Failed to write checkpoint to MongoDB")

        if stream:
            logger.info("[Run] Streaming mode enabled")
            text_out = str(final_output)

            async def gen():
                CHUNK = 400
                for i in range(0, len(text_out), CHUNK):
                    yield text_out[i : i + CHUNK]
                    await asyncio.sleep(0.01)

            return gen()

        logger.info("[Run] Returning final output (non-stream)")
        return final_output

