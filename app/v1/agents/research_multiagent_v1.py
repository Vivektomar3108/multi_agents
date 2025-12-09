# app/agents/research_multiagent_v1.py
import os
import asyncio
import logging
from typing import Optional, List, Dict, Any, Iterable, Callable
import concurrent.futures

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_groq import ChatGroq
from langchain_core.tools import Tool  # LangChain tool wrapper
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import the DeepAgent planner that knows tools
# from deep_agent import DeepAgent as PlannerDeepAgent
import json
import textwrap

from app.schemas.memory import MemoryEntry
from app.config.chroma import get_vector_store

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
logger.setLevel(logging.DEBUG)


# ---------------------------
# LLM: ChatGroq (configured safely)
# ---------------------------
logger.info("Initializing ChatGroq LLM...")
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    temperature=float(os.getenv("GROQ_TEMPERATURE", "0.0")),
    max_retries=int(os.getenv("GROQ_MAX_RETRIES", "2")),
    # IMPORTANT: disable Groq's native function calling to avoid tool_use_failed errors
    model_kwargs={"tool_choice": "none"},
)
logger.info("ChatGroq initialized")


# ---------------------------
# Defensive synchronous LLM wrapper (for places expecting sync)
# ---------------------------
def _call_llm_sync(prompt: str) -> str:
    try:
        resp = llm.invoke([("system", ""), ("human", prompt)])
        # langchain-style return handling
        if isinstance(resp, dict):
            return str(resp.get("output") or resp.get("content") or resp)
        if hasattr(resp, "content"):
            return str(getattr(resp, "content"))
        return str(resp)
    except Exception as e:
        logger.exception("LLM.invoke failed: %s", e)
        try:
            gen = llm.generate([("system", ""), ("human", prompt)])
            if hasattr(gen, "generations"):
                return str(gen.generations)
            return str(gen)
        except Exception as e2:
            logger.exception("LLM.generate failed: %s", e2)
            return ""


####################
# Utilities
####################
def safe_json_loads(s: str) -> Optional[Any]:
    """
    Try to parse JSON from the model output robustly:
    - direct json.loads
    - try to find first {...} block if the model wrapped JSON in text
    """
    try:
        return json.loads(s)
    except Exception:
        if not isinstance(s, str):
            return None
        # naive attempt: find first JSON object/array block
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        # try array
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
    return None

####################
# Deep agent orchestrator (planner-aware)
####################
class PlannerDeepAgent:
    def __init__(self, model, agent, tools: List[Any] = None, debug: bool = False):
        """
        model: the LLM object (not strictly required now, kept for compatibility)
        agent: the tool-aware agent (must implement async .ainvoke or .invoke)
        tools: list of tool descriptors (as returned by client.get_tools())
        """
        self.model = model
        self.agent = agent
        self.debug = debug
        self.tools = tools or []
        self.tools_text = self._format_tools(self.tools)

    def _format_tools(self, tools: List[Any]) -> str:
        """
        Best-effort formatting of the tools list into a concise readable form.
        Each tool object may be different shapes depending on MCP client; handle common cases.
        """
        lines = []
        for t in tools:
            try:
                # try typical attributes / dict shapes
                name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) or str(t)
                desc = getattr(t, "description", None) or (t.get("description") if isinstance(t, dict) else None) or ""
                # try parameter hints
                params = ""
                if hasattr(t, "args") and getattr(t, "args", None):
                    params = f" args={t.args}"
                elif isinstance(t, dict) and t.get("args"):
                    params = f" args={t.get('args')}"
                lines.append(f"- {name}: {desc}{params}")
            except Exception:
                lines.append(f"- {t}")
        if not lines:
            return "No tools available."
        return "\n".join(lines)

    async def plan(self, user_query: str, max_steps: int = 6) -> List[Dict[str, Any]]:
        """
        Produce a plan. Handles:
        - direct JSON content returned in the AI message
        - planner that invoked tools (tool_calls) -> convert to steps
        - other nested shapes from langchain/langgraph
        """
        # include available tools in the prompt so the planner knows what it can call
        prompt = textwrap.dedent(f"""
        You are a planning assistant that outputs a JSON plan (array of steps) to accomplish the user's request.
        The user request: {user_query!r}

        Available tools the planner may use (name: brief description / args):
        {self.tools_text}

        Requirements for the plan JSON:
        - Return a JSON array of step objects.
        - Each step object should have:
          - id: integer (1..n)
          - name: short name for the step (no spaces)
          - action: one-word action hint (e.g., search, summarise, cite, fetch)
          - query: the input the step needs (string)
          - note: optional human note (string)
        - Keep steps to a maximum of {max_steps}.
        - Return mostly JSON and avoid long human prose around the JSON.
        """).strip()

        messages = [
            {"role": "system", "content": "You are a planner that must return JSON only."},
            {"role": "user", "content": prompt},
        ]

        planner_resp = await self._invoke_agent_safe({"messages": messages})

        # === 1) Try to extract plain content from common shapes ===
        planner_text = None

        # If it's a simple dict with 'content'
        if isinstance(planner_resp, dict) and "content" in planner_resp:
            planner_text = planner_resp["content"]

        # If it's a mapping with 'messages' list (langgraph/langchain style), look for AIMessage with 'content'
        if planner_text is None and isinstance(planner_resp, dict) and planner_resp.get("messages"):
            msgs = planner_resp.get("messages")
            # msgs may be list of message objects/dicts; find first with a non-empty 'content' string from AI
            for m in msgs[::-1]:  # prefer latest messages at end
                # m may be a dict-like or object; handle both
                if isinstance(m, dict) and m.get("content"):
                    planner_text = m.get("content")
                    break
                # handle objects with attributes
                try:
                    if hasattr(m, "content") and m.content:
                        planner_text = m.content
                        break
                except Exception:
                    pass

        # If still None, check top-level 'choices' or string form
        if planner_text is None and isinstance(planner_resp, dict) and planner_resp.get("choices"):
            try:
                c = planner_resp["choices"][0]
                if isinstance(c, dict) and c.get("message") and isinstance(c["message"], dict) and c["message"].get("content"):
                    planner_text = c["message"]["content"]
            except Exception:
                pass

        # fallback: convert to string
        if planner_text is None:
            planner_text = str(planner_resp)

        if self.debug:
            logger.debug("PLANNER RAW OUTPUT (extracted text):\n%s", planner_text)

        # === 2) Try parse JSON plan ===
        plan = safe_json_loads(planner_text)
        if isinstance(plan, list):
            return plan

        # === 3) If no JSON, try to detect tool_calls inside planner_resp and convert ===
        tool_calls = []
        if isinstance(planner_resp, dict):
            # direct top-level tool_calls
            tc = planner_resp.get("tool_calls") or []
            if tc:
                tool_calls = tc
            # also inspect messages for tool_calls in additional_kwargs
            for m in (planner_resp.get("messages", []) + planner_resp.get("choices", [])):
                try:
                    if isinstance(m, dict) and m.get("tool_calls"):
                        tool_calls.extend(m.get("tool_calls"))
                    elif hasattr(m, "additional_kwargs") and getattr(m, "additional_kwargs", None) and m.additional_kwargs.get("tool_calls"):
                        tool_calls.extend(m.additional_kwargs.get("tool_calls"))
                except Exception:
                    pass

        if tool_calls:
            steps = []
            for idx, tc in enumerate(tool_calls[:max_steps], start=1):
                name = ""
                args = {}
                if isinstance(tc, dict):
                    # support both top-level shape and nested 'function' shape
                    name = tc.get("name") or (tc.get("function") or {}).get("name") or ""
                    args = tc.get("args") or (tc.get("function") or {}).get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                # build step from tool call
                if isinstance(args, dict):
                    query = args.get("query") or args.get("text") or args.get("url") or json.dumps(args)
                else:
                    query = str(args)
                step = {
                    "id": idx,
                    "name": name or f"step_{idx}",
                    "action": (name.split("_")[0] if name else "tool"),
                    "query": query,
                    "note": f"Auto-converted from tool_call {name}"
                }
                steps.append(step)
            if self.debug:
                logger.debug("Converted tool_calls to plan: %s", steps)
            return steps

        # nothing usable found
        raise ValueError("Planner did not return a JSON list plan and had no tool_calls. Raw (extracted):\n" + planner_text)

    async def _invoke_agent_safe(self, payload: Any) -> Any:
        """
        Try to invoke inner agent in a few ways (async .ainvoke, .invoke, or sync callable).
        """
        try:
            if hasattr(self.agent, "ainvoke"):
                return await self.agent.ainvoke(payload)
        except Exception as e:
            logger.debug("[Planner invoke] async ainvoke failed: %s", e)

        try:
            if hasattr(self.agent, "invoke"):
                maybe = self.agent.invoke(payload)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                return maybe
            if callable(self.agent):
                maybe = self.agent(payload)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                return maybe
        except Exception as e:
            logger.debug("[Planner invoke] sync invoke/call failed: %s", e)

        # last resort
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, lambda: getattr(self.agent, "invoke", lambda p: self.agent(p))(payload))
        except Exception as e:
            logger.exception("[Planner invoke] executor fallback failed: %s", e)
            return {"output": f"Planner agent invocation failed: {e}"}

    async def run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one step by invoking the same agent (the agent can call MCP tools).
        Returns a dict with step metadata and result.
        """
        step_id = step.get("id")
        step_name = step.get("name")
        query = step.get("query", "")
        note = step.get("note", "")
        system_prompt = textwrap.dedent(f"""
        You are an expert research assistant that can call tools. Execute the following step (id={step_id}, name={step_name}).
        Step purpose: {note}
        Input/query: {query}

        When relevant, call the registered tools to fetch papers, web pages, or metadata.
        Return a JSON object with:
        - id
        - name
        - status ("success" or "fail")
        - result: short textual summary of the output (or empty string)
        - raw: any raw tool output (string or small JSON)
        - citations: optional list of citation objects (title/author/year/url)
        Keep your reply focused and machine-parseable.
        """).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            resp = await self._invoke_agent_safe({"messages": messages})
            if isinstance(resp, dict) and "content" in resp:
                content = resp["content"]
            else:
                content = str(resp)
            if self.debug:
                logger.debug("STEP %s RAW: %s", step_id, content)
            parsed = safe_json_loads(content)
            if isinstance(parsed, dict):
                return {**parsed, "raw_text": content}
            else:
                return {
                    "id": step_id,
                    "name": step_name,
                    "status": "success",
                    "result": content.strip(),
                    "raw": content,
                    "citations": []
                }
        except Exception as e:
            logger.exception("[Run step] exception: %s", e)
            return {
                "id": step_id,
                "name": step_name,
                "status": "fail",
                "result": "",
                "raw": str(e),
                "citations": []
            }

    async def synthesize(self, user_query: str, steps_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize final answer from step outputs via the agent.
        """
        prompt = textwrap.dedent(f"""
        You are a synthesis assistant. The user's original request: {user_query!r}

        You executed the following step results (JSON). Produce:
        1) A short concise final answer.
        2) A bullet list of key findings.
        3) A small list of citations (title, author, year, url) if present in step outputs.

        Step results JSON:
        {json.dumps(steps_results, indent=2)}
        """).strip()

        messages = [
            {"role": "system", "content": "You are a research summarizer. Produce concise, factual outputs."},
            {"role": "user", "content": prompt}
        ]

        resp = await self._invoke_agent_safe({"messages": messages})
        if isinstance(resp, dict) and "content" in resp:
            content = resp["content"]
        else:
            content = str(resp)

        parsed = safe_json_loads(content)
        if isinstance(parsed, dict):
            return parsed
        else:
            return {"final_answer": content}

    async def run(self, user_query: str, max_steps: int = 6) -> Dict[str, Any]:
        plan = await self.plan(user_query, max_steps=max_steps)
        if self.debug:
            logger.debug("PLAN: %s", json.dumps(plan, indent=2))
        step_results = []
        for step in plan:
            try:
                res = await self.run_step(step)
            except Exception as e:
                logger.exception("[Planner run] step failed: %s", e)
                res = {"id": step.get("id"), "name": step.get("name"), "status": "fail", "raw": str(e)}
            step_results.append(res)
        synthesis = await self.synthesize(user_query, step_results)
        return {"plan": plan, "steps": step_results, "synthesis": synthesis}


# ---------------------------
# HybridMemory (unchanged semantics, but safer)
# ---------------------------
class HybridMemory:
    WINDOW_SIZE = 8
    MAX_STORE_CHARS = 4000  # truncate long stored texts

    @staticmethod
    async def store_message(user_id: str, chat_id: str, role: str, text: str) -> None:
        logger.info(
            "[HybridMemory] store_message called user=%s chat=%s role=%s text_len=%s",
            user_id,
            chat_id,
            role,
            len(text) if text else 0,
        )
        safe_text = text
        try:
            if text and len(text) > HybridMemory.MAX_STORE_CHARS:
                safe_text = text[: HybridMemory.MAX_STORE_CHARS] + "...[truncated]"
            vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
            # If vector store expects list of texts, ensure safe_text is str
            vs.add_texts([f"{role.upper()}: {safe_text}"])
        except Exception as e:
            logger.exception("[HybridMemory] Failed to add to vector store: %s", e)
        await HybridMemory._store_window(user_id, chat_id, role, safe_text)

    @staticmethod
    async def retrieve_relevant(user_id: str, chat_id: str, query: str, k: int = 4):
        try:
            vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
            results = vs.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.exception("[HybridMemory] Vector lookup failed: %s", e)
            return []

    @staticmethod
    async def _store_window(user_id: str, chat_id: str, role: str, text: str):
        key = "short_window_messages"
        try:
            # Try field-query style; if it fails (ODM mismatch), fallback to naive scan
            try:
                existing = await MemoryEntry.find_one(
                    MemoryEntry.user_id == user_id,
                    MemoryEntry.chat_id == chat_id,
                    MemoryEntry.key == key,
                )
            except Exception:
                # Fallback: attempt to find by dict-style if find_one supports dicts
                try:
                    existing = await MemoryEntry.find_one({"user_id": user_id, "chat_id": chat_id, "key": key})
                except Exception:
                    existing = None

            if existing:
                arr = existing.value.get("messages", [])
                arr.append({"role": role, "text": text})
                arr = arr[-HybridMemory.WINDOW_SIZE :]
                existing.value = {"messages": arr}
                await existing.save()
            else:
                await MemoryEntry(
                    user_id=user_id,
                    chat_id=chat_id,
                    key=key,
                    value={"messages": [{"role": role, "text": text}]},
                ).insert()
        except Exception as e:
            logger.exception("[HybridMemory] Failed to store window: %s", e)

    @staticmethod
    async def get_window(user_id: str, chat_id: str) -> List[Dict[str, str]]:
        try:
            entry = None
            try:
                entry = await MemoryEntry.find_one(
                    MemoryEntry.user_id == user_id,
                    MemoryEntry.chat_id == chat_id,
                    MemoryEntry.key == "short_window_messages",
                )
            except Exception:
                try:
                    entry = await MemoryEntry.find_one({"user_id": user_id, "chat_id": chat_id, "key": "short_window_messages"})
                except Exception:
                    entry = None

            if not entry:
                return []
            return entry.value.get("messages", [])
        except Exception as e:
            logger.exception("[HybridMemory] Failed to get window: %s", e)
            return []

    @staticmethod
    async def summarize(user_id: str, chat_id: str) -> str:
        try:
            window = await HybridMemory.get_window(user_id, chat_id)
            if not window:
                return ""
            text = "\n".join([f"{m['role'].upper()}: {m['text']}" for m in window])
            prompt = (
                f"Summarize the following conversation in 6-10 concise lines with an academic tone:\n\n{text}"
            )
            summary = _call_llm_sync(prompt) or ""
            try:
                vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
                vs.add_texts([f"SUMMARY: {summary}"])
            except Exception as e:
                logger.exception("[HybridMemory] Persisting summary failed: %s", e)
            return summary
        except Exception as e:
            logger.exception("[HybridMemory] Summarize failed: %s", e)
            return ""


# ---------------------------
# MCP client + tool loading
# ---------------------------
MCP_SERVERS = {
    "research_tools": {
        "url": os.getenv("RESEARCH_MCP_URL", "http://localhost:8008/mcp"),
        "transport": "streamable_http",
    }
}

_mcp_tools: Optional[List[Any]] = None
_mcp_client: Optional[MultiServerMCPClient] = None
_agents_created = False

async def _init_mcp_tools() -> List[Any]:
    global _mcp_tools, _mcp_client
    if _mcp_tools is not None:
        return _mcp_tools

    logger.info("[MCP] Initializing MultiServerMCPClient with config: %s", MCP_SERVERS)
    client = MultiServerMCPClient(MCP_SERVERS)
    _mcp_client = client

    # call initialize() if present and async
    try:
        if hasattr(client, "initialize"):
            maybe = client.initialize()
            if asyncio.iscoroutine(maybe):
                await maybe
            else:
                # initialize may be sync and potentially blocking: run in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: maybe)
    except Exception:
        logger.debug("[MCP] client.initialize() failed or not required; proceeding", exc_info=True)

    # get_tools might be async or blocking sync. Call properly.
    tools = []
    try:
        get_tools_fn = getattr(client, "get_tools", None)
        if get_tools_fn is None:
            tools = []
        else:
            maybe = get_tools_fn()
            if asyncio.iscoroutine(maybe):
                tools = await maybe
            else:
                # get_tools returned a result immediately OR it's a heavy sync function that was already executed
                # To be robust, if maybe is a callable result (unlikely), handle that; otherwise treat maybe as result.
                # If get_tools is a blocking function (callable), prefer executing it in executor:
                if callable(get_tools_fn):
                    loop = asyncio.get_event_loop()
                    # run the potentially blocking call in executor
                    tools = await loop.run_in_executor(None, lambda: get_tools_fn())
                else:
                    tools = maybe
    except Exception as e:
        logger.exception("[MCP] get_tools failed: %s", e)
        tools = []

    _mcp_tools = tools or []
    logger.info("[MCP] Loaded %s tools", len(_mcp_tools))
    return _mcp_tools


# Kick off initialization on import (safe in running loops)
def _start_init_mcp_tools():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_init_mcp_tools())
        else:
            # Not running: use asyncio.run which creates a fresh loop
            asyncio.run(_init_mcp_tools())
    except RuntimeError:
        asyncio.run(_init_mcp_tools())

_start_init_mcp_tools()


# ---------------------------
# Helper: convert MCP tool objects into LangChain Tool wrappers
# ---------------------------
def _callable_from_tool_obj(t: Any) -> Optional[Callable]:
    """
    Try to find a callable on the MCP tool object.
    The client.get_tools() typically returns objects exposing either:
      - .run(...) or .call(...) or .func(...) or the object is itself callable.
    We try several attribute names and return a (possibly async) callable or None.
    """
    candidates = ["run", "call", "func", "__call__"]
    for name in candidates:
        if hasattr(t, name):
            attr = getattr(t, name)
            if callable(attr):
                return attr
    # if t is a raw function
    if callable(t):
        return t
    return None


def _make_tool_wrappers(tools: Iterable[Any]) -> List[Tool]:
    """
    Convert MCP-returned tool objects into LangChain Tool wrappers that call the MCP tool
    via the underlying callable. We keep wrappers synchronous where possible because
    deepagents/create_deep_agent expects regular Python callables.
    """
    wrapped: List[Tool] = []
    for t in tools or []:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
        desc = getattr(t, "description", None) or (t.get("description") if isinstance(t, dict) else "")
        callable_obj = _callable_from_tool_obj(t)

        if not name or not callable_obj:
            logger.debug("[Tool Wrap] Skipping tool (no name or no callable): %s", t)
            continue

        # Build a wrapper that supports both sync and async underlying callables.
        def make_wrapper(callable_obj, tool_name):
            async def _async_wrapper(*args, **kwargs):
                try:
                    res = callable_obj(*args, **kwargs)
                    if asyncio.iscoroutine(res):
                        return await res
                    return res
                except Exception as e:
                    logger.exception("[Tool:%s] underlying call failed: %s", tool_name, e)
                    raise

            def _sync_wrapper(*args, **kwargs):
                """
                Provide a synchronous wrapper for the (possibly async) underlying callable.
                If called from an async-running event loop in the same thread, execute the
                coroutine in a separate thread to avoid blocking the running loop.
                """
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # no event loop in this thread
                    return asyncio.run(_async_wrapper(*args, **kwargs))

                if loop.is_running():
                    # We're in an active event loop (same thread). Run the coroutine in a new thread
                    # so the current loop is not blocked.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(lambda: asyncio.run(_async_wrapper(*args, **kwargs)))
                        return fut.result()
                else:
                    # No running loop in this thread: safe to run directly
                    return loop.run_until_complete(_async_wrapper(*args, **kwargs))

            return _sync_wrapper

        sync_callable = make_wrapper(callable_obj, name)
        try:
            wrapped_tool = Tool.from_function(sync_callable, name=name, description=desc or "")
            wrapped.append(wrapped_tool)
            logger.debug("[Tool Wrap] Wrapped MCP tool: %s", name)
        except Exception:
            # Last-resort: create a simple Tool with a lambda that calls the callable via executor
            def fallback_fn(*args, __callable=callable_obj, **kwargs):
                loop = asyncio.get_event_loop()
                return loop.run_in_executor(None, lambda: __callable(*args, **kwargs))
            try:
                wt = Tool.from_function(fallback_fn, name=name, description=desc or "")
                wrapped.append(wt)
            except Exception:
                logger.exception("[Tool Wrap] Failed to wrap tool %s", name)

    return wrapped


# ---------------------------
# Build DeepAgents after MCP tools ready
# ---------------------------
search_agent = None
writer_agent = None
research_agent = None

_search_prompt = """
You are SearchAgent.
Use your tools (arxiv_search, load_url, load_pdf, pubmed_search, semantic_scholar_search) to gather research information.
Return JSON-like structure:
{
  "papers": [
    {"title": "", "summary": "", "url": ""}
  ]
}
Be factual and concise.
"""

_writer_prompt = """
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

_root_prompt = """
You are ResearchSupervisor.

You have access to:
1) SearchAgent (MCP-backed tools)
2) WriterAgent
3) Vector memory + summaries

Pipeline:
1. Recall relevant memories.
2. Use MCP tools to gather sources.
3. Use writing tools to create the draft.
Return a single final JSON containing draft sections and references.
"""


def _filter_tools_by_names_obj(tools: Iterable[Tool], keep: Iterable[str]) -> List[Tool]:
    keep_set = set(keep)
    return [t for t in tools if getattr(t, "name", None) in keep_set]


def _make_agents():
    """Create deepagents using the MCP tool wrappers."""
    global _mcp_tools, search_agent, writer_agent, research_agent, _agents_created

    if _agents_created:
        return

    tools_raw = _mcp_tools or []
    logger.info("[Agent Init] Creating tool wrappers from %s raw MCP tools", len(tools_raw))

    # Wrapped LangChain Tool objects for agent runtime use
    wrapped_tools = _make_tool_wrappers(tools_raw)
    logger.info("[Agent Init] Created %s wrapped tools", len(wrapped_tools))

    # Choose tools per-agent
    search_names = ("arxiv_search", "load_url", "load_pdf", "pubmed_search", "semantic_scholar_search", "duckduckgo_search", "crossref_search")
    writer_names = ("citation_format", "export_pdf", "outline_generator", "save_memory")

    search_tools = _filter_tools_by_names_obj(wrapped_tools, search_names)
    writer_tools = _filter_tools_by_names_obj(wrapped_tools, writer_names)

    # Create agents (guarded)
    try:
        logger.info("[Agent Init] Creating SearchAgent")
        search_agent = create_deep_agent(model=llm, tools=search_tools, system_prompt=_search_prompt)
        logger.info("[Agent Init] SearchAgent created")
    except Exception:
        logger.exception("[Agent Init] Failed creating SearchAgent")
        search_agent = create_deep_agent(model=llm, tools=[], system_prompt=_search_prompt)

    try:
        logger.info("[Agent Init] Creating WriterAgent")
        writer_agent = create_deep_agent(model=llm, tools=writer_tools, system_prompt=_writer_prompt)
        logger.info("[Agent Init] WriterAgent created")
    except Exception:
        logger.exception("[Agent Init] Failed creating WriterAgent")
        writer_agent = create_deep_agent(model=llm, tools=[], system_prompt=_writer_prompt)

    try:
        # supervisor: attach wrapped tools for runtime + inform planner of raw tools
        sup_tools = list({t.name: t for t in (search_tools + writer_tools)}.values())

        # Create an inner agent that can be invoked by the planner
        inner_research_agent = create_deep_agent(model=llm, tools=sup_tools, system_prompt=_root_prompt)

        # Wrap the inner agent with the planner-aware DeepAgent (so planner sees available tools)
        research_agent = PlannerDeepAgent(model=llm, agent=inner_research_agent, tools=tools_raw, debug=True)
        logger.info("[Agent Init] ResearchSupervisor (planner-aware) created")
    except Exception:
        logger.exception("[Agent Init] Failed creating ResearchSupervisor")
        # fallback: plain agent without planner tooling
        try:
            fallback_inner = create_deep_agent(model=llm, tools=[], system_prompt=_root_prompt)
            research_agent = PlannerDeepAgent(model=llm, agent=fallback_inner, tools=tools_raw, debug=True)
        except Exception:
            logger.exception("[Agent Init] Fallback ResearchSupervisor creation failed")
            research_agent = None

    _agents_created = True


# create agents once tools are loaded (non-blocking safe)
if _mcp_tools:
    _make_agents()
else:
    async def _wait_and_make():
        await _init_mcp_tools()
        _make_agents()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_wait_and_make())
        else:
            asyncio.run(_wait_and_make())
    except RuntimeError:
        asyncio.run(_wait_and_make())


# ---------------------------
# Orchestrator: ResearchMultiAgent
# ---------------------------
class ResearchMultiAgent:
    def __init__(self):
        self.search_agent = None
        self.writer_agent = None
        self.research_agent = None

    async def _maybe_async_invoke(self, agent_obj, payload: Any) -> Any:
        """
        Try async invocation first; fallback to sync/callable or executor-run sync.
        """
        try:
            if hasattr(agent_obj, "ainvoke"):
                return await agent_obj.ainvoke(payload)
        except Exception as e:
            logger.debug("[Agent Call] async invoke failed, will try sync: %s", e)

        # try sync invoke / callable
        try:
            if hasattr(agent_obj, "invoke"):
                maybe = agent_obj.invoke(payload)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                return maybe
            if callable(agent_obj):
                maybe = agent_obj(payload)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                return maybe
        except Exception as e:
            logger.exception("[Agent Call] sync invoke failed: %s", e)

        # last resort: run blocking invoke in executor
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, lambda: getattr(agent_obj, "invoke", lambda p: agent_obj(p))(payload))
        except Exception as e:
            logger.exception("[Agent Call] executor fallback failed: %s", e)
            return {"output": f"Agent invocation failed: {e}"}

    async def run(self, query: str, user_id: str, chat_id: str, stream: bool = False):
        logger.info(
            "[Run] run called user=%s chat=%s stream=%s query_len=%s",
            user_id,
            chat_id,
            stream,
            len(query) if query else 0,
        )

        from langchain_core.messages import BaseMessage

        def _serialize(obj):
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

        # save user message
        await HybridMemory.store_message(user_id, chat_id, "user", query)

        # long-term recall
        recall = await HybridMemory.retrieve_relevant(user_id, chat_id, query)
        try:
            recall_text = "\n".join([getattr(d, "page_content", str(d)) for d in recall]) if recall else ""
        except Exception:
            recall_text = str(recall)

        # short-term summary
        summary = await HybridMemory.summarize(user_id, chat_id)

        memory_context = (
            f"### MEMORY_RECALL:\n{recall_text}\n\n"
            f"### SUMMARY:\n{summary}\n"
        )

        # ensure agents exist
        if not _agents_created:
            await _init_mcp_tools()
            _make_agents()

        global research_agent
        if research_agent is None:
            logger.warning("[Run] research_agent not ready; fallback LLM-only")
            fallback = _call_llm_sync(f"Please answer briefly: {query}")
            await HybridMemory.store_message(user_id, chat_id, "agent", fallback)
            return fallback

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"### MEMORY\n{memory_context}\n\n"
                        f"### QUERY\n{query}"
                    ),
                }
            ]
        }

        try:
            # If research_agent is a PlannerDeepAgent instance, use its run() (planner orchestration)
            if isinstance(research_agent, PlannerDeepAgent):
                agent_res = await research_agent.run(query, max_steps=5)
            else:
                agent_res = await self._maybe_async_invoke(research_agent, payload)
        except Exception as e:
            logger.exception("[Run] DeepAgent invocation failed: %s", e)
            agent_res = {"output": f"Fallback LLM response:\n{_call_llm_sync(query)}"}

        # normalize response
        if isinstance(agent_res, dict):
            final_output_raw = agent_res.get("output", agent_res)
        else:
            final_output_raw = agent_res

        try:
            final_output = _serialize(final_output_raw)
        except Exception:
            final_output = str(final_output_raw)

        # persist results
        await HybridMemory.store_message(user_id, chat_id, "agent", str(final_output))
        try:
            await MemoryEntry(
                user_id=user_id, chat_id=chat_id, key="research_result", value=final_output
            ).insert()
        except Exception:
            logger.exception("[Run] Failed persisting checkpoint")

        # streaming support
        if stream:
            text_out = str(final_output)
            async def gen():
                CHUNK = 400
                for i in range(0, len(text_out), CHUNK):
                    chunk = text_out[i:i+CHUNK]
                    yield chunk
                    await asyncio.sleep(0.01)
            return gen()

        return final_output
