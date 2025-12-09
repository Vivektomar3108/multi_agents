import os
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any, Iterable, Callable
import concurrent.futures
import textwrap

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.schemas.memory import MemoryEntry
from app.config.chroma import get_vector_store

load_dotenv()

# ------------------------------------
# LOGGING
# ------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResearchMultiAgent")

# ------------------------------------
# CHATOPENAI -> GROQ
# ------------------------------------

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

# ------------------------------------
# UTIL
# ------------------------------------

def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except:
        if isinstance(s, str):
            for a, b in [("{", "}"), ("[", "]")]:
                start = s.find(a)
                end = s.rfind(b)
                if start != -1 and end != -1:
                    try:
                        return json.loads(s[start:end+1])
                    except:
                        pass
    return None


# ------------------------------------
# MEMORY SYSTEM (UNCHANGED SEMANTICS)
# ------------------------------------

class HybridMemory:

    WINDOW_SIZE = 8

    @staticmethod
    async def store_message(user_id, chat_id, role, content):
        try:
            vs = get_vector_store(collection=f"user_{user_id}_{chat_id}")
            vs.add_texts([f"{role.upper()}: {content}"])
        except:
            pass

        key = "short_window_messages"

        existing = await MemoryEntry.find_one({
            "user_id": user_id, "chat_id": chat_id, "key": key
        })

        if existing:
            messages = existing.value.get("messages", [])
            messages.append({"role": role, "text": content})
            existing.value = {"messages": messages[-HybridMemory.WINDOW_SIZE:]}
            await existing.save()
        else:
            await MemoryEntry(
                user_id=user_id,
                chat_id=chat_id,
                key=key,
                value={"messages": [{"role": role, "text": content}]}
            ).insert()

    @staticmethod
    async def retrieve_relevant(user_id, chat_id, query, k=4):
        vs = get_vector_store(collection=f"user_{user_id}_{chat_id}")
        return vs.similarity_search(query, k=k)

    @staticmethod
    async def summarize(user_id, chat_id):
        entry = await MemoryEntry.find_one({
            "user_id": user_id,
            "chat_id": chat_id,
            "key": "short_window_messages"
        })
        if not entry:
            return ""

        text = "\n".join([f"{m['role']}: {m['text']}" for m in entry.value["messages"]])
        return llm.invoke(f"Summarize this:\n{text}").content


# ------------------------------------
# MCP
# ------------------------------------

MCP = MultiServerMCPClient({
    "research": {
        "url": os.getenv("RESEARCH_MCP_URL", "http://localhost:8008/mcp"),
        "transport": "streamable_http"
    }
})


def _callable_from_tool_obj(t: Any) -> Optional[Callable]:
    for n in ["run", "call", "func", "__call__"]:
        if hasattr(t, n):
            return getattr(t, n)
    if callable(t):
        return t
    return None


def wrap_tools(mcp_tools):

    wrapped = []

    for t in mcp_tools:
        name = getattr(t, "name", None)
        desc = getattr(t, "description", "")

        fn = _callable_from_tool_obj(t)

        if not name or not fn:
            continue

        def make_wrapper(fn, tool_name):
            async def _async(*args, **kwargs):
                r = fn(*args, **kwargs)
                if asyncio.iscoroutine(r):
                    return await r
                return r

            def _sync(*args, **kwargs):

                try:
                    loop = asyncio.get_event_loop()
                except:
                    return asyncio.run(_async(*args, **kwargs))

                if loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor(1) as e:
                        return e.submit(lambda: asyncio.run(_async(*args, **kwargs))).result()
                else:
                    return loop.run_until_complete(_async(*args, **kwargs))

            return _sync

        wrapped.append(Tool.from_function(make_wrapper(fn, name), name=name, description=desc))

    return wrapped


# ------------------------------------
# AGENTS
# ------------------------------------

RESEARCH_PROMPT = """
You are ResearchAgent.
Use MCP tools.
Return JSON:
{ "data": [ { "title":"", "summary":"", "url":"" } ] }
"""

WRITER_PROMPT = """
You are WriterAgent.
Turn data into report.
Return:
{ "sections": [ { "title":"", "content":"" } ] }
"""

SUPERVISOR_PROMPT = """
You are the Deep Research Supervisor.
You CREATE PLANS and CONTROL EXECUTION.
"""


class SupervisorDeepAgent:

    def __init__(self, planner_agent, research_agent, writer_agent, tools):
        self.planner = planner_agent
        self.research_agent = research_agent
        self.writer_agent = writer_agent
        self.tools = tools

    async def plan(self, query):

        prompt = f"""
Create a JSON plan:

QUERY: {query}

FORMAT:
[
  {{"id":1, "agent":"ResearchAgent", "query":"..."}},
  {{"id":2, "agent":"WriterAgent", "query":"..."}}
]
"""

        r = await self.planner.ainvoke({
            "messages": [{"role":"user","content":prompt}]
        })

        plan = safe_json_loads(r["content"])
        return plan

    async def interrupt_for_human(self, plan):

        print("\n🧠 PLAN CREATED:")
        print(json.dumps(plan, indent=2))

        choice = input("\n✅ Approve (y) | ✏ Modify (m) | ❌ Cancel (n): ").strip().lower()

        if choice == "y":
            return plan

        if choice == "n":
            return None

        if choice == "m":
            new_plan = input("Enter new JSON plan:\n")
            return safe_json_loads(new_plan)

    async def run(self, query, user_id, chat_id):

        plan = await self.plan(query)

        approved_plan = await self.interrupt_for_human(plan)

        if not approved_plan:
            return {"status":"cancelled"}

        results = []

        for step in approved_plan:
            if step["agent"] == "ResearchAgent":
                res = await self.research_agent.ainvoke({"messages":[{"role":"user","content":step["query"]}]})

            elif step["agent"] == "WriterAgent":
                res = await self.writer_agent.ainvoke({"messages":[{"role":"user","content":step["query"]}]})

            else:
                continue

            results.append(res)

        synthesis_prompt = f"""
User query: {query}

Data:
{json.dumps(results, indent=2)}

Return final report.
"""

        final = await self.planner.ainvoke({
            "messages":[{"role":"user","content":synthesis_prompt}]
        })

        await HybridMemory.store_message(user_id, chat_id, "agent", final["content"])

        return final["content"]


# ------------------------------------
# BUILD EVERYTHING
# ------------------------------------

async def build():

    tools = await MCP.get_tools()
    wrapped = wrap_tools(tools)

    research_agent = create_agent(
        model=llm,
        tools=wrapped,
        system_prompt=RESEARCH_PROMPT
    )

    writer_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=WRITER_PROMPT
    )

    planner_agent = create_deep_agent(
        model=llm,
        tools=wrapped,
        system_prompt=SUPERVISOR_PROMPT,
        debug=True
    )

    return SupervisorDeepAgent(
        planner_agent,
        research_agent,
        writer_agent,
        wrapped
    )


# ------------------------------------
# RUN
# ------------------------------------

async def main():

    sup = await build()

    user_id = "123"
    chat_id = "abc"

    query = "Find top 10 AI agent research papers (2023-2025)"

    result = await sup.run(query, user_id, chat_id)

    print("\n==================== FINAL ====================\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
