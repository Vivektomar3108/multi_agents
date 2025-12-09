import os
import asyncio
import json
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# =========================================================
# INIT
# =========================================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResearchAgent")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")


# =========================================================
# LLM (ALLOW tools, NOT FORCE)
# =========================================================
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0,
    model_kwargs={
        "tool_choice": "auto"    # <-- FIXED (was required)
    }
)


# =========================================================
# Allowed Tools
# =========================================================
ALLOWED_TOOLS = [
    "arxiv_search",
    "semantic_scholar_search",
    "pubmed_search",
    "duckduckgo_search",
    "load_url"
]


# =========================================================
# MCP CLIENT
# =========================================================
mcp = MultiServerMCPClient(
    {
        "research_tools": {
            "transport": "streamable_http",
            "url": "http://localhost:8008/mcp",
        }
    }
)


async def load_tools():
    logger.info("Loading tools...")
    tools = await mcp.get_tools()

    logger.info("Found MCP tools: %s", [t.name for t in tools])

    filtered = [t for t in tools if t.name in ALLOWED_TOOLS]

    logger.info("Allowed tools: %s", [t.name for t in filtered])

    return filtered


# =========================================================
# Dynamic Prompt (UPDATED)
# =========================================================
def build_prompt(tools):
    tool_text = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    return f"""
You are ResearchAgent.
You MAY use the following MCP tools when needed:

{tool_text}

Rules:
- Use a tool ONLY when necessary.
- After calling tools, YOU MUST produce final JSON.
- Do NOT ask the user questions.
- Do NOT call tools repeatedly.

FINAL OUTPUT FORMAT (JSON ONLY):
{{
  "papers": [
    {{"title": "", "year": "", "url": "", "summary": ""}}
  ]
}}
"""


# =========================================================
# MAIN
# =========================================================
async def main():
    tools = await load_tools()

    system_prompt = build_prompt(tools)

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    logger.info("Running Research Agent...")

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Find top 10 AI agent papers 2023–2025"
            }
        ]
    })

    print("\n======== RESULT ========\n")
    print(result)
    # print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
