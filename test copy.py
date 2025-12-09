import asyncio
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

async def main():
    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=2,
    )

    # MCP client — NO connect() required
    client = MultiServerMCPClient({
        "research_agent": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })

    # Fetch tools directly
    tools = await client.get_tools()
    print("Fetched tools:", [t.name for t in tools])
    llm_tools = llm.bind_tools(tools=tools)
    # Create agent
    agent = create_agent(
        model=llm_tools,
        tools=tools,
        system_prompt=(
            "You are a research assistant. Use tools when needed."
        ),
        debug=True
    )

    # Invoke agent
    response = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": "search research paper on arXiv about EMG "}
        ]
    })

    print("Agent response:", response)

if __name__ == "__main__":
    asyncio.run(main())
