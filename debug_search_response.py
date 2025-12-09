import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.agents.research_agent.agents.search_agent_test import SearchAgent
from app.config.mongo import init_db, close_db

load_dotenv()
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ Missing GROQ_API_KEY in environment variables!")

llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    temperature=0.3,
    model_kwargs={"tool_choice": "auto"}
)

async def debug_search():
    await init_db()
    
    agent = SearchAgent(llm)
    await agent.initialize()
    
    # Make a search query
    query = "What is MS Dhoni's date of birth?"
    
    # Call the agent directly to see the raw response
    print("\n=== CALLING AGENT ===")
    response = await agent.agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    
    print("\n=== RAW RESPONSE TYPE ===")
    print(f"Type: {type(response)}")
    
    print("\n=== RESPONSE DICT KEYS ===")
    if isinstance(response, dict):
        print(f"Keys: {response.keys()}")
    
    print("\n=== MESSAGES ===")
    if isinstance(response, dict) and "messages" in response:
        for i, msg in enumerate(response["messages"]):
            print(f"\n--- Message {i} ---")
            print(f"Type: {type(msg)}")
            
            if hasattr(msg, '__dict__'):
                print(f"Attributes: {msg.__dict__}")
            
            if hasattr(msg, 'tool_calls'):
                print(f"Tool calls: {msg.tool_calls}")
            
            if hasattr(msg, 'type'):
                print(f"Message type: {msg.type}")
            
            if hasattr(msg, 'content'):
                print(f"Content preview: {str(msg.content)[:500]}")
    
    print("\n=== FORMATTED RESULT ===")
    result = await agent.run("test-user", "test-chat", query)
    print(json.dumps(result, indent=2))
    
    await close_db()

if __name__ == "__main__":
    asyncio.run(debug_search())
