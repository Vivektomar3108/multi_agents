"""
Debug script to test SearchAgent and inspect response structure
Run this to see the actual format of responses from the search agent
"""
import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.config.mongo import init_db, close_db
from app.agents.research_agent.agents.search_agent_test import SearchAgent

# Setup logging to see all debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

async def test_search_agent():
    """Test the search agent and print the full response structure"""
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError("❌ Missing GROQ_API_KEY in environment variables!")

    # Create LLM
    llm = ChatOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0.3,
        model_kwargs={"tool_choice": "auto"}
    )
    
    try:
        await init_db()
        
        # Initialize agent
        logger.info("Initializing SearchAgent...")
        agent = SearchAgent(llm)
        await agent.initialize()
        
        # Test query
        query = "Who is MS Dhoni?"
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing query: {query}")
        logger.info(f"{'='*60}\n")
        
        # Run the agent
        result = await agent.run("test-user", "test-chat", query)
        
        # Print the full result
        logger.info(f"\n{'='*60}")
        logger.info("RESULT STRUCTURE:")
        logger.info(f"{'='*60}")
        logger.info(json.dumps(result, indent=2, default=str))
        
        # Print specific fields
        logger.info(f"\n{'='*60}")
        logger.info("EXTRACTED DATA:")
        logger.info(f"{'='*60}")
        logger.info(f"Response: {result.get('response', 'N/A')}")
        logger.info(f"URLs: {result.get('urls', [])}")
        logger.info(f"Results count: {len(result.get('results', []))}")
        logger.info(f"Results: {json.dumps(result.get('results', []), indent=2, default=str)}")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)
    finally:
        await close_db()

if __name__ == "__main__":
    asyncio.run(test_search_agent())
