# app/services/research_agent_service.py

import asyncio
from typing import Optional

from app.agents.research_multiagent_v1 import ResearchMultiAgent

research = ResearchMultiAgent()
async def process_normal(user_id: str, chat_id: str, query: str, file_url: Optional[str]):
    """
    Runs the Groq Supervisor once and returns the final structured result.
    """
    result = await research.run(
        query=query,
        user_id=user_id,
        chat_id=chat_id,
    )

    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "query": query,
        "result": result  # contains final JSON from Groq agent
    }


def process_stream(user_id: str, chat_id: str, query: str, file_url: Optional[str]):
    """
    Returns an async generator for streaming the agent's output.
    """
    # run_research_pipeline returns a dict (not a generator),
    # so for true streaming you would extend the Groq client to use the stream API.
    # For now we wrap the call into a generator manually.
    async def generator():
        res = await research.run(
            query=query,
            user_id=user_id,
            chat_id=chat_id,
            stream=False
        )
        text = str(res)
        chunk = 400
        for i in range(0, len(text), chunk):
            yield text[i:i+chunk]
            await asyncio.sleep(0.01)

    return generator()
