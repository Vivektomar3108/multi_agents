"""
Social Media Agent Controller
Handles Instagram automation through LangChain agent
"""
import os
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.agents.social_media_agent import SocialMediaAgent


class SocialMediaController:
    """
    Controller for the Social Media Agent
    Handles Instagram automation, posting, analytics, and engagement
    """

    def __init__(self):
        load_dotenv()

        # Initialize LLM (same configuration as other agents)
        self.llm = ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            temperature=0.1,
            model_kwargs={"tool_choice": "auto"}
        )

        # Initialize Social Media Agent
        self.agent = SocialMediaAgent(self.llm)

    # =========================================================
    # NON-STREAMING RESPONSE
    # =========================================================
    
    async def execute_command(
        self,
        user_id: str,
        session_id: str,
        command: str
    ) -> dict:
        """
        Execute a social media command and return full response
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            command: Natural language command
            
        Returns:
            Dict with status and response
        """
        if not command:
            raise HTTPException(400, "Command cannot be empty")

        try:
            result = await self.agent.run(
                query=command,
                user_id=user_id,
                session_id=session_id
            )

            return {
                "status": "success",
                "session_id": session_id,
                "data": {
                    "response": result.get("response", ""),
                    "metadata": result.get("metadata", {}),
                    "success": result.get("success", True)
                }
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error executing command: {str(e)}"
            )

    # =========================================================
    # STREAMING RESPONSE
    # =========================================================
    
    async def stream_command(
        self,
        user_id: str,
        session_id: str,
        command: str
    ):
        """
        Stream social media command response for real-time UI updates
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            command: Natural language command
            
        Yields:
            Server-Sent Events formatted chunks
        """
        if not command:
            raise HTTPException(400, "Command cannot be empty")

        try:
            async for chunk in self.agent.stream(
                query=command,
                user_id=user_id,
                session_id=session_id
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"


# Global controller instance
social_media_controller = SocialMediaController()
