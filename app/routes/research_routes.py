from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.middleware.auth import get_current_user
from app.schemas.user.user_model import User
from app.schemas.chat_session import ChatSession, ChatStatus
from app.controllers.research_agent_controller import ResearchController


router = APIRouter(
    prefix="/research",
    tags=["Research Agent"],
)

controller = ResearchController()


# -------------------------
# 📌 Request Body Model
# -------------------------
class ResearchRequest(BaseModel):
    chat_id: str
    query: str


# -------------------------
# 🧠 NON-STREAMING ENDPOINT
# -------------------------
@router.post(
    "/run",
    summary="Run research agent (non-streaming)",
    response_description="Returns completed research output."
)
async def run_research(
    payload: ResearchRequest,
    user: User = Depends(get_current_user)
):
    """
    Executes the research agent and returns the full response at once.
    """

    # Ensure session exists and belongs to authenticated user
    chat = await ChatSession.find_one(
        ChatSession.chat_id == payload.chat_id,
        ChatSession.user_id == str(user.user_id)
    )

    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")

    result = await controller.run_research(
        user_id=str(user.user_id),
        chat_id=payload.chat_id,
        query=payload.query
    )

    return {
        "success": True,
        "mode": "full_response",
        "chat_id": payload.chat_id,
        "result": result
    }


# -------------------------
# ⚡ STREAMING ENDPOINT (SSE)
# -------------------------
@router.post(
    "/stream",
    summary="Stream research agent response (Server-Sent Events)",
    response_description="Streams live agent output chunk-by-chunk."
)
async def stream_research(
    payload: ResearchRequest,
    user: User = Depends(get_current_user)
):
    """
    Streams chunks of agent response in real-time.
    Useful for large queries requiring progressive output.
    """

    chat = await ChatSession.find_one(
        ChatSession.chat_id == payload.chat_id,
        ChatSession.user_id == str(user.user_id)
    )

    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")

    async def event_generator():
        async for token in controller.stream_research(
            user_id=str(user.user_id),
            chat_id=payload.chat_id,
            query=payload.query
        ):
            yield f"data: {token}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
