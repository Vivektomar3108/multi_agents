from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.middleware.auth import get_current_user
from app.schemas.user.user_model import User
from app.schemas.chat_session import ChatSession
from app.controllers.writer_agent_controller import WriterController


router = APIRouter(
    prefix="/writer",
    tags=["Writer Agent"],
)

controller = WriterController()


# -------------------------
# 📌 Request Body Model
# -------------------------
class WriterRequest(BaseModel):
    chat_id: str
    instruction: str


# -------------------------
# 🧠 NON-STREAMING ENDPOINT
# -------------------------
@router.post(
    "/run",
    summary="Run WriterAgent to generate document (Non-stream response)",
    response_description="Returns the full generated writing result."
)
async def run_writer(
    payload: WriterRequest,
    user: User = Depends(get_current_user)
):
    """
    Generates written content based on user instruction.
    """

    # Validate chat session belongs to the authenticated user
    chat = await ChatSession.find_one(
        ChatSession.chat_id == payload.chat_id,
        ChatSession.user_id == str(user.user_id)
    )

    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    result = await controller.write(
        user_id=str(user.user_id),
        chat_id=payload.chat_id,
        instruction=payload.instruction
    )

    return {
        "success": True,
        "mode": "writer_full_response",
        "chat_id": payload.chat_id,
        "instruction": payload.instruction,
        "result": result
    }


# -------------------------
# ⚡ STREAMING ENDPOINT (SSE)
# -------------------------
@router.post(
    "/stream",
    summary="Stream WriterAgent response (real-time writing)",
    response_description="Streams the generated text chunk-by-chunk."
)
async def stream_writer(
    payload: WriterRequest,
    user: User = Depends(get_current_user)
):
    """
    Streams the writing process in real time.
    Useful for long papers or iterative writing.
    """

    # Check chat ownership
    chat = await ChatSession.find_one(
        ChatSession.chat_id == payload.chat_id,
        ChatSession.user_id == str(user.user_id)
    )

    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    async def event_generator():
        async for chunk in controller.stream_write(
            user_id=str(user.user_id),
            chat_id=payload.chat_id,
            instruction=payload.instruction
        ):
            yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
