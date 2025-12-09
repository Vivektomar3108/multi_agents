from fastapi import APIRouter, Query, Depends, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
from uuid import uuid4

from app.middleware.auth import get_current_user
from app.schemas.user.user_model import User
from app.schemas.chat_session import ChatSession, ChatStatus
from app.controllers.search_agent_controller import SearchController

# =========================================================
# ROUTER SETUP
# =========================================================

search_router = APIRouter(
    prefix="/search",
    tags=["Search Agent"],
)

controller = SearchController()

# =========================================================
# CREATE SEARCH SESSION
# =========================================================

@search_router.post(
    "/create",
    summary="Create a new search session",
    response_description="Returns a new chat_id for the authenticated user."
)
async def create_search_session(
    title: str = "New Search",
    user: User = Depends(get_current_user),
):
    """
    Creates a new search session.
    """
    chat_id = str(uuid4())

    new_chat = ChatSession(
        user_id=str(user.user_id),
        chat_id=chat_id,
        title=title,
        status=ChatStatus.active,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    await new_chat.insert()

    return {
        "success": True,
        "message": "Search session created successfully",
        "chat_id": chat_id,
        "title": title,
        "created_at": new_chat.created_at.isoformat()
    }


# =========================================================
# NON-STREAMING SEARCH ENDPOINT
# =========================================================

@search_router.post(
    "/chat",
    summary="Perform a web search and receive structured results",
    response_description="Returns search summary, URLs, and detailed results"
)
async def search_chat(
    query: str = Query(..., description="Search query"),
    chat_id: str = Query(..., description="Chat/Search session ID"),
    user: User = Depends(get_current_user)
):
    """
    Performs a web search using the SearchAgent.
    Returns structured JSON with:
    - summary: Brief explanation of search results
    - urls: List of relevant URLs found
    - results: Detailed search results from web
    """

    # Ensure chat session exists
    chat = await ChatSession.find_one(
        ChatSession.chat_id == chat_id,
        ChatSession.user_id == str(user.user_id)
    )

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Search session not found"
        )

    result = await controller.run_search(
        user_id=str(user.user_id),
        chat_id=chat_id,
        query=query
    )

    return result


# =========================================================
# STREAMING SEARCH ENDPOINT
# =========================================================

@search_router.post(
    "/stream",
    summary="Perform a web search with streaming response",
)
async def search_stream(
    query: str = Query(..., description="Search query"),
    chat_id: str = Query(..., description="Chat/Search session ID"),
    user: User = Depends(get_current_user)
):
    """
    Streaming version of the search endpoint.
    Returns Server-Sent Events (SSE) stream.
    """

    # Ensure chat session exists
    chat = await ChatSession.find_one(
        ChatSession.chat_id == chat_id,
        ChatSession.user_id == str(user.user_id)
    )

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Search session not found"
        )

    return StreamingResponse(
        controller.stream_research(
            user_id=str(user.user_id),
            chat_id=chat_id,
            query=query
        ),
        media_type="text/event-stream"
    )
