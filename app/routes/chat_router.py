from fastapi import APIRouter, Query, Depends, Request, HTTPException
from datetime import datetime
from uuid import uuid4
from typing import List, Optional

from app.middleware.auth import get_current_user
from app.schemas.user.user_model import User
from app.schemas.chat_session import ChatSession, ChatStatus
from app.schemas.chat_history import ChatHistory
from app.controllers.chat_controller import AIChatController
from app.services.chat_history_service import ChatHistoryService

# =========================================================
# ROUTER SETUP
# =========================================================

chat_router = APIRouter(
    prefix="/chat",
    tags=["Chat Sessions"],
)

controller = AIChatController()

# =========================================================
# CREATE CHAT SESSION
# =========================================================

@chat_router.post(
    "/create",
    summary="Create a new chat session",
    response_description="Returns a new chat_id for the authenticated user."
)
async def create_chat(
    title: str = "New Chat",
    user: User = Depends(get_current_user),
):
    """
    Creates a new chat session.
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
        "message": "Chat session created successfully",
        "chat_id": chat_id,
        "title": title,
        "created_at": new_chat.created_at.isoformat()
    }


# =========================================================
# NON-STREAMING CHAT ENDPOINT
# =========================================================

@chat_router.post(
    "/ask",
    summary="Send a message and receive full Supervisor response (non-streaming)",
)
async def ask_chat(
    query: str = Query(..., description="User prompt"),
    chat_id: str = Query(..., description="Chat session ID"),
    user: User = Depends(get_current_user)
):
    """
    Normal NON-streaming endpoint.
    Calls SupervisorAgent -> tools -> writer_agent/research_agent and
    returns final Markdown in one response.
    """

    # Ensure chat session exists
    chat = await ChatSession.find_one(
        ChatSession.chat_id == chat_id,
        ChatSession.user_id == str(user.user_id)
    )

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    result = await controller.chat(
        user_id=str(user.user_id),
        chat_id=chat_id,
        query=query
    )

    return result


# =========================================================
# GET CHAT HISTORY BY CHAT_ID
# =========================================================

@chat_router.get(
    "/history/{chat_id}",
    summary="Get chat history for a specific chat session",
    response_description="Returns all messages in the chat session"
)
async def get_chat_history(
    chat_id: str,
    user: User = Depends(get_current_user)
):
    """
    Retrieves all chat history for a specific chat_id.
    Verifies that the chat belongs to the authenticated user.
    """
    
    # Verify chat session exists and belongs to user
    chat = await ChatSession.find_one(
        ChatSession.chat_id == chat_id,
        ChatSession.user_id == str(user.user_id)
    )
    
    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found or you don't have access to it"
        )
    
    # Get chat history
    history = await ChatHistoryService.get_chat_history(chat_id)
    
    return {
        "success": True,
        "chat_id": chat_id,
        "user_id": str(user.user_id),
        "title": chat.title,
        "message_count": len(history),
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "source": msg.source,
                "metadata": msg.metadata.dict() if msg.metadata else {},
                "created_at": msg.created_at.isoformat()
            }
            for msg in history
        ]
    }


# =========================================================
# GET PAGINATED CHAT HISTORY
# =========================================================

@chat_router.get(
    "/history/{chat_id}/paginated",
    summary="Get paginated chat history for a specific chat session",
    response_description="Returns paginated messages in the chat session"
)
async def get_paginated_chat_history(
    chat_id: str,
    skip: int = Query(0, ge=0, description="Number of messages to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),
    user: User = Depends(get_current_user)
):
    """
    Retrieves paginated chat history for a specific chat_id.
    Verifies that the chat belongs to the authenticated user.
    """
    
    # Verify chat session exists and belongs to user
    chat = await ChatSession.find_one(
        ChatSession.chat_id == chat_id,
        ChatSession.user_id == str(user.user_id)
    )
    
    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found or you don't have access to it"
        )
    
    # Get paginated history
    history = await ChatHistoryService.get_paginated(chat_id, skip=skip, limit=limit)
    
    # Get total count
    total_count = await ChatHistory.find(ChatHistory.chat_id == chat_id).count()
    
    return {
        "success": True,
        "chat_id": chat_id,
        "user_id": str(user.user_id),
        "title": chat.title,
        "pagination": {
            "skip": skip,
            "limit": limit,
            "total": total_count,
            "returned": len(history)
        },
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "source": msg.source,
                "metadata": msg.metadata.dict() if msg.metadata else {},
                "created_at": msg.created_at.isoformat()
            }
            for msg in history
        ]
    }


# =========================================================
# GET LAST N MESSAGES FROM CHAT
# =========================================================

@chat_router.get(
    "/history/{chat_id}/recent",
    summary="Get recent messages from a chat session",
    response_description="Returns the most recent N messages"
)
async def get_recent_chat_history(
    chat_id: str,
    limit: int = Query(20, ge=1, le=100, description="Number of recent messages to return"),
    user: User = Depends(get_current_user)
):
    """
    Retrieves the most recent N messages for a specific chat_id.
    Verifies that the chat belongs to the authenticated user.
    Messages are returned in reverse chronological order (newest first).
    """
    
    # Verify chat session exists and belongs to user
    chat = await ChatSession.find_one(
        ChatSession.chat_id == chat_id,
        ChatSession.user_id == str(user.user_id)
    )
    
    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found or you don't have access to it"
        )
    
    # Get recent messages
    history = await ChatHistoryService.get_last_messages(chat_id, limit=limit)
    
    return {
        "success": True,
        "chat_id": chat_id,
        "user_id": str(user.user_id),
        "title": chat.title,
        "limit": limit,
        "message_count": len(history),
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "source": msg.source,
                "metadata": msg.metadata.dict() if msg.metadata else {},
                "created_at": msg.created_at.isoformat()
            }
            for msg in history
        ]
    }


# =========================================================
# GET ALL CHAT SESSIONS FOR USER
# =========================================================

@chat_router.get(
    "/sessions",
    summary="Get all chat sessions for the authenticated user",
    response_description="Returns all chat sessions"
)
async def get_user_chat_sessions(
    status: Optional[str] = Query(None, description="Filter by status: active, archived, or deleted"),
    user: User = Depends(get_current_user)
):
    """
    Retrieves all chat sessions for the authenticated user.
    Can optionally filter by status.
    """
    
    query = {"user_id": str(user.user_id)}
    
    if status:
        try:
            query["status"] = ChatStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join([s.value for s in ChatStatus])}"
            )
    
    sessions = await ChatSession.find(query).sort("-updated_at").to_list()
    
    return {
        "success": True,
        "user_id": str(user.user_id),
        "session_count": len(sessions),
        "sessions": [
            {
                "chat_id": session.chat_id,
                "title": session.title,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }
            for session in sessions
        ]
    }


# =========================================================
# GET FILTERED MESSAGES BY ROLE OR SOURCE
# =========================================================

@chat_router.get(
    "/history/{chat_id}/filter",
    summary="Get filtered chat messages by role or source",
    response_description="Returns filtered messages"
)
async def get_filtered_chat_history(
    chat_id: str,
    role: Optional[str] = Query(None, description="Filter by role: user, assistant, system, tool, supervisor, agent"),
    source: Optional[str] = Query(None, description="Filter by source: e.g., SupervisorAgent, ResearchAgent, WriterAgent"),
    user: User = Depends(get_current_user)
):
    """
    Retrieves filtered chat history for a specific chat_id.
    Can filter by role and/or source.
    Verifies that the chat belongs to the authenticated user.
    """
    
    # Verify chat session exists and belongs to user
    chat = await ChatSession.find_one(
        ChatSession.chat_id == chat_id,
        ChatSession.user_id == str(user.user_id)
    )
    
    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found or you don't have access to it"
        )
    
    # Get filtered messages
    history = await ChatHistoryService.filter_messages(chat_id, role=role, source=source)
    
    return {
        "success": True,
        "chat_id": chat_id,
        "user_id": str(user.user_id),
        "title": chat.title,
        "filters": {
            "role": role,
            "source": source
        },
        "message_count": len(history),
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "source": msg.source,
                "metadata": msg.metadata.dict() if msg.metadata else {},
                "created_at": msg.created_at.isoformat()
            }
            for msg in history
        ]
    }
