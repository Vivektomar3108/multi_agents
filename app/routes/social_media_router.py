"""
Social Media Agent Router
FastAPI routes for Instagram automation
"""
from fastapi import APIRouter, Query, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from datetime import datetime
from uuid import uuid4

from app.middleware.auth import get_current_user
from app.schemas.user.user_model import User
from app.schemas.chat_session import ChatSession, ChatStatus
from app.controllers.social_media_controller import social_media_controller


# =========================================================
# ROUTER SETUP
# =========================================================

social_media_router = APIRouter(
    prefix="/social-media",
    tags=["Social Media Agent"],
)


# =========================================================
# CREATE SESSION
# =========================================================

@social_media_router.post(
    "/create",
    summary="Create a new Instagram automation session",
    response_description="Returns a new session_id for the authenticated user."
)
async def create_session(
    title: str = "New Instagram Session",
    user: User = Depends(get_current_user),
):
    """
    Creates a new Instagram automation session.
    """
    session_id = str(uuid4())

    new_session = ChatSession(
        user_id=str(user.user_id),
        chat_id=session_id,
        title=title,
        status=ChatStatus.active,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    await new_session.insert()

    return {
        "status": "success",
        "session_id": session_id,
        "title": title,
        "message": "Instagram session created successfully"
    }


# =========================================================
# EXECUTE COMMAND (NON-STREAMING)
# =========================================================

@social_media_router.post(
    "/command",
    summary="Execute Instagram command",
    response_description="Returns the full response from the agent"
)
async def execute_command(
    session_id: str = Form(..., description="Session ID"),
    command: str = Form(..., description="Natural language command (e.g., 'get my account insights')"),
    user: User = Depends(get_current_user),
):
    """
    Execute an Instagram automation command using natural language.
    
    Examples:
    - "Login to Instagram with username 'myuser' and password 'mypass'"
    - "Get my account insights"
    - "Upload a photo from 'photo.jpg' with caption 'Hello!' and hashtags travel, nature"
    - "Analyze competitor @username"
    - "Suggest hashtags for a fitness post"
    """
    user_id = str(user.user_id)
    
    result = await social_media_controller.execute_command(
        user_id=user_id,
        session_id=session_id,
        command=command
    )
    
    return result


# =========================================================
# EXECUTE COMMAND (STREAMING)
# =========================================================

@social_media_router.post(
    "/command/stream",
    summary="Execute Instagram command with streaming response",
    response_description="Server-Sent Events stream"
)
async def stream_command(
    session_id: str = Form(..., description="Session ID"),
    command: str = Form(..., description="Natural language command"),
    user: User = Depends(get_current_user),
):
    """
    Execute an Instagram command with streaming response for real-time UI updates.
    
    Returns Server-Sent Events (SSE) stream.
    """
    user_id = str(user.user_id)
    
    return StreamingResponse(
        social_media_controller.stream_command(
            user_id=user_id,
            session_id=session_id,
            command=command
        ),
        media_type="text/event-stream"
    )


# =========================================================
# GET SESSIONS
# =========================================================

@social_media_router.get(
    "/sessions",
    summary="Get all Instagram sessions for user",
    response_description="List of user's Instagram sessions"
)
async def get_sessions(
    limit: int = Query(20, description="Number of sessions to return"),
    skip: int = Query(0, description="Number of sessions to skip"),
    user: User = Depends(get_current_user),
):
    """
    Retrieve all Instagram automation sessions for the authenticated user.
    """
    user_id = str(user.user_id)
    
    sessions = await ChatSession.find(
        ChatSession.user_id == user_id
    ).sort("-created_at").skip(skip).limit(limit).to_list()
    
    return {
        "status": "success",
        "count": len(sessions),
        "sessions": [
            {
                "session_id": s.chat_id,
                "title": s.title,
                "status": s.status,
                "created_at": s.created_at,
                "updated_at": s.updated_at
            }
            for s in sessions
        ]
    }


# =========================================================
# DELETE SESSION
# =========================================================

@social_media_router.delete(
    "/sessions/{session_id}",
    summary="Delete an Instagram session",
    response_description="Deletion confirmation"
)
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
):
    """
    Delete a specific Instagram automation session.
    """
    user_id = str(user.user_id)
    
    session = await ChatSession.find_one(
        ChatSession.chat_id == session_id,
        ChatSession.user_id == user_id
    )
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await session.delete()
    
    return {
        "status": "success",
        "message": f"Session {session_id} deleted successfully"
    }


# =========================================================
# QUICK ACTIONS (CONVENIENCE ENDPOINTS)
# =========================================================

@social_media_router.post(
    "/login",
    summary="Quick login to Instagram",
    response_description="Login result"
)
async def quick_login(
    session_id: str = Form(..., description="Session ID"),
    username: str = Form(..., description="Instagram username"),
    password: str = Form(..., description="Instagram password"),
    verification_code: str = Form(None, description="2FA code (if enabled)"),
    user: User = Depends(get_current_user),
):
    """
    Convenience endpoint for Instagram login.
    """
    user_id = str(user.user_id)
    
    command = f"Login to Instagram with username '{username}' and password '{password}'"
    if verification_code:
        command += f" and verification code '{verification_code}'"
    
    result = await social_media_controller.execute_command(
        user_id=user_id,
        session_id=session_id,
        command=command
    )
    
    return result


@social_media_router.get(
    "/insights",
    summary="Get Instagram account insights",
    response_description="Account analytics"
)
async def get_insights(
    session_id: str = Query(..., description="Session ID"),
    user: User = Depends(get_current_user),
):
    """
    Convenience endpoint to get Instagram account insights.
    """
    user_id = str(user.user_id)
    
    result = await social_media_controller.execute_command(
        user_id=user_id,
        session_id=session_id,
        command="Get my Instagram account insights including followers, engagement rate, and recent performance"
    )
    
    return result


@social_media_router.post(
    "/upload-photo",
    summary="Upload a photo to Instagram",
    response_description="Upload result"
)
async def upload_photo(
    session_id: str = Form(..., description="Session ID"),
    photo_path: str = Form(..., description="Path to photo file"),
    caption: str = Form("", description="Photo caption"),
    hashtags: str = Form("", description="Comma-separated hashtags (without #)"),
    user: User = Depends(get_current_user),
):
    """
    Convenience endpoint to upload a photo to Instagram.
    """
    user_id = str(user.user_id)
    
    command = f"Upload a photo from '{photo_path}'"
    if caption:
        command += f" with caption '{caption}'"
    if hashtags:
        hashtag_list = hashtags.replace(" ", "").split(",")
        command += f" and hashtags {', '.join(hashtag_list)}"
    
    result = await social_media_controller.execute_command(
        user_id=user_id,
        session_id=session_id,
        command=command
    )
    
    return result


@social_media_router.post(
    "/analyze-competitor",
    summary="Analyze a competitor's Instagram account",
    response_description="Competitor analysis"
)
async def analyze_competitor(
    session_id: str = Form(..., description="Session ID"),
    username: str = Form(..., description="Competitor's Instagram username"),
    user: User = Depends(get_current_user),
):
    """
    Convenience endpoint to analyze a competitor's Instagram account.
    """
    user_id = str(user.user_id)
    
    result = await social_media_controller.execute_command(
        user_id=user_id,
        session_id=session_id,
        command=f"Analyze competitor Instagram account @{username} and provide insights on their content strategy, engagement, and posting patterns"
    )
    
    return result
