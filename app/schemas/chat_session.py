# app/schemas/chat_session.py

from beanie import Document, Indexed
from pydantic import Field
from datetime import datetime
from typing import Optional
from enum import Enum


class ChatStatus(str, Enum):
    active = "active"
    archived = "archived"
    deleted = "deleted"


class ChatSession(Document):
    """
    Tracks user conversations and links uploaded files + research memory.

    Fields:
    - chat_id: UUID or generated string to uniquely identify a conversation thread
    - status: allows UI-level archiving without deleting DB history
    - title: auto-generated or user-updated conversation label
    """

    # Foreign user identifier
    user_id: Indexed(str) = Field(...)

    # Unique chat thread identifier
    chat_id: Indexed(str, unique=True) = Field(...)

    # Optional dynamic title ("Research on LLMs", "Resume Upload #1", etc.)
    title: Optional[str] = Field(default="New Chat")

    # Conversation lifecycle state
    status: ChatStatus = Field(default=ChatStatus.active)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chat_sessions"

    # --- Helpers ---

    def update_timestamp(self):
        """Call before saving to track last activity."""
        self.updated_at = datetime.utcnow()

    async def soft_delete(self):
        """Marks the chat as deleted without removing content."""
        self.status = ChatStatus.deleted
        self.update_timestamp()
        await self.save()

    async def archive(self):
        """Archives chat but keeps it readable."""
        self.status = ChatStatus.archived
        self.update_timestamp()
        await self.save()

    async def rename(self, title: str):
        """Rename chat session for UI clarity."""
        self.title = title
        self.update_timestamp()
        await self.save()
