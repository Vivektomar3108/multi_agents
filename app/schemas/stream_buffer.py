from datetime import datetime
from typing import List, Dict

from beanie import Document, Indexed
from pydantic import Field


class StreamBuffer(Document):
    """
    Stores streamed chunks for an AI response to allow:
    - Resume on network disconnect (SSE Last-Event-ID)
    - Multiple users and multiple chat sessions
    - History replay if needed
    """

    user_id: Indexed(str) = Field(..., description="User owning this stream")
    chat_id: Indexed(str) = Field(..., description="Chat session ID")
    
    # Unique deterministic id based on (user_id, chat_id, query)
    stream_id: Indexed(str, unique=True) = Field(..., description="Unique stream identifier")

    # Stores each streamed chunk as {"id": int, "text": str}
    chunks: List[Dict] = Field(default_factory=list)

    # ⬇⬇ Updated — this line replaces the broken index config ⬇⬇
    created_at: Indexed(datetime, expireAfterSeconds=172800) = Field(default_factory=datetime.utcnow)

    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "stream_buffers"

    def append(self, text: str):
        """Utility method to append tokens safely."""
        next_id = len(self.chunks) + 1
        self.chunks.append({"id": next_id, "text": text})
        self.updated_at = datetime.utcnow()

    @staticmethod
    async def get_stream(user_id: str, chat_id: str, stream_id: str):
        """
        Convenience method to fetch existing buffer for a stream.
        """
        return await StreamBuffer.find_one(
            StreamBuffer.stream_id == stream_id,
            StreamBuffer.user_id == user_id,
            StreamBuffer.chat_id == chat_id,
        )
