# app/schemas/memory.py
from beanie import Document
from pydantic import Field
from typing import Any, Dict, Optional
from datetime import datetime


class MemoryEntry(Document):
    user_id: str = Field(...)
    chat_id: Optional[str] = Field(None)
    key: str = Field(...)
    value: Dict[str, Any] = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "memory_entries"
