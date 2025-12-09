from datetime import datetime
from typing import Optional, Literal, Dict, Any, List

from beanie import Document
from pydantic import BaseModel, Field


class EmbeddedMetadata(BaseModel):
    """Optional metadata like citations, links, tool outputs, intermediate reasoning."""
    agent: Optional[str] = None
    tool_name: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[List[str]] = Field(default_factory=list)
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatHistory(Document):
    """
    Stores scalable conversation history across multiple agents.
    Designed for:
    - autonomous agent orchestration
    - retrieval augmented generation
    - long-term memory summarization
    """

    user_id: str = Field(..., index=True)
    chat_id: str = Field(..., index=True)

    # message
    role: Literal["user", "assistant", "system", "tool", "supervisor", "agent"]
    content: str = Field(...)

    # agent or subsystem responsible for generating content
    source: Optional[str] = Field(default="unknown")  # ex: "SupervisorAgent", "ResearchAgent", "WriterAgent"

    # optional metadata
    metadata: EmbeddedMetadata = Field(default_factory=EmbeddedMetadata)

    # optional embedding for vector storage enabling retrieval
    embedding: Optional[List[float]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chat_history"
