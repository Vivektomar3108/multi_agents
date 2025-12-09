# app/schemas/user/user_tokens.py
from beanie import Document, Indexed
from pydantic import Field
from typing import List, Dict, Optional
from datetime import datetime


class UserToken(Document):
    """Beanie ODM model for user tokens."""

    user_id: Indexed(str, description="User that owns this token") # type: ignore
    token_type: Indexed(str, description="Type of token: api / refresh / session / llm") # type: ignore
    token: Indexed(str, unique=True, description="Unique token string") # type: ignore

    scopes: List[str] = Field(default_factory=list, description="List of scopes granted to this token")
    audience: Optional[str] = Field(None, description="e.g., 'team:<id>' or 'organization'")
    metadata: Dict = Field(default_factory=dict, description="Custom metadata or extra info")

    issued_at: datetime = Field(default_factory=datetime.utcnow, description="Token issue timestamp")
    expires_at: Optional[datetime] = Field(None, description="Token expiration timestamp")

    # ──────────────────────────────
    # Meta Configuration
    # ──────────────────────────────
    class Settings:
        name = "user_tokens"  # MongoDB collection name
        indexes = [
            "user_id",
            "token",
            "token_type",
        ]

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    # ──────────────────────────────
    # Convenience methods
    # ──────────────────────────────
    @property
    def is_expired(self) -> bool:
        """Return True if the token is expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at

    @property
    def lifetime_seconds(self) -> Optional[int]:
        """Return token lifetime in seconds."""
        if not self.expires_at:
            return None
        return int((self.expires_at - self.issued_at).total_seconds())
