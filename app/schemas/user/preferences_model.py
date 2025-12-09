# app/models/preferences_model.py
from datetime import datetime
from typing import Dict, Literal, Optional
from beanie import Document, PydanticObjectId
from pydantic import Field


class UserPreferences(Document):
    """User preferences stored in MongoDB via Beanie."""

    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")

    language: str = Field(
        default="en",
        max_length=8,
        description="Preferred language code (e.g., en, es, fr)"
    )
    theme: Literal["light", "dark", "system"] = Field(
        default="system",
        description="Theme preference: light, dark, or system"
    )
    notifications: Dict[str, bool] = Field(
        default_factory=lambda: {"email": True, "sms": False, "push": True},
        description="Notification preferences per channel"
    )
    ai_assistant_preferences: Dict[str, str] = Field(
        default_factory=lambda: {"voice": "neutral", "style": "concise"},
        description="AI assistant personalization options"
    )
    experimental_features: bool = Field(
        default=False,
        description="Opt-in to experimental features"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record last update timestamp"
    )

    # ──────────────────────────────
    # Beanie collection settings
    # ──────────────────────────────
    class Settings:
        name = "user_preferences"  # MongoDB collection name
        # Add unique indexes if required
        indexes = [
            "language",
            "theme",
        ]

    class Config:
        validate_assignment = True
        json_encoders = {datetime: lambda v: v.isoformat()}
        arbitrary_types_allowed = True
        populate_by_name = True  # support alias "_id"
