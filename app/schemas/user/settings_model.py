# app/models/settings_model.py
from datetime import datetime
from typing import Dict, Optional
from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import Field


class UserSettings(Document):
    """Beanie model for user settings"""

    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")
    user_id: str = Field(..., description="Reference to User, required & unique")

    # User preferences
    privacy: Dict[str, bool] = Field(
        default_factory=lambda: {"profile_visible": True, "last_seen": True},
        description="Privacy preferences for profile and last seen"
    )
    integrations: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Third-party integration configurations"
    )
    workspace: Dict[str, str] = Field(
        default_factory=dict,
        description="Workspace-level settings"
    )
    experimental_features: bool = Field(
        default=False,
        description="Opt-in flag for experimental features"
    )
    account_type: str = Field(
        default="free",
        description="Account type: free, pro, enterprise"
    )

    # System timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # ──────────────────────────────
    # MongoDB Collection Settings
    # ──────────────────────────────
    class Settings:
        name = "user_settings"
        indexes = [
            # Example: unique account type for each user (optional)
            IndexModel([("user_id", ASCENDING)], name="unique_user_profile_idx", unique=True),
            IndexModel([("account_type", ASCENDING)], name="account_type_idx", unique=False)
        ]

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}
        arbitrary_types_allowed = True
