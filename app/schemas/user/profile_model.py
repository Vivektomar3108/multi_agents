# app/models/profile_model.py
from typing import Optional, List, Dict
from datetime import datetime
from pydantic import Field, HttpUrl
from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING


class UserProfile(Document):
    """Beanie Document for user profiles"""

    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")
    user_id: str = Field(..., description="Reference to User, required & unique")

    first_name: Optional[str] = Field(None, max_length=80, description="User first name")
    last_name: Optional[str] = Field(None, max_length=80, description="User last name")
    display_name: Optional[str] = Field(None, max_length=120, description="Public display name")
    avatar_url: Optional[HttpUrl] = Field(None, description="Profile avatar URL")
    bio: Optional[str] = Field(None, max_length=2000, description="User biography/about section")
    location: Optional[str] = Field(None, max_length=120, description="Location string")
    timezone: str = Field("UTC", max_length=64, description="Preferred timezone")

    # AI-platform related
    skills: List[str] = Field(default_factory=list, description="User skill tags")
    social_links: Dict[str, str] = Field(default_factory=dict, description="Map of social platform -> URL")
    status_message: Optional[str] = Field(None, max_length=140, description="Short status or tagline")

    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Profile update timestamp")

    # ──────────────────────────────
    # Beanie / MongoDB Settings
    # ──────────────────────────────
    class Settings:
        name = "user_profiles"
        indexes = [
            IndexModel([("user_id", ASCENDING)], name="unique_user_profile_idx", unique=True)
        ]

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    # ──────────────────────────────
    # Helper Methods
    # ──────────────────────────────
    async def save_profile(self):
        """Save or update the profile in the database"""
        self.updated_at = datetime.utcnow()
        await self.save()
        return self

    @staticmethod
    async def get_by_user_id(user_id: str) -> Optional["UserProfile"]:
        """Fetch profile by user_id"""
        return await UserProfile.find_one(UserProfile.user_id == user_id)

    @staticmethod
    async def delete_by_user_id(user_id: str) -> bool:
        """Delete profile by user_id"""
        profile = await UserProfile.find_one(UserProfile.user_id == user_id)
        if profile:
            await profile.delete()
            return True
        return False
