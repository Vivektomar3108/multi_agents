from pydantic import Field, EmailStr, validator
from datetime import datetime
from typing import Optional, List, Literal
from bson import ObjectId
from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from app.schemas.user.profile_model import UserProfile
from app.schemas.user.auth_model import AuthData
from app.schemas.user.preferences_model import UserPreferences
from app.schemas.user.settings_model import UserSettings
from app.schemas.user.activity_model import UserActivity
from app.schemas.user.user_tokens import UserToken
from app.schemas.user.auth_provider_model import AuthProvider


class User(Document):
    """Unified user model for MongoDB (Beanie ODM)."""

    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")
    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email (must be unique)")
    username: str = Field(..., max_length=64, description="Unique username")
    password: Optional[str] = Field(None, description="Hashed password (None if OAuth only)")

    # Auth
    login_type: Literal["email", "google", "github", "facebook", "apple"] = Field(
        default="email", description="User login method"
    )
    auth_providers: List[AuthProvider] = Field(default_factory=list, description="Linked OAuth providers")

    # Profile & preferences
    profile: Optional[UserProfile] = None
    preferences: Optional[UserPreferences] = None
    settings: Optional[UserSettings] = None
    activities: List[UserActivity] = Field(default_factory=list)
    tokens: List[UserToken] = Field(default_factory=list)

    #email verification
    is_email_verified: bool = Field(default=False, description="Whether user's email is verified via OTP")
    email_verified_at: Optional[datetime] = Field(default=None, description="Timestamp when email was verified")
    
    # Metadata
    role: str = Field(default="user", description="Role: user/admin/superadmin")
    is_active: Literal["active", "inactive", "banned"] = Field(default="active", description="Account status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    class Settings:
        name = "users"
        indexes = [
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel([("username", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING)], unique=True),
        ]

    class Config:
        validate_by_name = True  # replaces allow_population_by_field_name in Pydantic v2
        json_encoders = {datetime: lambda v: v.isoformat()}
        arbitrary_types_allowed = True

    @validator("auth_providers", pre=True, always=True)
    def normalize_auth_providers(cls, v):
        """Normalize linked OAuth provider data."""
        if not v:
            return []
        if isinstance(v, list) and all(isinstance(x, AuthProvider) for x in v):
            return v
        if isinstance(v, list) and all(isinstance(x, dict) for x in v):
            return [AuthProvider(**x) for x in v]
        if isinstance(v, list) and all(isinstance(x, (str, ObjectId)) for x in v):
            return [str(x) for x in v]
        if isinstance(v, (str, ObjectId)):
            return [str(v)]
        return v
