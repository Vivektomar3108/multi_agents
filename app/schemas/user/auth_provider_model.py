from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field, HttpUrl, EmailStr
from typing import Optional, Literal
from datetime import datetime


class AuthProvider(Document):
    """
    MongoDB model for storing OAuth/auth provider info linked to a user.
    """

    user_id: Optional[str] = Field(None, description="Linked User ID")

    provider: Literal["google", "github", "facebook", "twitter", "apple"] = Field(
        ..., description="Auth provider name"
    )

    provider_user_id: Indexed(str) = Field( # type: ignore
        ..., description="Unique ID of the user from the provider"
    )

    email: Optional[EmailStr] = Field(None, description="Email returned from provider")
    name: Optional[str] = Field(None, description="Display name from provider")
    avatar_url: Optional[HttpUrl] = Field(None, description="Profile image URL from provider")

    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expiry: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "user_auth"
        indexes = [
            ["provider_user_id"],  # single index
            "user_id",
            "email",
        ]
