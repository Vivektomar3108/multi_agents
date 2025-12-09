from pydantic import Field, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional, Literal
from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING


class EmailOTPVerification(Document):
    """
    Schema to manage OTP-based email verification.
    Each OTP is unique, time-bound, and linked to a specific email.
    """

    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")
    email: EmailStr = Field(..., description="Email for which OTP is generated")
    otp_code: str = Field(..., min_length=4, max_length=8, description="One-Time Password (numeric or alphanumeric)")
    purpose: Literal["login", "signup", "password_reset", "email_verification"] = Field(
        default="email_verification",
        description="Purpose of OTP verification"
    )
    is_verified: bool = Field(default=False, description="Whether the OTP has been successfully verified")
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=10))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    verified_at: Optional[datetime] = Field(default=None, description="Timestamp when OTP was verified")
    attempts: int = Field(default=0, description="Number of OTP verification attempts made")

    class Settings:
        name = "email_otp_verifications"
        indexes = [
            IndexModel([("email", ASCENDING)], unique=False),
            IndexModel([("otp_code", ASCENDING)], unique=False),
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),
        ]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        arbitrary_types_allowed = True

    @validator("otp_code")
    def validate_otp_code(cls, v):
        if not v.isalnum():
            raise ValueError("OTP code must be alphanumeric")
        return v

    def is_expired(self) -> bool:
        """Check if OTP is expired."""
        return datetime.utcnow() > self.expires_at

    def mark_verified(self):
        """Mark OTP as successfully verified."""
        self.is_verified = True
        self.verified_at = datetime.utcnow()
