from typing import Optional, Dict, List
from datetime import datetime
from beanie import Document, Indexed
from pydantic import BaseModel, Field
from bson import ObjectId


class ProviderToken(BaseModel):
    """
    Standardized OAuth token schema for provider-level credentials.
    Stores access and refresh tokens along with expiration and metadata.
    """
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    token_type: Optional[str] = None
    issued_at: datetime = Field(default_factory=datetime.utcnow)


class AgentAuth(BaseModel):
    """
    Agent-level authentication data.
    Each agent (e.g., EmailAgent, CalendarAgent, CRM) can store its own tokens or credentials here.
    """
    agent_name: str = Field(..., description="Name of the agent (e.g., gmail, notion, slack)")
    provider: Optional[str] = Field(None, description="OAuth provider name (e.g., Google, GitHub)")
    provider_user_id: Optional[str] = Field(None, description="User ID from the provider side")
    credentials: Optional[ProviderToken] = Field(None, description="OAuth credentials or API tokens")
    additional_data: Dict[str, str] = Field(default_factory=dict, description="Extra metadata or identifiers")
    last_refreshed: Optional[datetime] = Field(None, description="Last token refresh timestamp")
    

class AuthData(Document):
    """
    Comprehensive authentication model for a user.
    Supports:
      - Local credentials
      - Multi-provider OAuth data
      - Per-agent authentication
      - Multi-factor authentication
    """

    user_id: Optional[str] = Field(None, description="Associated user ID", index=True)

    # Local password hashing
    salt: Optional[str] = Field(None, description="Salt for password hashing (if applicable)")

    # Global OAuth providers (legacy/global)
    oauth_providers: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Global OAuth providers (e.g., google, github, etc.)"
    )

    # Agent-specific authentication (new modular design)
    agent_auth: List[AgentAuth] = Field(
        default_factory=list,
        description="List of per-agent authentication data"
    )

    # Local session tokens or API keys (non-OAuth)
    token: Dict = Field(default_factory=dict, description="All other token types or API keys")

    # Security controls
    mfa_enabled: bool = Field(default=False, description="Multi-factor authentication enabled")
    mfa_method: Optional[str] = Field(None, description="MFA method (sms, email, authenticator app)")
    recovery_codes: List[str] = Field(default_factory=list, description="MFA recovery codes")

    # Metadata
    last_used: Optional[datetime] = Field(None, description="Last time this auth record was used")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "auth_data"  # MongoDB collection name
        use_state_management = True

    