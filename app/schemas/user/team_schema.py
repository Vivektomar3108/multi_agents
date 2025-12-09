# app/models/team_schema.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime


class Team(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB document ID

    name: str = Field(..., max_length=120, description="Team name (must be unique)")
    description: Optional[str] = Field(None, description="Team description")
    owner_id: str = Field(..., description="User ID of the team owner")
    member_ids: List[str] = Field(default_factory=list, description="List of user IDs in the team")
    roles: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of user_id → role (owner|admin|member|viewer)"
    )
    permissions: Dict[str, bool] = Field(
        default_factory=dict,
        description="Permission flags for the team"
    )
    integrations: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Shared integrations & configurations"
    )
    team_settings: Dict[str, str] = Field(
        default_factory=dict,
        description="Team-level custom settings"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the team was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the team was last updated"
    )
