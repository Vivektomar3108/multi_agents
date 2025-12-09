# app/models/agent_permissions.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime


class AgentPermissions(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB document ID
    user_id: str = Field(..., description="Reference to the user")

    allowed_tools: List[str] = Field(default_factory=list, description="List of allowed tools")
    restricted_tools: List[str] = Field(default_factory=list, description="List of restricted tools")

    usage_limits: Dict[str, int] = Field(
        default_factory=dict,
        description="Tool usage limits, e.g. {'search': 100, 'crawl': 50}"
    )
    exemptions: Dict[str, bool] = Field(
        default_factory=dict,
        description="Exemptions from rules, e.g. {'admin_override': True}"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
