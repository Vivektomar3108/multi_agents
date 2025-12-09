# app/models/activity_model.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict
from datetime import datetime


class UserActivity(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB document ID

    activity_type: str = Field(
        ..., 
        max_length=64,
        description="login/logout/action/token_refresh/ai_task_execution/team_update"
    )
    user_id: Optional[str] = Field(None, description="Reference to user who performed activity")
    metadata: Dict = Field(default_factory=dict, description="Additional contextual data")
    session_id: Optional[str] = Field(None, description="Session identifier")
    ip_address: Optional[str] = Field(None, description="User IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Activity timestamp")
