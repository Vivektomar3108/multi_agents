# app/models/user_activity_model.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict
from datetime import datetime


class UserActivity(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB document ID
    user_id: str = Field(..., description="User performing the action")
    action: str = Field(..., max_length=128, description="Type of action performed")
    details: Optional[Dict] = Field(None, description="Additional metadata about the action")
    session_id: Optional[str] = Field(None, description="User session identifier")
    ip_address: Optional[str] = Field(None, description="IP address of the user")
    user_agent: Optional[str] = Field(None, description="User's device/browser info")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the action occurred")
