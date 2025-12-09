# app/models/user_settings_model.py
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Dict, Optional


class UserSettings(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB ObjectId
    user_id: str = Field(..., description="Unique reference to the user")
    preferences: Dict[str, str] = Field(default_factory=dict)   # {"theme": "dark", ...}
    security: Dict[str, str | bool] = Field(default_factory=dict)   # {"mfa": True, ...}
    ai_config: Dict[str, str | int] = Field(
        default_factory=lambda: {"default_agent": "researcher", "max_tokens": 4096}
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
