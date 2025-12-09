from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime


class Workspace(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB ObjectId
    name: str = Field(..., max_length=120)
    description: Optional[str] = None
    owner_id: str
    team_ids: List[str] = Field(default_factory=list)
    user_ids: List[str] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
