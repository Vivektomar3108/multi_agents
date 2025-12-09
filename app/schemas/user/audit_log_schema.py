# app/models/audit_log_schema.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict
from datetime import datetime


class AuditLog(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB document ID

    actor_user_id: Optional[str] = Field(
        None, description="User who performed the action"
    )
    target_type: Optional[str] = Field(
        None, description="Entity type affected (e.g., user, team, workspace)"
    )
    target_id: Optional[str] = Field(
        None, description="ID of the entity affected"
    )
    action: str = Field(
        ..., description="Action performed (e.g., update_profile, delete_token)"
    )
    details: Dict = Field(
        default_factory=dict,
        description="Extra info: before/after snapshots, diffs, reason"
    )
    ip_address: Optional[str] = Field(None, description="IP address of actor")
    user_agent: Optional[str] = Field(None, description="User agent of actor")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    severity: Optional[str] = Field(
        "info", description="Log severity: info, warn, error"
    )
