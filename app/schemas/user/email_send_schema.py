# app/models/email_send_schema.py
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import Dict, Optional
from datetime import datetime


class EmailLog(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB document ID

    user_id: Optional[str] = Field(None, description="User who triggered the email")
    email_type: str = Field(
        ..., 
        max_length=64, 
        description="Type of email: verification, reset_password, alert, newsletter"
    )
    recipient: EmailStr = Field(..., description="Recipient email address")
    subject: str = Field(..., max_length=512, description="Email subject line")
    body: str = Field(..., description="Email body content")
    status: str = Field(
        "pending", 
        description="Email status: pending/sent/failed/opened"
    )
    retry_count: int = Field(0, description="Number of retry attempts")
    metadata: Dict = Field(
        default_factory=dict, 
        description="Provider response, message_id, tracking data"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Log creation timestamp"
    )
    sent_at: Optional[datetime] = Field(None, description="When email was sent")
    opened_at: Optional[datetime] = Field(None, description="When recipient opened email")
    is_important: bool = Field(
        False, 
        description="Flag if the email is high priority"
    )
