# app/schemas/user/contact_schema.py
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime


class Contact(BaseModel):
    """Contact form submission schema"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: Optional[str] = Field(None, alias="_id")  # MongoDB document ID

    full_name: str = Field(
        ..., 
        min_length=2,
        max_length=128, 
        description="Full name of the contact person"
    )
    email: EmailStr = Field(..., description="Contact email address")
    company: Optional[str] = Field(
        None, 
        max_length=256, 
        description="Company name (optional)"
    )
    subject: str = Field(
        ..., 
        min_length=3,
        max_length=256, 
        description="Subject of the contact message"
    )
    message: str = Field(
        ..., 
        min_length=10,
        max_length=5000, 
        description="Contact message content"
    )
    
    status: str = Field(
        default="pending", 
        description="Status: pending/reviewed/replied/archived"
    )
    priority: str = Field(
        default="normal",
        description="Priority level: low/normal/high/urgent"
    )
    assigned_to: Optional[str] = Field(
        None, 
        description="User ID of assigned team member"
    )
    
    ip_address: Optional[str] = Field(None, description="IP address of submitter")
    user_agent: Optional[str] = Field(None, description="Browser user agent")
    source: str = Field(
        default="web",
        description="Source of contact: web/mobile/api"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Contact submission timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Last update timestamp"
    )
    reviewed_at: Optional[datetime] = Field(
        None, 
        description="When the contact was reviewed"
    )
    replied_at: Optional[datetime] = Field(
        None, 
        description="When a reply was sent"
    )
        json_schema_extra = {
            "example": {
                "full_name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Corp",
                "subject": "Product Inquiry",
                "message": "I would like to learn more about your services and pricing options.",
                "status": "pending",
                "priority": "normal",
                "source": "web"
            }
        }
