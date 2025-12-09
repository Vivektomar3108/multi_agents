# app/schemas/user/token_usage_schema.py
from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime
from pymongo import IndexModel, ASCENDING, DESCENDING


class TokenUsageLog(Document):
    """
    Logs every token usage by users for agents and services.
    Provides audit trail and analytics for token consumption.
    """
    
    # Core fields
    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")
    user_id: str = Field(..., description="Reference to the user")
    
    # Usage details
    agent_type: str = Field(..., description="Type of agent used (e.g., 'email', 'websearch', 'sheet')")
    agent_name: Optional[str] = Field(None, description="Specific agent name or identifier")
    tokens_used: int = Field(..., ge=0, description="Number of tokens consumed")
    tokens_before: int = Field(..., ge=0, description="Token balance before usage")
    tokens_after: int = Field(..., ge=0, description="Token balance after usage")
    
    # Operation context
    operation: str = Field(..., description="Operation performed (e.g., 'send_email', 'web_search', 'analyze_data')")
    status: str = Field(..., description="Status: success/failed/partial")
    
    # Request details
    request_id: Optional[str] = Field(None, description="Unique request identifier for tracking")
    endpoint: Optional[str] = Field(None, description="API endpoint that was called")
    
    # Metadata
    metadata: Dict = Field(
        default_factory=dict, 
        description="Additional context (prompt length, response size, etc.)"
    )
    error_message: Optional[str] = Field(None, description="Error message if status is failed")
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="When the token usage occurred"
    )

    class Settings:
        name = "token_usage_logs"
        indexes = [
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel([("user_id", ASCENDING), ("agent_type", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("request_id", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
        ]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            PydanticObjectId: str,
        }


class TokenUsageCreateSchema(BaseModel):
    """Schema for creating token usage log entries"""
    user_id: str
    agent_type: str
    agent_name: Optional[str] = None
    tokens_used: int = Field(..., ge=0)
    tokens_before: int = Field(..., ge=0)
    tokens_after: int = Field(..., ge=0)
    operation: str
    status: str = "success"
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    error_message: Optional[str] = None


class TokenUsageSummary(BaseModel):
    """Summary of token usage for analytics"""
    total_tokens_used: int
    total_operations: int
    successful_operations: int
    failed_operations: int
    tokens_by_agent: Dict[str, int]
    operations_by_agent: Dict[str, int]
