# app/schemas/chunk_record.py

from beanie import Document, Indexed, Link
from typing import Optional, List, Dict, Any
from pydantic import Field
from datetime import datetime
from app.schemas.uploaded_file import UploadedFile


class ChunkRecord(Document):
    user_id: Indexed(str) = Field(...)
    chat_id: Indexed(str) = Field(...)
    
    file: Link[UploadedFile] = Field(...)      # Foreign key link
    weaviate_id: Indexed(str, unique=True) = Field(...)  # Vector DB UUID
    
    page_number: Optional[int] = None
    chunk_index: int = Field(...)
    token_count: int = Field(...)
    
    preview_text: str = Field(..., description="First 200 chars summary preview")
    images: Optional[List[str]] = Field(default=None)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chunk_records"
