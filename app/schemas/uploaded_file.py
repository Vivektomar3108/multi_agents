# app/schemas/uploaded_file.py

from beanie import Document, Indexed, Link
from datetime import datetime
from typing import Optional, List
from pydantic import Field
from enum import Enum
from app.schemas.chat_session import ChatSession


class FileStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class UploadedFile(Document):
    user_id: Indexed(str) = Field(...)
    chat: Link[ChatSession] = Field(...)   # Foreign reference

    file_id: Indexed(str, unique=True) = Field(...)
    file_name: str = Field(...)
    mime_type: str = Field(...)
    size_bytes: int = Field(...)

    # 🔹 Public HTTP S3 URL for main uploaded file
    s3_url: Optional[str] = Field(None)

    # 🔹 NEW: List of extracted image URLs from the file
    images: List[str] = Field(default_factory=list)

    status: FileStatus = Field(default=FileStatus.uploaded)

    page_count: Optional[int] = None
    total_chunks: Optional[int] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "uploaded_files"

    # ------------------------
    # Utility methods
    # ------------------------
    def update_progress(self, status: FileStatus):
        self.status = status
        self.updated_at = datetime.utcnow()

    def add_images(self, new_images: List[str]):
        """Append new extracted image URLs and update timestamp."""
        self.images.extend(new_images)
        self.updated_at = datetime.utcnow()

    def set_file_url(self, url: str):
        """Store final uploaded S3 HTTP file URL."""
        self.s3_url = url
        self.updated_at = datetime.utcnow()
