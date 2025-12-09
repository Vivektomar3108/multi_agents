from fastapi import UploadFile, HTTPException, status
from app.services.pdf_service import FileServiceBatch
from app.config.aws_s3 import S3Service
from app.config.weaviate_service import WeaviateService
import logging

logger = logging.getLogger(__name__)


class FileController:
    """Controller responsible for validating and processing uploaded documents."""

    MAX_FILE_SIZE_MB = 30  # Production file size limit (stream-safe)

    # Allowed MIME types
    ALLOWED_MIME_TYPES = {
        "application/pdf",
        "application/x-pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain",
        "text/markdown",
        "text/html",
    }

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".md", ".html", ".htm"
    }

    def __init__(self, s3_service: S3Service):
        """
        Initialize file processing dependencies and Weaviate vector DB connector.
        """
        self.weaviate_service = WeaviateService()
        self.file_service = FileServiceBatch(
            s3_service=s3_service,
            weaviate_service=self.weaviate_service
        )

    async def handle_file(self, file: UploadFile, user_id: str, chat_id: str) -> dict:
        """
        Validates, streams, extracts, embeds, and stores file content.

        Args:
            file (UploadFile): The uploaded document
            user_id (str): Owner user identifier (from JWT)
            chat_id (str): Target conversation ID to group chunks

        Returns:
            dict: Processing summary and metadata
        """
        filename = file.filename or "uploaded_file"

        try:
            # ---------------------------
            # 1. Validate MIME type
            # ---------------------------
            if file.content_type not in self.ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Invalid file type: {file.content_type}. "
                        f"Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
                    ),
                )

            # ---------------------------
            # 2. Validate file extension
            # ---------------------------
            if not any(filename.lower().endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported extension. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
                )

            # ---------------------------
            # 3. Validate file size in streamed chunks
            # ---------------------------
            file_size = 0
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB per read
                if not chunk:
                    break

                file_size += len(chunk)

                if file_size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {self.MAX_FILE_SIZE_MB}MB limit"
                    )

            # Reset stream for later processing
            await file.seek(0)

            logger.info(f"[User: {user_id} | Chat: {chat_id}] Processing file: {filename}")

            # ---------------------------
            # 4. Hybrid RAG processing
            # ---------------------------
            result_map = await self.file_service.process_files(
                files=[file],
                filenames=[filename],
                metadata={
                    "user_id": user_id,
                    "chat_id": chat_id
                }
            )

            final_result = result_map.get(filename, {})

            return {
                "success": True,
                "filename": filename,
                "user_id": user_id,
                "chat_id": chat_id,
                "processing_result": final_result
            }

        except HTTPException:
            raise

        except Exception as e:
            logger.exception("Unexpected error in FileController.handle_file()")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected server error during document processing."
            )
