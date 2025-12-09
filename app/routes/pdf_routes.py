from fastapi import (
    APIRouter, 
    UploadFile, 
    File, 
    HTTPException, 
    status, 
    Depends, 
    Query
)
from fastapi.responses import JSONResponse
from app.controllers.pdf_controller import FileController
from app.config.aws_s3 import S3Service
from app.config.setting import settings
from app.middleware.auth import get_current_user
from app.schemas.user.user_model import User
import logging

logger = logging.getLogger(__name__)

file_router = APIRouter(
    prefix="/file",
    tags=["File Processing"],
)

# ============================================
# Initialize S3 + File Controller (Singleton)
# ============================================
try:
    s3_service = S3Service(
        bucket_name=settings.s3_bucket,
        region_name=settings.aws_default_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key,
    )

    file_controller = FileController(s3_service=s3_service)

except Exception as exc:
    logger.exception("Failed to initialize S3Service or FileController")
    raise RuntimeError("Critical initialization error: cannot start file services") from exc


# ============================================
# Upload → Full Hybrid Processing
# ============================================
@file_router.post(
    "/chunk",
    summary=(
        "Upload a file → extract text/images → upload images → chunk → "
        "embed (GPU) → store in Weaviate with user + chat isolation."
    ),
    responses={
        200: {"description": "File processed successfully"},
        400: {"description": "Invalid file upload"},
        413: {"description": "File too large"},
        500: {"description": "Server error"},
    },
)
async def chunk_file(
    chat_id: str = Query(..., description="Unique chat/session identifier"),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """
    Hybrid RAG processing pipeline with multi-tenant support:

    - Validates file type/size
    - Extracts text (unstructured) + images (PyMuPDF)
    - Uploads images to S3
    - Replaces placeholders with public S3 links
    - Uses smart chunking + embeddings (GPU)
    - Stores vector chunks in Weaviate tagged by `user_id` & `chat_id`
    """

    try:
        result = await file_controller.handle_file(
            file=file,
            user_id=str(user.user_id),
            chat_id=chat_id
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result,
            headers={
                "X-Service": "rag-file-processor",
                "X-User-ID": str(user.user_id),
                "X-Chat-ID": chat_id,
            },
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected error in /file/chunk endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while processing the file.",
        )
