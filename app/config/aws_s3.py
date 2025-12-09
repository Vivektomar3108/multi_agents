import boto3
import uuid
import logging
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException
from typing import Optional
from app.config.setting import settings

logger = logging.getLogger(__name__)


class S3Service:
    def __init__(
        self,
        bucket_name: str = settings.s3_bucket,
        region_name: str = settings.aws_default_region,
        access_key: Optional[str] = settings.aws_access_key_id,
        secret_key: Optional[str] = settings.aws_secret_access_key,
        public: bool = False,
    ):
        """
        S3 Upload Service (production-grade)

        :param bucket_name: S3 bucket name
        :param region_name: AWS region
        :param access_key: Optional - uses IAM Role if not provided
        :param secret_key: Optional
        :param public: Allow public read URLs
        """

        self.bucket = bucket_name
        self.region = region_name
        self.public = public

        # Best practice → Use IAM Role credentials if access keys not provided
        session_args = {"region_name": region_name}
        if access_key and secret_key:
            session_args["aws_access_key_id"] = access_key
            session_args["aws_secret_access_key"] = secret_key

        try:
            self.s3 = boto3.client("s3", **session_args)
            logger.info("S3Service initialized successfully.")
        except Exception:
            logger.exception("Failed to initialize S3 client")
            raise RuntimeError("Failed to initialize AWS S3 client.")

    # =========================
    # Upload FileObj (UploadFile)
    # =========================
    async def upload_file(self, file) -> str:
        """
        Uploads a FastAPI UploadFile and returns the S3 URL.
        """
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        file_ext = file.filename.split(".")[-1]
        key = f"uploads/{uuid.uuid4()}.{file_ext}"

        try:
            self.s3.upload_fileobj(
                file.file,
                self.bucket,
                key,
                ExtraArgs={"ACL": "public-read"} if self.public else {},
            )
        except (BotoCoreError, ClientError):
            logger.exception("S3 upload_fileobj failed")
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")

        return self._build_url(key)

    # =========================
    # Upload Bytes (PDF image upload)
    # =========================
    def upload_bytes(self, content: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload raw bytes to S3 (used for PDF extracted images)
        """
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
                ACL="public-read" if self.public else "private",
            )
        except (BotoCoreError, ClientError):
            logger.exception("S3 put_object (bytes upload) failed")
            raise HTTPException(status_code=500, detail="Failed to upload bytes to S3")

        return self._build_url(key)

    # =========================
    # Build Full S3 URL
    # =========================
    def _build_url(self, key: str) -> str:
        """
        Build public/private S3 URL.
        """
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
