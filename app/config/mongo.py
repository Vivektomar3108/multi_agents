# app/config/mongo.py
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.schemas.user.auth_provider_model import AuthProvider
from app.schemas.user.auth_model import AuthData
from app.schemas.user.user_model import User
from app.schemas.user.email_otp_verification import EmailOTPVerification
from app.schemas.user.user_tokens import UserToken
from app.schemas.user.profile_model import UserProfile
from app.schemas.user.preferences_model import UserPreferences
from app.schemas.user.settings_model import UserSettings
from app.schemas.user.billing_schema import BillingInfo
from app.schemas.user.token_usage_schema import TokenUsageLog
from app.schemas.user.plan_schema import Plan
from app.schemas.memory import MemoryEntry
from app.schemas.chat_session import ChatSession
from app.schemas.chunk_record import ChunkRecord
from app.schemas.chat_history import ChatHistory
from app.schemas.uploaded_file import UploadedFile
from app.schemas.stream_buffer import StreamBuffer

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "resworb")

_client: AsyncIOMotorClient | None = None


async def init_db():
    global _client

    if _client is not None:
        return _client

    try:
        _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = _client[DB_NAME]

        await init_beanie(
            database=db,
            document_models=[
                AuthData, AuthProvider, 
                BillingInfo, EmailOTPVerification, Plan,
                TokenUsageLog, User, UserToken, UserProfile,
                UserPreferences, UserSettings,MemoryEntry,ChatSession,ChatHistory,ChunkRecord,UploadedFile,StreamBuffer
            ],
        )

        logging.info(f"✅ MongoDB Connected: {DB_NAME}")
        return _client

    except Exception as e:
        logging.exception(f"❌ MongoDB connection failed: {e}")
        raise e


def get_db():
    if not _client:
        raise RuntimeError("Mongo not initialized")
    return _client[DB_NAME]


async def close_db():
    global _client
    if _client:
        _client.close()
        logging.info("🔒 MongoDB connection closed.")
        _client = None
