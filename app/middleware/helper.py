# app/db/user_helpers.py
import logging
from typing import Optional
from bson import ObjectId
from app.config.mongo import get_db
from app.schemas.user.user_model import User
from app.schemas.user.user_tokens import UserToken

logger = logging.getLogger(__name__)


# ──────────────────────────────
# User Helpers
# ──────────────────────────────
async def get_user_by_id(user_id: str) -> Optional[User]:
    """Fetch a user by user_id using Motor (async)."""
    try:
        db = get_db()
        collection = db["users"]
        data = await collection.find_one({"user_id": user_id})

        if not data:
            return None

        if "_id" in data and isinstance(data["_id"], ObjectId):
            data["_id"] = str(data["_id"])

        return User(**data)
    except Exception as e:
        logger.exception(f"Error fetching user by ID {user_id}: {e}")
        return None


async def get_user_by_email(email: str) -> Optional[User]:
    """Fetch a user by email using Motor (async)."""
    try:
        db = get_db()
        collection = db["users"]
        data = await collection.find_one({"email": email})

        if not data:
            return None

        if "_id" in data and isinstance(data["_id"], ObjectId):
            data["_id"] = str(data["_id"])

        return User(**data)
    except Exception as e:
        logger.exception(f"Error fetching user by email {email}: {e}")
        return None


# ──────────────────────────────
# Token Helpers
# ──────────────────────────────
async def get_token_by_value(token: str) -> Optional[UserToken]:
    """Fetch a user token by token value using Motor (async)."""
    try:
        db = get_db()
        collection = db["user_tokens"]
        data = await collection.find_one({"token": token})

        if not data:
            return None

        if "_id" in data and isinstance(data["_id"], ObjectId):
            data["_id"] = str(data["_id"])

        return UserToken(**data)
    except Exception as e:
        logger.exception(f"Error fetching token {token}: {e}")
        return None
