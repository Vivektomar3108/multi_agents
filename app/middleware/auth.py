# app/middleware/auth_middleware.py
from jose import jwt, JWTError, ExpiredSignatureError
import bcrypt
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.schemas.user.user_model import User
from app.schemas.user.user_tokens import UserToken
from app.config.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)  # allows fallback to cookies


# ───────────────────────────────
# 🔑 Password Hashing Utilities
# ───────────────────────────────
def hash_password(password: str) -> str:
    try:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password hashing error")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        print(bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8")))
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password verification error")


# ───────────────────────────────
# 🔑 JWT Token Utilities
# ───────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    try:
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes))
        to_encode = {**data, "exp": expire, "scope": "access_token"}
        return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    except Exception as e:
        logger.error(f"Access token creation failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create access token")


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    try:
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=settings.jwt_refresh_token_expire_days))
        to_encode = {**data, "exp": expire, "scope": "refresh_token"}
        return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    except Exception as e:
        logger.error(f"Refresh token creation failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create refresh token")


def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except JWTError:
        logger.warning("Invalid token provided")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except Exception as e:
        logger.error(f"Unexpected token decode error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token decoding error")


# ───────────────────────────────
# 🔒 Fetch Current User
# ───────────────────────────────
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = None
    
    # Authorization header
    if credentials:
        token = credentials.credentials
    
    # Fallback to cookie
    if not token and request:
        cookie_token = request.cookies.get("access_token")
        if cookie_token:
            token = cookie_token.replace("Bearer ", "").strip()
    
    if not token:
        raise credentials_exception
    
    # Decode JWT
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_exception
        if payload.get("scope") != "access_token":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token scope")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token validation error")
    
    # Fetch user via Beanie
    try:
        
        user: Optional[User] = await User.find_one(User.user_id == user_id)
        if not user:
            raise credentials_exception
        if user.login_type == "email":
            # Email login must be verified
            if not user.is_email_verified:
                raise HTTPException(status_code=401, detail="Email not verified")
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch user")


# ───────────────────────────────
# 🔒 Role-based Access Control
# ───────────────────────────────
def require_roles(*roles: str):
    async def role_checker(user: User = Depends(get_current_user)):
        try:
            if user.role not in roles:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"User role '{user.role}' not permitted")
            return user
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected role check error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Role validation error")
    return role_checker
