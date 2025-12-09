# app/middleware/token_middleware.py
"""
Middleware for validating token balance before agent execution.
Similar to auth middleware, but checks if user has sufficient tokens.
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from app.schemas.user.user_model import User
from app.middleware.auth import get_current_user
from app.services.user.token_service import TokenService

logger = logging.getLogger(__name__)


async def check_token_balance(
    agent_type: str,
    custom_cost: Optional[int] = None,
    user: User = Depends(get_current_user)
) -> User:
    """
    Dependency that checks if user has sufficient tokens for the requested agent.
    
    Args:
        agent_type: Type of agent being used (e.g., 'email', 'websearch')
        custom_cost: Override the default token cost for this agent
        user: Current authenticated user
    
    Returns:
        User object if they have sufficient tokens
    
    Raises:
        HTTPException: If insufficient tokens
    """
    # Get the token cost for this agent
    token_cost = custom_cost if custom_cost is not None else TokenService.get_agent_cost(agent_type)
    
    # Check if user has sufficient tokens
    has_tokens = await TokenService.has_sufficient_tokens(user.user_id, token_cost)
    
    if not has_tokens:
        current_balance = await TokenService.get_user_tokens(user.user_id)
        logger.warning(
            f"Insufficient tokens for user {user.user_id}: "
            f"has {current_balance}, needs {token_cost} for {agent_type}"
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "Insufficient tokens",
                "current_balance": current_balance,
                "required": token_cost,
                "agent_type": agent_type,
                "message": f"You need {token_cost} tokens to use {agent_type} agent, but you only have {current_balance} tokens."
            }
        )
    
    logger.info(f"Token check passed for user {user.user_id}: agent={agent_type}, cost={token_cost}")
    return user


def require_tokens(agent_type: str, custom_cost: Optional[int] = None):
    """
    Factory function to create a token validation dependency.
    
    Usage in routes:
        @router.post("/email/send")
        async def send_email(
            data: EmailRequest,
            user: User = Depends(require_tokens("email"))
        ):
            # User has been validated and has sufficient tokens
            ...
    
    Args:
        agent_type: Type of agent (e.g., 'email', 'websearch')
        custom_cost: Optional custom token cost (overrides default)
    
    Returns:
        Dependency function that can be used with FastAPI Depends()
    """
    async def token_checker(user: User = Depends(get_current_user)):
        return await check_token_balance(agent_type, custom_cost, user)
    
    return token_checker


async def get_user_with_balance(user: User = Depends(get_current_user)) -> tuple[User, int]:
    """
    Dependency that returns both the user and their current token balance.
    Useful for endpoints that need to display token information.
    
    Returns:
        Tuple of (User, token_balance)
    """
    balance = await TokenService.get_user_tokens(user.user_id)
    return user, balance


# ───────────────────────────────
# Utility Functions
# ───────────────────────────────

async def validate_and_deduct_tokens(
    user_id: str,
    agent_type: str,
    operation: str,
    agent_name: Optional[str] = None,
    custom_cost: Optional[int] = None,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    metadata: Optional[dict] = None
) -> tuple[bool, Optional[int]]:
    """
    Validate token balance and deduct tokens in one operation.
    Use this in agent execution logic.
    
    Returns:
        Tuple of (success: bool, tokens_after: Optional[int])
    """
    token_cost = custom_cost if custom_cost is not None else TokenService.get_agent_cost(agent_type)
    
    # Check balance
    has_tokens = await TokenService.has_sufficient_tokens(user_id, token_cost)
    if not has_tokens:
        logger.warning(f"Cannot deduct tokens - insufficient balance for user {user_id}")
        return False, None
    
    # Deduct tokens
    success, usage_log = await TokenService.deduct_tokens(
        user_id=user_id,
        tokens_to_deduct=token_cost,
        agent_type=agent_type,
        operation=operation,
        agent_name=agent_name,
        request_id=request_id,
        endpoint=endpoint,
        metadata=metadata
    )
    
    if success and usage_log:
        return True, usage_log.tokens_after
    
    return False, None
