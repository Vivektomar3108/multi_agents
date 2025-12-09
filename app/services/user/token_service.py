# app/services/user/token_service.py
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta
from beanie import PydanticObjectId
from app.schemas.user.token_usage_schema import (
    TokenUsageLog, 
    TokenUsageCreateSchema,
    TokenUsageSummary
)
from app.schemas.user.billing_schema import BillingInfo
from app.services.user.billing_service import BillingService
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────
# Agent Token Costs Configuration
# ──────────────────────────────
AGENT_TOKEN_COSTS = {
    "email": 100,
    "websearch": 50,
    "sheet": 75,
    "orchestrator": 150,
    "executor": 100,
    "helpbot": 25,
    "blog": 80,
    # Add more agents as needed
}

# Default token allocation by plan (fallback if Plan schema not found)
PLAN_TOKEN_ALLOCATION = {
    "free": 1000,
    "professional": 20000,
    "enterprise": 100000,
}


class TokenService:
    """Service for managing user tokens and usage tracking"""

    @staticmethod
    async def get_user_tokens(user_id: str) -> int:
        """Get current token balance for a user"""
        billing = await BillingInfo.find_one(BillingInfo.user_id == user_id)
        if not billing:
            logger.warning(f"No billing info found for user {user_id}")
            return 0
        return billing.usage_credits
    
    @staticmethod
    async def get_plan_token_allocation(plan_name: str) -> int:
        """
        Get token allocation for a plan from Plan schema.
        Falls back to PLAN_TOKEN_ALLOCATION if plan not found.
        """
        from app.schemas.user.plan_schema import Plan
        
        # Try to find plan in database (check both with and without billing interval)
        plan = await Plan.find_one({"name": {"$regex": f"^{plan_name}$", "$options": "i"}})
        if not plan:
            plan = await Plan.find_one({"plan_id": {"$regex": f"^{plan_name}", "$options": "i"}})
        
        if plan and plan.token_allocation:
            return plan.token_allocation.initial_tokens
        
        # Fallback to hardcoded values
        return PLAN_TOKEN_ALLOCATION.get(plan_name.lower(), 1000)

    @staticmethod
    async def has_sufficient_tokens(user_id: str, required_tokens: int) -> bool:
        """Check if user has enough tokens"""
        current_tokens = await TokenService.get_user_tokens(user_id)
        return current_tokens >= required_tokens

    @staticmethod
    async def deduct_tokens(
        user_id: str,
        tokens_to_deduct: int,
        agent_type: str,
        operation: str,
        agent_name: Optional[str] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> Tuple[bool, Optional[TokenUsageLog]]:
        """
        Deduct tokens from user and log the usage.
        Returns (success: bool, log: TokenUsageLog)
        """
        billing_service = BillingService()
        
        # Get current balance
        tokens_before = await TokenService.get_user_tokens(user_id)
        
        if tokens_before < tokens_to_deduct:
            logger.warning(f"Insufficient tokens for user {user_id}: has {tokens_before}, needs {tokens_to_deduct}")
            return False, None
        
        # Deduct tokens from billing
        billing = await billing_service.deduct_credits(user_id, tokens_to_deduct)
        
        if not billing:
            logger.error(f"Failed to deduct tokens for user {user_id}")
            return False, None
        
        tokens_after = billing.usage_credits
        
        # Log the usage
        usage_log = TokenUsageLog(
            user_id=user_id,
            agent_type=agent_type,
            agent_name=agent_name,
            tokens_used=tokens_to_deduct,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            operation=operation,
            status=status,
            request_id=request_id,
            endpoint=endpoint,
            metadata=metadata or {},
            error_message=error_message,
            created_at=datetime.utcnow()
        )
        
        await usage_log.insert()
        logger.info(f"Token usage logged: user={user_id}, agent={agent_type}, tokens={tokens_to_deduct}, new_balance={tokens_after}")
        
        return True, usage_log

    @staticmethod
    async def refund_tokens(
        user_id: str,
        tokens_to_refund: int,
        agent_type: str,
        operation: str,
        reason: str,
        agent_name: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Refund tokens to user (e.g., if operation failed).
        Also logs the refund as a usage entry.
        """
        billing_service = BillingService()
        
        tokens_before = await TokenService.get_user_tokens(user_id)
        
        # Add tokens back
        billing = await billing_service.add_credits(user_id, tokens_to_refund)
        
        if not billing:
            logger.error(f"Failed to refund tokens for user {user_id}")
            return False
        
        tokens_after = billing.usage_credits
        
        # Log the refund
        usage_log = TokenUsageLog(
            user_id=user_id,
            agent_type=agent_type,
            agent_name=agent_name,
            tokens_used=-tokens_to_refund,  # Negative to indicate refund
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            operation=f"refund_{operation}",
            status="refunded",
            request_id=request_id,
            metadata={"reason": reason},
            created_at=datetime.utcnow()
        )
        
        await usage_log.insert()
        logger.info(f"Tokens refunded: user={user_id}, amount={tokens_to_refund}, reason={reason}")
        
        return True

    @staticmethod
    async def get_user_usage_logs(
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        agent_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[TokenUsageLog], int]:
        """Get usage logs for a user with optional filtering"""
        query = TokenUsageLog.find(TokenUsageLog.user_id == user_id)
        
        if agent_type:
            query = query.find(TokenUsageLog.agent_type == agent_type)
        
        if start_date:
            query = query.find(TokenUsageLog.created_at >= start_date)
        
        if end_date:
            query = query.find(TokenUsageLog.created_at <= end_date)
        
        query = query.sort("-created_at")
        total = await query.count()
        logs = await query.skip(offset).limit(limit).to_list()
        
        return logs, total

    @staticmethod
    async def get_usage_summary(
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> TokenUsageSummary:
        """Get aggregated usage summary for a user"""
        query = TokenUsageLog.find(TokenUsageLog.user_id == user_id)
        
        if start_date:
            query = query.find(TokenUsageLog.created_at >= start_date)
        
        if end_date:
            query = query.find(TokenUsageLog.created_at <= end_date)
        
        logs = await query.to_list()
        
        total_tokens = sum(log.tokens_used for log in logs if log.tokens_used > 0)
        total_ops = len(logs)
        successful = sum(1 for log in logs if log.status == "success")
        failed = sum(1 for log in logs if log.status == "failed")
        
        # Aggregate by agent type
        tokens_by_agent: Dict[str, int] = {}
        ops_by_agent: Dict[str, int] = {}
        
        for log in logs:
            if log.tokens_used > 0:  # Exclude refunds
                tokens_by_agent[log.agent_type] = tokens_by_agent.get(log.agent_type, 0) + log.tokens_used
                ops_by_agent[log.agent_type] = ops_by_agent.get(log.agent_type, 0) + 1
        
        return TokenUsageSummary(
            total_tokens_used=total_tokens,
            total_operations=total_ops,
            successful_operations=successful,
            failed_operations=failed,
            tokens_by_agent=tokens_by_agent,
            operations_by_agent=ops_by_agent
        )

    @staticmethod
    async def initialize_user_tokens(user_id: str, plan: str = "free") -> bool:
        """
        Initialize token balance for a new user based on their plan.
        This should be called when creating a new billing record.
        Fetches token allocation from Plan schema.
        """
        from app.services.user.billing_service import BillingCreateSchema
        from app.schemas.user.billing_schema import PlanType
        
        # Get token allocation from Plan schema
        token_allocation = await TokenService.get_plan_token_allocation(plan)
        
        billing_data = BillingCreateSchema(
            user_id=user_id,
            plan=PlanType(plan),
            usage_credits=token_allocation,
            total_credits=token_allocation,
        )
        
        billing_service = BillingService()
        
        try:
            await billing_service.create_billing(billing_data)
            logger.info(f"Initialized tokens for user {user_id} with {token_allocation} tokens (plan: {plan})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize tokens for user {user_id}: {e}")
            return False

    @staticmethod
    def get_agent_cost(agent_type: str) -> int:
        """Get token cost for a specific agent type"""
        return AGENT_TOKEN_COSTS.get(agent_type.lower(), 50)  # Default 50 tokens

    @staticmethod
    async def add_tokens_to_user(
        user_id: str,
        tokens_to_add: int,
        reason: str = "manual_addition"
    ) -> bool:
        """Add tokens to user balance (admin operation or purchase)"""
        billing_service = BillingService()
        
        tokens_before = await TokenService.get_user_tokens(user_id)
        billing = await billing_service.add_credits(user_id, tokens_to_add)
        
        if not billing:
            return False
        
        # Log the addition (use 0 for tokens_used since this is an addition, not consumption)
        usage_log = TokenUsageLog(
            user_id=user_id,
            agent_type="system",
            tokens_used=0,  # No tokens consumed, this is an addition
            tokens_before=tokens_before,
            tokens_after=billing.usage_credits,
            operation="token_addition",
            status="success",
            metadata={
                "reason": reason,
                "tokens_added": tokens_to_add  # Store the actual amount added
            },
            created_at=datetime.utcnow()
        )
        
        await usage_log.insert()
        logger.info(f"Tokens added: user={user_id}, amount={tokens_to_add}, reason={reason}")
        
        return True
