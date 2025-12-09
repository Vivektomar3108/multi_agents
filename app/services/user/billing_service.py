# app/services/user/billing_service.py
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta
from beanie import PydanticObjectId
from pydantic import BaseModel, Field
from app.schemas.user.billing_schema import (
    BillingInfo,
    PaymentProvider,
    PlanType,
    SubscriptionStatus,
    BillingInterval,
    TransactionRecord,
    BillingAddress
)


# ──────────────────────────────
# Pydantic Schemas for Validation
# ──────────────────────────────
class BillingCreateSchema(BaseModel):
    user_id: str
    plan: PlanType = Field(default=PlanType.FREE)
    billing_interval: BillingInterval = Field(default=BillingInterval.MONTHLY)
    usage_credits: int = Field(default=0, ge=0)
    total_credits: int = Field(default=0, ge=0)
    payment_provider: PaymentProvider = Field(default=PaymentProvider.MANUAL)
    subscription_status: SubscriptionStatus = Field(default=SubscriptionStatus.ACTIVE)
    amount: float = Field(default=0.0, ge=0)
    currency: str = Field(default="USD")
    
    # Provider-specific IDs
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_payment_method_id: Optional[str] = None
    stripe_price_id: Optional[str] = None
    
    razorpay_customer_id: Optional[str] = None
    razorpay_subscription_id: Optional[str] = None
    razorpay_payment_method_id: Optional[str] = None
    razorpay_plan_id: Optional[str] = None
    
    # Dates
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    renewal_date: Optional[datetime] = None
    
    billing_address: Optional[BillingAddress] = None
    billing_metadata: Dict = Field(default_factory=dict)


class BillingUpdateSchema(BaseModel):
    plan: Optional[PlanType] = None
    billing_interval: Optional[BillingInterval] = None
    usage_credits: Optional[int] = Field(default=None, ge=0)
    total_credits: Optional[int] = Field(default=None, ge=0)
    payment_provider: Optional[PaymentProvider] = None
    subscription_status: Optional[SubscriptionStatus] = None
    amount: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = None
    
    # Provider-specific IDs
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_payment_method_id: Optional[str] = None
    stripe_price_id: Optional[str] = None
    
    razorpay_customer_id: Optional[str] = None
    razorpay_subscription_id: Optional[str] = None
    razorpay_payment_method_id: Optional[str] = None
    razorpay_plan_id: Optional[str] = None
    
    # Dates
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    renewal_date: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    cancel_at_period_end: Optional[bool] = None
    
    billing_address: Optional[BillingAddress] = None
    billing_metadata: Optional[Dict] = None
    failed_payment_count: Optional[int] = Field(default=None, ge=0)
    last_payment_error: Optional[str] = None


class TransactionCreateSchema(BaseModel):
    """Schema for adding a transaction to billing record"""
    transaction_id: str
    amount: float
    currency: str = "USD"
    status: str
    payment_method: Optional[str] = None
    provider: PaymentProvider
    provider_transaction_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


# ──────────────────────────────
# Billing Service
# ──────────────────────────────
class BillingService:

    @staticmethod
    async def create_billing(data: BillingCreateSchema) -> BillingInfo:
        """Create a new billing record for a user"""
        billing = BillingInfo(**data.dict())
        billing.created_at = datetime.utcnow()
        billing.updated_at = datetime.utcnow()
        await billing.insert()
        return billing

    @staticmethod
    async def get_billing(billing_id: str) -> Optional[BillingInfo]:
        """Get billing record by ID"""
        return await BillingInfo.get(PydanticObjectId(billing_id))

    @staticmethod
    async def get_billing_by_user(user_id: str) -> Optional[BillingInfo]:
        """Get billing record by user ID"""
        return await BillingInfo.find_one(BillingInfo.user_id == user_id)

    @staticmethod
    async def update_billing(billing_id: str, data: BillingUpdateSchema) -> Optional[BillingInfo]:
        """Update billing record"""
        billing = await BillingInfo.get(PydanticObjectId(billing_id))
        if not billing:
            return None
        update_data = data.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(billing, key, value)
        billing.updated_at = datetime.utcnow()
        await billing.save()
        return billing

    @staticmethod
    async def delete_billing(billing_id: str) -> bool:
        """Delete billing record"""
        billing = await BillingInfo.get(PydanticObjectId(billing_id))
        if not billing:
            return False
        await billing.delete()
        return True

    @staticmethod
    async def add_credits(user_id: str, credits: int) -> Optional[BillingInfo]:
        """Add credits to user's billing account"""
        billing = await BillingInfo.find_one(BillingInfo.user_id == user_id)
        if not billing:
            return None
        billing.usage_credits += credits
        billing.updated_at = datetime.utcnow()
        await billing.save()
        return billing

    @staticmethod
    async def deduct_credits(user_id: str, credits: int) -> Optional[BillingInfo]:
        """Deduct credits from user's billing account"""
        billing = await BillingInfo.find_one(BillingInfo.user_id == user_id)
        if not billing:
            return None
        billing.usage_credits = max(0, billing.usage_credits - credits)
        billing.updated_at = datetime.utcnow()
        await billing.save()
        return billing

    @staticmethod
    async def change_plan(
        user_id: str, 
        new_plan: PlanType, 
        renewal_date: Optional[datetime] = None,
        amount: Optional[float] = None,
        billing_interval: Optional[BillingInterval] = None
    ) -> Optional[BillingInfo]:
        """Change user's subscription plan"""
        billing = await BillingInfo.find_one(BillingInfo.user_id == user_id)
        if not billing:
            return None
        billing.plan = new_plan
        if renewal_date:
            billing.renewal_date = renewal_date
        if amount is not None:
            billing.amount = amount
        if billing_interval:
            billing.billing_interval = billing_interval
        billing.updated_at = datetime.utcnow()
        await billing.save()
        return billing

    @staticmethod
    async def cancel_subscription(user_id: str, immediate: bool = False) -> Optional[BillingInfo]:
        """Cancel user's subscription"""
        billing = await BillingInfo.find_one(BillingInfo.user_id == user_id)
        if not billing:
            return None
        
        if immediate:
            billing.subscription_status = SubscriptionStatus.CANCELED
            billing.canceled_at = datetime.utcnow()
        else:
            billing.cancel_at_period_end = True
        
        billing.updated_at = datetime.utcnow()
        await billing.save()
        return billing

    @staticmethod
    async def add_transaction(
        user_id: str, 
        transaction_data: TransactionCreateSchema
    ) -> Optional[BillingInfo]:
        """Add a transaction record to billing"""
        billing = await BillingInfo.find_one(BillingInfo.user_id == user_id)
        if not billing:
            return None
        
        transaction = TransactionRecord(**transaction_data.dict())
        billing.transactions.append(transaction)
        
        # Update failed payment count
        if transaction.status == "failed":
            billing.failed_payment_count += 1
            billing.last_payment_error = transaction.description or "Payment failed"
        elif transaction.status == "success":
            billing.failed_payment_count = 0
            billing.last_payment_error = None
        
        billing.updated_at = datetime.utcnow()
        await billing.save()
        return billing

    @staticmethod
    async def update_payment_provider_info(
        user_id: str,
        provider: PaymentProvider,
        customer_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        payment_method_id: Optional[str] = None,
        plan_id: Optional[str] = None
    ) -> Optional[BillingInfo]:
        """Update payment provider-specific information"""
        billing = await BillingInfo.find_one(BillingInfo.user_id == user_id)
        if not billing:
            return None
        
        billing.payment_provider = provider
        
        if provider == PaymentProvider.STRIPE:
            if customer_id:
                billing.stripe_customer_id = customer_id
            if subscription_id:
                billing.stripe_subscription_id = subscription_id
            if payment_method_id:
                billing.stripe_payment_method_id = payment_method_id
            if plan_id:
                billing.stripe_price_id = plan_id
                
        elif provider == PaymentProvider.RAZORPAY:
            if customer_id:
                billing.razorpay_customer_id = customer_id
            if subscription_id:
                billing.razorpay_subscription_id = subscription_id
            if payment_method_id:
                billing.razorpay_payment_method_id = payment_method_id
            if plan_id:
                billing.razorpay_plan_id = plan_id
        
        billing.updated_at = datetime.utcnow()
        await billing.save()
        return billing

    @staticmethod
    async def get_active_subscriptions(limit: int = 50, offset: int = 0) -> Tuple[List[BillingInfo], int]:
        """Get all active subscriptions"""
        query = BillingInfo.find(
            BillingInfo.subscription_status == SubscriptionStatus.ACTIVE
        ).sort("-created_at")
        total = await query.count()
        records = await query.skip(offset).limit(limit).to_list()
        return records, total

    @staticmethod
    async def get_expiring_subscriptions(days: int = 7) -> List[BillingInfo]:
        """Get subscriptions expiring in the next N days"""
        cutoff_date = datetime.utcnow() + timedelta(days=days)
        return await BillingInfo.find(
            BillingInfo.renewal_date <= cutoff_date,
            BillingInfo.subscription_status == SubscriptionStatus.ACTIVE
        ).to_list()

    @staticmethod
    async def list_billing(limit: int = 50, offset: int = 0) -> Tuple[List[BillingInfo], int]:
        """Return list of billing records with total count for pagination"""
        query = BillingInfo.find_all().sort("-created_at")
        total = await query.count()
        records = await query.skip(offset).limit(limit).to_list()
        return records, total
    
    @staticmethod
    async def get_billing_by_provider_customer(
        provider: PaymentProvider, 
        customer_id: str
    ) -> Optional[BillingInfo]:
        """Get billing record by payment provider customer ID"""
        if provider == PaymentProvider.STRIPE:
            return await BillingInfo.find_one(BillingInfo.stripe_customer_id == customer_id)
        elif provider == PaymentProvider.RAZORPAY:
            return await BillingInfo.find_one(BillingInfo.razorpay_customer_id == customer_id)
        return None
