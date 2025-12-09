# app/schemas/user/billing_schema.py
from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Literal
from datetime import datetime
from enum import Enum
from pymongo import IndexModel, ASCENDING


class PaymentProvider(str, Enum):
    """Supported payment providers"""
    STRIPE = "stripe"
    RAZORPAY = "razorpay"
    MANUAL = "manual"  # For admin-managed billing


class PlanType(str, Enum):
    """Subscription plan types"""
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription lifecycle states"""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    PAUSED = "paused"


class BillingInterval(str, Enum):
    """Billing cycle intervals"""
    MONTHLY = "monthly"
    YEARLY = "yearly"
    QUARTERLY = "quarterly"


class TransactionRecord(BaseModel):
    """Individual transaction/payment record"""
    transaction_id: str = Field(..., description="Unique transaction ID from payment provider")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(default="INR", description="Currency code (USD, INR, etc.)")
    status: str = Field(..., description="Payment status: success/failed/pending/refunded")
    payment_method: Optional[str] = Field(None, description="Payment method used")
    provider: PaymentProvider = Field(..., description="Payment provider")
    provider_transaction_id: Optional[str] = Field(None, description="Provider's transaction reference")
    description: Optional[str] = Field(None, description="Transaction description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)


class BillingAddress(BaseModel):
    """Customer billing address"""
    line1: Optional[str] = None
    line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class BillingInfo(Document):
    """
    Unified billing model supporting multiple payment providers (Stripe, Razorpay).
    Stores subscription, payment, and usage information.
    """
    
    # Core fields
    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")
    user_id: str = Field(..., description="Reference to the user")
    
    # Plan & Subscription
    plan: PlanType = Field(default=PlanType.FREE, description="Current subscription plan")
    billing_interval: BillingInterval = Field(default=BillingInterval.MONTHLY, description="Billing cycle")
    subscription_status: SubscriptionStatus = Field(
        default=SubscriptionStatus.ACTIVE, 
        description="Current subscription status"
    )
    
    # Usage & Credits
    usage_credits: int = Field(default=0, ge=0, description="Remaining usage credits")
    total_credits: int = Field(default=0, ge=0, description="Total credits allocated for current period")
    
    # Payment Provider Information
    payment_provider: PaymentProvider = Field(
        default=PaymentProvider.MANUAL, 
        description="Active payment provider"
    )
    
    # Stripe-specific fields
    stripe_customer_id: Optional[str] = Field(None, description="Stripe customer ID")
    stripe_subscription_id: Optional[str] = Field(None, description="Stripe subscription ID")
    stripe_payment_method_id: Optional[str] = Field(None, description="Stripe payment method ID")
    stripe_price_id: Optional[str] = Field(None, description="Stripe price ID for current plan")
    
    # Razorpay-specific fields
    razorpay_customer_id: Optional[str] = Field(None, description="Razorpay customer ID")
    razorpay_subscription_id: Optional[str] = Field(None, description="Razorpay subscription ID")
    razorpay_payment_method_id: Optional[str] = Field(None, description="Razorpay payment method ID")
    razorpay_plan_id: Optional[str] = Field(None, description="Razorpay plan ID")
    
    # Pricing & Currency
    amount: float = Field(default=0.0, ge=0, description="Subscription amount per billing cycle")
    currency: str = Field(default="USD", description="Currency code (USD, INR, EUR, etc.)")
    
    # Dates & Periods
    trial_start: Optional[datetime] = Field(None, description="Trial period start date")
    trial_end: Optional[datetime] = Field(None, description="Trial period end date")
    current_period_start: Optional[datetime] = Field(None, description="Current billing period start")
    current_period_end: Optional[datetime] = Field(None, description="Current billing period end")
    renewal_date: Optional[datetime] = Field(None, description="Next renewal/billing date")
    canceled_at: Optional[datetime] = Field(None, description="Subscription cancellation timestamp")
    cancel_at_period_end: bool = Field(
        default=False, 
        description="Whether to cancel subscription at period end"
    )
    
    # Billing Address
    billing_address: Optional[BillingAddress] = Field(None, description="Customer billing address")
    
    # Payment History
    transactions: List[TransactionRecord] = Field(
        default_factory=list, 
        description="Transaction/payment history"
    )
    
    # Metadata & Extras
    billing_metadata: Dict = Field(
        default_factory=dict, 
        description="Custom metadata for billing (webhooks, notes, etc.)"
    )
    
    # Failure Tracking
    failed_payment_count: int = Field(default=0, ge=0, description="Count of consecutive failed payments")
    last_payment_error: Optional[str] = Field(None, description="Last payment failure reason")
    
    # Audit Fields
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Record update timestamp")

    class Settings:
        name = "billing"
        indexes = [
            IndexModel([("user_id", ASCENDING)], unique=True),
            IndexModel([("stripe_customer_id", ASCENDING)]),
            IndexModel([("razorpay_customer_id", ASCENDING)]),
            IndexModel([("subscription_status", ASCENDING)]),
            IndexModel([("plan", ASCENDING)]),
            IndexModel([("renewal_date", ASCENDING)]),
        ]

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            PydanticObjectId: str,
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        return datetime.utcnow()
