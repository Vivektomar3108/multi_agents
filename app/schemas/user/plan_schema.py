# app/schemas/user/plan_schema.py
from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum
from pymongo import IndexModel, ASCENDING


class PlanTier(str, Enum):
    """Plan tiers"""
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class PlanInterval(str, Enum):
    """Billing intervals for plans"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class PlanFeature(BaseModel):
    """Individual feature in a plan"""
    name: str = Field(..., description="Feature name")
    description: Optional[str] = Field(None, description="Feature description")
    enabled: bool = Field(default=True, description="Whether feature is enabled")
    limit: Optional[int] = Field(None, description="Limit if applicable (e.g., max users, max storage)")
    metadata: Dict = Field(default_factory=dict, description="Additional feature metadata")


class PlanPricing(BaseModel):
    """Pricing information for different intervals"""
    monthly_price: float = Field(default=0.0, ge=0, description="Monthly price")
    quarterly_price: float = Field(default=0.0, ge=0, description="Quarterly price (if applicable)")
    yearly_price: float = Field(default=0.0, ge=0, description="Yearly price (if applicable)")
    currency: str = Field(default="INR", description="Currency code (INR, USD, EUR, etc.)")
    
    # Discount information
    yearly_discount_percent: float = Field(default=0.0, ge=0, le=100, description="Yearly discount percentage")
    quarterly_discount_percent: float = Field(default=0.0, ge=0, le=100, description="Quarterly discount percentage")


class TokenAllocation(BaseModel):
    """Token allocation details for a plan"""
    initial_tokens: int = Field(..., ge=0, description="Initial tokens on signup/upgrade")
    monthly_tokens: int = Field(..., ge=0, description="Tokens allocated per month")
    rollover_enabled: bool = Field(default=False, description="Whether unused tokens roll over to next month")
    max_rollover_tokens: Optional[int] = Field(None, description="Maximum tokens that can be rolled over")
    token_purchase_allowed: bool = Field(default=True, description="Whether user can buy additional tokens")
    token_purchase_rate: float = Field(default=0.0, description="Price per additional token")


class Plan(Document):
    """
    Subscription Plan schema storing all plan details including pricing,
    features, token allocation, and limits.
    """
    
    # Core fields
    id: Optional[PydanticObjectId] = Field(default=None, alias="_id")
    plan_id: str = Field(..., description="Unique plan identifier (e.g., 'basic-monthly')")
    
    # Plan details
    name: str = Field(..., description="Plan display name (e.g., 'Basic Plan')")
    tier: PlanTier = Field(..., description="Plan tier")
    interval: PlanInterval = Field(default=PlanInterval.MONTHLY, description="Billing interval")
    
    description: Optional[str] = Field(None, description="Plan description")
    tagline: Optional[str] = Field(None, description="Marketing tagline for the plan")
    
    # Pricing
    pricing: PlanPricing = Field(..., description="Pricing information")
    
    # Token allocation
    token_allocation: TokenAllocation = Field(..., description="Token allocation details")
    
    # Features
    features: List[PlanFeature] = Field(default_factory=list, description="List of plan features")
    
    # Limits
    max_users: Optional[int] = Field(None, description="Maximum users/seats (for team plans)")
    max_projects: Optional[int] = Field(None, description="Maximum projects allowed")
    max_storage_gb: Optional[int] = Field(None, description="Maximum storage in GB")
    api_rate_limit: Optional[int] = Field(None, description="API calls per minute")
    
    # Agent limits
    agent_limits: Dict[str, int] = Field(
        default_factory=dict,
        description="Specific limits for agents (e.g., {'email': 100, 'websearch': 50})"
    )
    
    # Payment provider IDs
    stripe_price_id: Optional[str] = Field(None, description="Stripe price ID for this plan")
    razorpay_plan_id: Optional[str] = Field(None, description="Razorpay plan ID")
    
    # Status and visibility
    is_active: bool = Field(default=True, description="Whether plan is active and available")
    is_featured: bool = Field(default=False, description="Whether to feature this plan")
    is_popular: bool = Field(default=False, description="Mark as 'Most Popular'")
    display_order: int = Field(default=0, description="Order in which to display plans")
    
    # Trial
    trial_days: int = Field(default=0, ge=0, description="Number of trial days")
    trial_tokens: int = Field(default=0, ge=0, description="Tokens during trial period")
    
    # Metadata
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional plan metadata (custom fields, settings, etc.)"
    )
    
    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Plan creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Plan update timestamp")
    created_by: Optional[str] = Field(None, description="Admin user who created the plan")

    class Settings:
        name = "plans"
        indexes = [
            IndexModel([("plan_id", ASCENDING)], unique=True),
            IndexModel([("tier", ASCENDING)]),
            IndexModel([("interval", ASCENDING)]),
            IndexModel([("is_active", ASCENDING)]),
            IndexModel([("display_order", ASCENDING)]),
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
    
    def get_price_for_interval(self, interval: str) -> float:
        """Get price for specific interval"""
        interval_map = {
            "monthly": self.pricing.monthly_price,
            "quarterly": self.pricing.quarterly_price,
            "yearly": self.pricing.yearly_price,
        }
        return interval_map.get(interval.lower(), self.pricing.monthly_price)
    
    def get_effective_monthly_price(self) -> float:
        """Calculate effective monthly price based on interval"""
        if self.interval == PlanInterval.MONTHLY:
            return self.pricing.monthly_price
        elif self.interval == PlanInterval.QUARTERLY:
            return self.pricing.quarterly_price / 3
        elif self.interval == PlanInterval.YEARLY:
            return self.pricing.yearly_price / 12
        return self.pricing.monthly_price


# ──────────────────────────────
# Pydantic Schemas for API
# ──────────────────────────────

class PlanCreateSchema(BaseModel):
    """Schema for creating a new plan"""
    plan_id: str
    name: str
    tier: PlanTier
    interval: PlanInterval = PlanInterval.MONTHLY
    description: Optional[str] = None
    tagline: Optional[str] = None
    
    pricing: PlanPricing
    token_allocation: TokenAllocation
    features: List[PlanFeature] = Field(default_factory=list)
    
    max_users: Optional[int] = None
    max_projects: Optional[int] = None
    max_storage_gb: Optional[int] = None
    api_rate_limit: Optional[int] = None
    agent_limits: Dict[str, int] = Field(default_factory=dict)
    
    stripe_price_id: Optional[str] = None
    razorpay_plan_id: Optional[str] = None
    
    is_active: bool = True
    is_featured: bool = False
    is_popular: bool = False
    display_order: int = 0
    
    trial_days: int = 0
    trial_tokens: int = 0
    
    metadata: Dict = Field(default_factory=dict)


class PlanUpdateSchema(BaseModel):
    """Schema for updating an existing plan"""
    name: Optional[str] = None
    tier: Optional[PlanTier] = None
    interval: Optional[PlanInterval] = None
    description: Optional[str] = None
    tagline: Optional[str] = None
    
    pricing: Optional[PlanPricing] = None
    token_allocation: Optional[TokenAllocation] = None
    features: Optional[List[PlanFeature]] = None
    
    max_users: Optional[int] = None
    max_projects: Optional[int] = None
    max_storage_gb: Optional[int] = None
    api_rate_limit: Optional[int] = None
    agent_limits: Optional[Dict[str, int]] = None
    
    stripe_price_id: Optional[str] = None
    razorpay_plan_id: Optional[str] = None
    
    is_active: Optional[bool] = None
    is_featured: Optional[bool] = None
    is_popular: Optional[bool] = None
    display_order: Optional[int] = None
    
    trial_days: Optional[int] = None
    trial_tokens: Optional[int] = None
    
    metadata: Optional[Dict] = None


class PlanResponse(BaseModel):
    """Response schema for plan details (public facing)"""
    plan_id: str
    name: str
    tier: str
    interval: str
    description: Optional[str]
    tagline: Optional[str]
    
    # Pricing
    price: float  # Price for current interval
    currency: str
    effective_monthly_price: float  # For comparison
    
    # Tokens
    initial_tokens: int
    monthly_tokens: int
    rollover_enabled: bool
    
    # Features
    features: List[PlanFeature]
    
    # Limits
    max_users: Optional[int]
    max_projects: Optional[int]
    max_storage_gb: Optional[int]
    
    # Display flags
    is_featured: bool
    is_popular: bool
    
    # Trial
    trial_days: int
    trial_tokens: int
