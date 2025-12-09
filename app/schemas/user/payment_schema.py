# app/schemas/user/payment_schema.py
"""
Schemas for payment operations (Razorpay order creation and verification)
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from enum import Enum


class Currency(str, Enum):
    """Supported currencies"""
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class PaymentStatus(str, Enum):
    """Payment status"""
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    REFUNDED = "refunded"
    FAILED = "failed"


class CreatePaymentRequest(BaseModel):
    """Request schema for creating a payment order"""
    amount: float = Field(..., gt=0, description="Payment amount (will be converted to smallest currency unit)")
    currency: Currency = Field(default=Currency.INR, description="Currency code")
    receipt: Optional[str] = Field(None, description="Unique receipt ID for your reference")
    notes: Optional[Dict[str, str]] = Field(default_factory=dict, description="Key-value pairs for additional info")
    
    class Config:
        json_schema_extra = {
            "example": {
                "amount": 499.00,
                "currency": "INR",
                "receipt": "receipt_order_123",
                "notes": {
                    "user_id": "user_12345",
                    "plan": "professional"
                }
            }
        }


class CreatePaymentResponse(BaseModel):
    """Response schema after creating a payment order"""
    order_id: str = Field(..., description="Razorpay order ID")
    amount: int = Field(..., description="Amount in smallest currency unit (e.g., paise for INR)")
    currency: str = Field(..., description="Currency code")
    receipt: Optional[str] = Field(None, description="Receipt ID")
    status: str = Field(..., description="Order status")
    created_at: int = Field(..., description="Unix timestamp of order creation")
    
    # Additional fields for frontend
    razorpay_key: str = Field(..., description="Razorpay Key ID for checkout")
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "order_MNop1234567890",
                "amount": 49900,
                "currency": "INR",
                "receipt": "receipt_order_123",
                "status": "created",
                "created_at": 1699000000,
                "razorpay_key": "rzp_test_xxxxxxxxxxxxx"
            }
        }


class VerifyPaymentRequest(BaseModel):
    """Request schema for verifying payment after successful payment"""
    razorpay_order_id: str = Field(..., description="Order ID from Razorpay")
    razorpay_payment_id: str = Field(..., description="Payment ID from Razorpay")
    razorpay_signature: str = Field(..., description="Signature from Razorpay for verification")
    
    class Config:
        json_schema_extra = {
            "example": {
                "razorpay_order_id": "order_MNop1234567890",
                "razorpay_payment_id": "pay_MNop9876543210",
                "razorpay_signature": "abc123def456..."
            }
        }


class VerifyPaymentResponse(BaseModel):
    """Response schema after payment verification"""
    success: bool = Field(..., description="Whether payment verification succeeded")
    message: str = Field(..., description="Verification status message")
    order_id: str = Field(..., description="Order ID")
    payment_id: str = Field(..., description="Payment ID")
    amount: Optional[float] = Field(None, description="Payment amount")
    currency: Optional[str] = Field(None, description="Currency")
    status: Optional[str] = Field(None, description="Payment status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Payment verified successfully",
                "order_id": "order_MNop1234567890",
                "payment_id": "pay_MNop9876543210",
                "amount": 499.00,
                "currency": "INR",
                "status": "captured"
            }
        }


class PaymentErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(default=False, description="Always False for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "SIGNATURE_VERIFICATION_FAILED",
                "message": "Payment signature verification failed"
            }
        }
