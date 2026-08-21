"""
Razorpay RTO Risk-Ops & Profit Protection Engine
REST API Service: High-Throughput Developer & Checkout Integration Endpoints
"""

import os
import sys
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup sys.path to ensure src imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.risk_engine import RiskEngine
from src.explain import RiskExplainer

# Initialize FastAPI app
app = FastAPI(
    title="Razorpay RTO Risk-Ops & Profit Protection API",
    description="Automated Profit Arbitrage & Return-to-Origin (RTO) Mitigation API for D2C E-commerce checkouts.",
    version="2.0.0"
)

# Initialize engines
try:
    engine = RiskEngine()
    explainer = RiskExplainer(engine)
except Exception as e:
    print(f"Error loading Risk Scorer pipeline: {e}")
    engine = None
    explainer = None


# ----------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ----------------------------------------------------------------------

class OrderPayload(BaseModel):
    order_id: Optional[str] = Field(None, example="ORD_98241", description="Unique Merchant Order ID")
    pincode_tier: str = Field(..., example="Tier 3", description="Pincode classification tier (Tier 1, Tier 2, Tier 3)")
    pincode_rto_rate: float = Field(..., ge=0.0, le=1.0, example=0.25, description="Historical RTO rate of the target pincode")
    payment_mode: str = Field(..., example="COD", description="COD or Prepaid")
    order_value: float = Field(..., ge=0.0, example=2499.00, description="Total order monetary value in INR")
    discount_pct: float = Field(..., ge=0.0, le=100.0, example=15.0, description="Applied discount percentage")
    category: str = Field(..., example="Apparel", description="Product category (Apparel, Footwear, Beauty, Electronics, Home)")
    is_weekend_order: int = Field(..., ge=0, le=1, example=1, description="Whether order was checked out on a weekend (0 or 1)")
    address_length: int = Field(..., ge=0, example=65, description="String length of user delivery address")
    address_has_landmark: int = Field(..., ge=0, le=1, example=1, description="Whether delivery address specifies a local landmark (0 or 1)")
    pin_matches_city: int = Field(..., ge=0, le=1, example=1, description="Pincode matches selected city (0 or 1)")
    customer_tenure_days: int = Field(..., ge=0, example=120, description="Account age of customer in days")
    customer_past_orders: int = Field(..., ge=0, example=4, description="Count of historical orders completed by this customer")
    customer_past_rto_rate: float = Field(..., ge=0.0, le=1.0, example=0.25, description="Customer's historical RTO rate")
    
    # Optional merchant economics overrides
    gross_margin: Optional[float] = Field(0.40, ge=0.0, le=1.0, description="Product gross profit margin percentage")
    forward_shipping: Optional[float] = Field(70.0, ge=0.0, description="Forward logistics freight cost in INR")
    reverse_shipping: Optional[float] = Field(90.0, ge=0.0, description="Reverse return freight cost in INR")
    packaging_cost: Optional[float] = Field(40.0, ge=0.0, description="Packaging and deadweight handling cost in INR")


class ActionPayload(BaseModel):
    display_message: str = Field(..., description="Actionable customer-facing banner for checkout")
    suggested_discount_inr: float = Field(..., description="Instant cashback/discount offered to convert COD to Prepaid")
    allow_cod: bool = Field(..., description="Whether to display Cash on Delivery option at checkout")
    require_otp_verification: bool = Field(..., description="Whether to trigger automated WhatsApp/SMS OTP confirmation")


class RiskFactor(BaseModel):
    factor: str = Field(..., description="Plain-English explanation of the risk contributor")
    impact: str = Field(..., description="Percentage contribution (+/- probability change)")


class ProfitEvaluationResponse(BaseModel):
    order_id: str = Field(..., description="Merchant Order Identifier")
    risk_score: float = Field(..., description="Calibrated failure/return probability (0.0 to 1.0)")
    risk_tier: str = Field(..., description="Risk categorization: Low, Medium, Medium-High, High")
    expected_profit_cod: float = Field(..., description="Expected net merchant profit in INR if fulfilled via COD")
    expected_profit_prepaid: float = Field(..., description="Expected net merchant profit in INR if fulfilled via Prepaid")
    recommended_action: str = Field(..., description="Arbitrage action code: AUTO_SHIP, VERIFY_ADDRESS_OTP, INCENTIVIZE_PREPAID, STRICT_PREPAID_ONLY")
    action_payload: ActionPayload = Field(..., description="Dynamic checkout routing and intervention parameters")
    risk_factors: List[RiskFactor] = Field(..., description="SHAP feature attribution explaining the score")


# ----------------------------------------------------------------------
# API ENDPOINTS
# ----------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "active",
        "service": "Razorpay RTO Risk-Ops & Profit Protection Engine",
        "winning_model": engine.winner_name if engine else "None",
        "version": "2.0.0"
    }


@app.post("/v1/evaluate-order", response_model=ProfitEvaluationResponse)
def evaluate_order(payload: OrderPayload):
    """
    Main evaluation endpoint for dynamic checkout routing and profit maximization.
    """
    if engine is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Risk engine assets not loaded. Run train.py and evaluate.py on server host first."
        )

    try:
        order_dict = payload.model_dump()
        order_id = order_dict.get("order_id") or f"ORD_{uuid.uuid4().hex[:8].upper()}"

        # Inject customer_past_rto_count expected by internal model features
        order_dict["customer_past_rto_count"] = int(order_dict["customer_past_orders"] * order_dict["customer_past_rto_rate"])

        gross_margin = float(order_dict.get("gross_margin") or 0.40)
        forward_shipping = float(order_dict.get("forward_shipping") or 70.0)
        reverse_shipping = float(order_dict.get("reverse_shipping") or 90.0)
        packaging_cost = float(order_dict.get("packaging_cost") or 40.0)

        # 1. Run profit-arbitrage scoring
        verdict = engine.score_order(
            order=order_dict,
            gross_margin=gross_margin,
            forward_shipping=forward_shipping,
            reverse_shipping=reverse_shipping,
            packaging_cost=packaging_cost,
        )

        # 2. Extract SHAP explanations
        shap_factors = explainer.explain_prediction(order_dict)
        formatted_factors = []
        for factor in shap_factors:
            sign = "+" if factor["direction"] == "+" else "-"
            formatted_factors.append(
                RiskFactor(
                    factor=factor["factor"],
                    impact=f"{sign}{factor['impact']*100:.1f}%"
                )
            )

        action_data = verdict["action_payload"]
        action_payload_obj = ActionPayload(
            display_message=action_data["display_message"],
            suggested_discount_inr=float(action_data["suggested_discount_inr"]),
            allow_cod=bool(action_data["allow_cod"]),
            require_otp_verification=bool(action_data["require_otp_verification"])
        )

        return ProfitEvaluationResponse(
            order_id=order_id,
            risk_score=verdict["risk_probability"],
            risk_tier=verdict["risk_tier"],
            expected_profit_cod=verdict["expected_profit_cod"],
            expected_profit_prepaid=verdict["expected_profit_prepaid"],
            recommended_action=verdict["action_code"],
            action_payload=action_payload_obj,
            risk_factors=formatted_factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Order profit evaluation failed: {str(e)}")


@app.post("/v1/score", response_model=ProfitEvaluationResponse)
def legacy_score_alias(payload: OrderPayload):
    """Backwards compatibility alias for /v1/score."""
    return evaluate_order(payload)
