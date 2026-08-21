import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup sys.path to ensure src imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.risk_engine import RiskEngine
    from src.explain import RiskExplainer
except ModuleNotFoundError:
    from risk_engine import RiskEngine
    from explain import RiskExplainer

# Initialize FastAPI app
app = FastAPI(
    title="Razorpay Return-Risk Scorer API",
    description="Developer-facing REST API for scoring Return-to-Origin (RTO) transaction risks in real-time.",
    version="1.0.0"
)

# Initialize engines
try:
    engine = RiskEngine()
    explainer = RiskExplainer(engine)
except Exception as e:
    print(f"Error loading Risk Scorer pipeline: {e}")
    engine = None
    explainer = None

# Define input order Pydantic schema
class OrderPayload(BaseModel):
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

# Define response schema
class ScoreResponse(BaseModel):
    risk_probability: float = Field(..., description="Calibrated probability of RTO transaction failure")
    risk_tier: str = Field(..., description="Assigned risk bucket: Low, Medium, High")
    recommended_action: str = Field(..., description="Action policy verdict based on optimal thresholds and fraud rules")
    rule_override_applied: bool = Field(..., description="True if a hardcoded fraud override rule was triggered")
    override_reason: str = Field(None, description="Reason for fraud rule override trigger, if any")
    optimal_decision_threshold: float = Field(..., description="Current threshold in use that minimizes cost")
    top_risk_factors: list = Field(..., description="SHAP feature attribution explaining the prediction score")

@app.get("/")
def read_root():
    return {
        "status": "active",
        "service": "Razorpay Return-Risk Scorer API",
        "model_loaded": engine.winner_name if engine else "None"
    }

@app.post("/v1/score", response_model=ScoreResponse)
def score_transaction(payload: OrderPayload):
    if engine is None or explainer is None:
        raise HTTPException(
            status_code=503, 
            detail="Risk Scoring model assets not loaded. Run train.py and evaluate.py on server host first."
        )
        
    try:
        # Convert Pydantic payload to dictionary
        order_dict = payload.model_dump()
        
        # Inject customer_past_rto_count expected by internal loaders
        order_dict["customer_past_rto_count"] = int(order_dict["customer_past_orders"] * order_dict["customer_past_rto_rate"])
        
        # 1. Run model prediction + override rules
        verdict = engine.score_order(order_dict)
        
        # 2. Get local SHAP factors
        shap_factors = explainer.explain_prediction(order_dict)
        
        # Simplify SHAP output schema for API consumers
        formatted_factors = []
        for factor in shap_factors:
            sign = "+" if factor["direction"] == "+" else "-"
            formatted_factors.append({
                "feature": factor["feature"],
                "description": factor["factor"],
                "impact": f"{sign}{factor['impact']*100:.1f}%"
            })
            
        return ScoreResponse(
            risk_probability=verdict["risk_probability"],
            risk_tier=verdict["risk_tier"],
            recommended_action=verdict["recommended_action"],
            rule_override_applied=verdict["override_reason"] is not None,
            override_reason=verdict["override_reason"],
            optimal_decision_threshold=verdict["optimal_threshold"],
            top_risk_factors=formatted_factors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk scoring evaluation failed: {str(e)}")
