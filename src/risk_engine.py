import os
import joblib
import pandas as pd
import numpy as np

# Define global path constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_artifacts.joblib")

class RiskEngine:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifacts not found at {model_path}. Please run train.py and evaluate.py first.")
            
        # Load artifacts saved during training and threshold tuning
        self.artifacts = joblib.load(model_path)
        self.winner_name = self.artifacts["winner_name"]
        self.model = self.artifacts["winning_model"]
        self.label_encoders = self.artifacts["label_encoders"]
        self.scaler = self.artifacts["scaler"]
        self.feature_columns = self.artifacts["feature_columns"]
        
        # Load optimal threshold, falling back to 0.5 if not yet tuned
        self.optimal_threshold = self.artifacts.get("optimal_threshold", 0.50)
        
    def _preprocess_order_df(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Helper to preprocess a DataFrame to match training columns and scale."""
        df = df_input.copy()
        
        # Add missing columns if any with default values (safeguard)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
                
        # Encode categorical columns
        for col, le in self.label_encoders.items():
            # Handle unseen categories by mapping to the first class or default
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
        # Select features in the exact training order
        df = df[self.feature_columns]
        
        return df

    def score_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Scores a batch of orders.
        Input: DataFrame with raw feature columns.
        Output: Copy of DataFrame with risk_probability, risk_tier, recommended_action columns.
        """
        # Preprocess features
        df_processed = self._preprocess_order_df(df_input)
        
        # Run predictions
        if self.winner_name == "Logistic Regression":
            df_scaled = self.scaler.transform(df_processed)
            probs = self.model.predict_proba(df_scaled)[:, 1]
        else:
            probs = self.model.predict_proba(df_processed)[:, 1]
            
        # Create output DataFrame copy
        df_output = df_input.copy()
        df_output["risk_probability"] = probs.round(4)
        
        # Map probabilities to tiers and actions
        df_output["risk_tier"] = df_output["risk_probability"].apply(self._get_risk_tier)
        df_output["recommended_action"] = df_output["risk_probability"].apply(self._get_recommended_action)
        
        return df_output

    def score_order(self, order: dict) -> dict:
        """
        Scores a single order dictionary.
        Input: Dictionary with order characteristics.
        Output: Dictionary with risk metadata.
        """
        # Convert dictionary to single-row DataFrame
        df_single = pd.DataFrame([order])
        df_scored = self.score_batch(df_single)
        scored_row = df_scored.iloc[0]
        
        prob = float(scored_row["risk_probability"])
        
        # Extract risk factors (to be populated in detail in explain.py)
        # We define a placeholder here that gets enhanced by explain.py if needed
        risk_factors = []
        
        return {
            "risk_probability": prob,
            "risk_tier": self._get_risk_tier(prob),
            "recommended_action": self._get_recommended_action(prob),
            "optimal_threshold": self.optimal_threshold,
            "top_risk_factors": risk_factors
        }
        
    def _get_risk_tier(self, probability: float) -> str:
        """Low (<0.30), Medium (0.30-0.60), High (>0.60)."""
        if probability < 0.30:
            return "Low"
        elif probability <= 0.60:
            return "Medium"
        else:
            return "High"
            
    def _get_recommended_action(self, probability: float) -> str:
        """
        Action policy mapping probabilities relative to the OPTIMAL cost-sensitive threshold.
        """
        # Decisions are made relative to the optimal tuned threshold!
        if probability < (self.optimal_threshold - 0.10):
            return "Auto-Ship (Pre-approved)"
        elif probability <= self.optimal_threshold:
            return "SMS/IVR Verification (Medium risk boundary)"
        else:
            return "Hold & Require Prepaid (High RTO risk flagged)"

if __name__ == "__main__":
    # Self-test to verify RiskEngine initializes and scores correctly
    try:
        engine = RiskEngine()
        print("RiskEngine loaded successfully!")
        
        test_order = {
            "pincode_tier": "Tier 3",
            "pincode_rto_rate": 0.45,
            "payment_mode": "COD",
            "order_value": 4500,
            "discount_pct": 65.0,
            "category": "Apparel",
            "is_weekend_order": 1,
            "address_length": 15,
            "address_has_landmark": 0,
            "pin_matches_city": 1,
            "customer_tenure_days": 10,
            "customer_past_orders": 0,
            "customer_past_rto_count": 0,
            "customer_past_rto_rate": 0.0
        }
        
        res = engine.score_order(test_order)
        print(f"\nSample Order Scoring Results:")
        print(f" - Risk Probability: {res['risk_probability']:.4f}")
        print(f" - Risk Tier:        {res['risk_tier']}")
        print(f" - Recommended:      {res['recommended_action']}")
        print(f" - Threshold Used:   {res['optimal_threshold']:.2f}")
    except Exception as e:
        print(f"Error during self-test: {e}")
        print("Make sure train.py and evaluate.py are executed first.")
