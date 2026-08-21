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
            if isinstance(le, dict):
                # Dict mapping (new format): map values directly, default unseen categories to -1
                df[col] = df[col].map(le).fillna(-1).astype(int)
            else:
                # LabelEncoder (fallback format):
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
                
        # Select features in the exact training order
        df = df[self.feature_columns]
        
        return df

    def apply_rule_overrides(self, order: dict, base_prob: float) -> tuple[float, str]:
        """
        Applies merchant fraud rule overrides on top of raw ML model probabilities.
        Returns: (final_probability, override_reason or None)
        """
        # Rule 1: High frequency repeat offender block
        past_orders = int(order.get("customer_past_orders", 0))
        past_rto_rate = float(order.get("customer_past_rto_rate", 0.0))
        if past_orders >= 3 and past_rto_rate >= 0.66:
            return 0.98, "High-Frequency Repeat Offender (RTO Rate >= 66% on 3+ past orders)"
            
        # Rule 2: High value mismatch flag
        order_val = float(order.get("order_value", 0.0))
        pin_match = int(order.get("pin_matches_city", 1))
        if pin_match == 0 and order_val >= 5000:
            return max(base_prob, 0.85), "High-Value Transaction Address Mismatch (Value >= ₹5,000 & Pincode/City Mismatch)"
            
        # Rule 3: COD Exploitation check (large COD order on fresh customer with short address)
        payment_mode = str(order.get("payment_mode", "Prepaid"))
        addr_len = int(order.get("address_length", 100))
        tenure = int(order.get("customer_tenure_days", 365))
        if payment_mode == "COD" and order_val >= 7500 and addr_len < 30 and tenure <= 7:
            return max(base_prob, 0.90), "COD Exploitation Risk (High-Value COD order on brand-new profile with incomplete address)"

        return base_prob, None

    def score_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Scores a batch of orders.
        Input: DataFrame with raw feature columns.
        Output: Copy of DataFrame with risk_probability, risk_tier, recommended_action columns.
        """
        # Preprocess features
        df_processed = self._preprocess_order_df(df_input)
        
        # Run predictions
        requires_scaling = self.artifacts.get("requires_scaling", False)
        if requires_scaling:
            df_scaled = self.scaler.transform(df_processed)
            probs = self.model.predict_proba(df_scaled)[:, 1]
        else:
            probs = self.model.predict_proba(df_processed)[:, 1]
            
        # Create output DataFrame copy
        df_output = df_input.copy()
        df_output["risk_probability"] = probs.round(4)
        
        # Apply rule overrides row-by-row
        final_probs = []
        override_reasons = []
        
        for idx, row in df_output.iterrows():
            row_dict = row.to_dict()
            final_p, reason = self.apply_rule_overrides(row_dict, row_dict["risk_probability"])
            final_probs.append(final_p)
            override_reasons.append(reason)
            
        df_output["risk_probability"] = final_probs
        df_output["rule_override_reason"] = override_reasons
        
        # Map probabilities to tiers and actions
        df_output["risk_tier"] = df_output["risk_probability"].apply(self._get_risk_tier)
        
        actions = []
        for idx, row in df_output.iterrows():
            act = self._get_recommended_action(row["risk_probability"])
            if row["rule_override_reason"]:
                act = f"{act} (Override: {row['rule_override_reason']})"
            actions.append(act)
        df_output["recommended_action"] = actions
        
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
        override_reason = scored_row["rule_override_reason"]
        
        # Extract risk factors (to be populated in detail in explain.py)
        # We define a placeholder here that gets enhanced by explain.py if needed
        risk_factors = []
        
        return {
            "risk_probability": prob,
            "risk_tier": self._get_risk_tier(prob),
            "recommended_action": scored_row["recommended_action"],
            "optimal_threshold": self.optimal_threshold,
            "override_reason": override_reason if pd.notna(override_reason) else None,
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
