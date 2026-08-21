"""
Razorpay RTO Risk-Ops & Profit Protection Engine
Core Module: Decision Intelligence & Margin-Aware Profit Arbitrage
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Define global path constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_artifacts.joblib")


class RiskEngine:
    """
    Margin-Aware Risk & Profit Arbitrage Engine.
    
    Transforms crude binary RTO risk scoring into dynamic profit-maximizing
    checkout interventions that protect gross margins, eliminate logistics waste,
    and maximize merchant GMV.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifacts not found at {model_path}. "
                "Please run train.py and evaluate.py first."
            )

        # Load artifacts saved during training and threshold tuning
        self.artifacts = joblib.load(model_path)
        self.winner_name: str = self.artifacts["winner_name"]
        self.model = self.artifacts["winning_model"]
        self.label_encoders: Dict[str, Any] = self.artifacts["label_encoders"]
        self.scaler = self.artifacts["scaler"]
        self.feature_columns: List[str] = self.artifacts["feature_columns"]

        # Whether the winning model requires StandardScaler input
        self.requires_scaling: bool = self.artifacts.get("requires_scaling", False)

        # Load optimal threshold, falling back to 0.20 if not yet tuned
        self.optimal_threshold: float = float(self.artifacts.get("optimal_threshold", 0.20))

    # ------------------------------------------------------------------
    # PREPROCESSING & UNSEEN CATEGORY DEFENSE
    # ------------------------------------------------------------------

    def _preprocess_order_df(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a raw order DataFrame into a model-ready feature matrix.
        Defensively handles missing features, unknown categories, and type coercion.
        """
        df = df_input.copy()

        # Add missing columns with a safe default of 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Encode categorical columns safely
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                df[col] = -1
                continue

            if isinstance(le, dict):
                # Dict mapping format: unknown/null mapped safely to -1
                df[col] = df[col].map(le).fillna(-1).astype(int)
            else:
                # Legacy LabelEncoder fallback
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])

        # Coerce numeric columns to float/int to prevent string leakage
        for col in self.feature_columns:
            if col not in self.label_encoders:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Select and order features exactly as during training
        df = df[self.feature_columns]
        return df

    # ------------------------------------------------------------------
    # FINANCIAL ECONOMICS & EXPECTED PROFIT FORMULAS
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_expected_profit(
        order_value: float,
        rto_prob: float,
        gross_margin: float = 0.40,
        forward_shipping: float = 70.0,
        reverse_shipping: float = 90.0,
        packaging_cost: float = 40.0,
        is_prepaid: bool = False,
        discount_inr: float = 0.0,
    ) -> float:
        """
        Calculates the expected net profit for a transaction under COD vs Prepaid.

        Formula:
          Expected Net Profit =
            (1 - P(RTO)) * ((Order Value - Discount) * Margin - Forward Shipping)
            - P(RTO) * (Forward Shipping + Reverse Shipping + Packaging Loss)

        Prepaid transactions carry a minimal baseline return rate (~3%) as buyers are
        financially committed.
        """
        effective_rto_prob = 0.03 if is_prepaid else max(0.0, min(1.0, rto_prob))
        effective_order_val = max(0.0, order_value - discount_inr)

        success_profit = (effective_order_val * gross_margin) - forward_shipping
        failure_loss = forward_shipping + reverse_shipping + packaging_cost

        expected_profit = ((1.0 - effective_rto_prob) * success_profit) - (effective_rto_prob * failure_loss)
        return round(expected_profit, 2)

    # ------------------------------------------------------------------
    # FRAUD & HEURISTIC OVERRIDES
    # ------------------------------------------------------------------

    def _apply_rule_overrides_row(
        self, row: pd.Series, base_prob: float
    ) -> Tuple[float, Optional[str]]:
        """
        Applies high-confidence merchant fraud safeguards.
        Returns: (final_probability, override_reason)
        """
        past_orders = int(row.get("customer_past_orders", 0))
        past_rto_rate = float(row.get("customer_past_rto_rate", 0.0))
        order_val = float(row.get("order_value", 0.0))
        pin_match = int(row.get("pin_matches_city", 1))
        payment_mode = str(row.get("payment_mode", "Prepaid"))
        addr_len = int(row.get("address_length", 100))
        tenure = int(row.get("customer_tenure_days", 365))

        # Rule 1: High-frequency repeat offender
        if past_orders >= 3 and past_rto_rate >= 0.66:
            return (
                0.98,
                "High-Frequency Repeat Offender (RTO Rate >= 66% on 3+ past orders)",
            )

        # Rule 2: High-value address mismatch
        if pin_match == 0 and order_val >= 5000:
            return (
                max(base_prob, 0.85),
                "High-Value Address Mismatch (Value >= ₹5,000 & Pincode/City Mismatch)",
            )

        # Rule 3: COD exploitation risk on new unverified profile
        if (
            payment_mode == "COD"
            and order_val >= 7500
            and addr_len < 30
            and tenure <= 7
        ):
            return (
                max(base_prob, 0.90),
                "COD Exploitation Risk (High-Value COD order on new profile with incomplete address)",
            )

        return base_prob, None

    def apply_rule_overrides(
        self, order: dict, base_prob: float
    ) -> Tuple[float, Optional[str]]:
        return self._apply_rule_overrides_row(pd.Series(order), base_prob)

    # ------------------------------------------------------------------
    # DYNAMIC PROFIT-ARBITRAGE INTERVENTIONS
    # ------------------------------------------------------------------

    def determine_profit_intervention(
        self,
        prob: float,
        order_value: float,
        gross_margin: float = 0.40,
        forward_shipping: float = 70.0,
        reverse_shipping: float = 90.0,
        packaging_cost: float = 40.0,
        override_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Determines the optimal checkout action path to maximize merchant profit:
        1. GREEN  (AUTO_SHIP)            -> High expected profit, instant 1-click COD.
        2. YELLOW (VERIFY_ADDRESS_OTP)   -> Borderline risk, OTP/address correction.
        3. ORANGE (INCENTIVIZE_PREPAID)  -> Risky COD, offer instant ₹50 discount for prepaid.
        4. RED    (STRICT_PREPAID_ONLY)  -> High risk / fraud, COD disabled.
        """
        # Calculate expected profit margins
        exp_profit_cod = self.calculate_expected_profit(
            order_value=order_value,
            rto_prob=prob,
            gross_margin=gross_margin,
            forward_shipping=forward_shipping,
            reverse_shipping=reverse_shipping,
            packaging_cost=packaging_cost,
            is_prepaid=False,
        )

        exp_profit_prepaid = self.calculate_expected_profit(
            order_value=order_value,
            rto_prob=prob,
            gross_margin=gross_margin,
            forward_shipping=forward_shipping,
            reverse_shipping=reverse_shipping,
            packaging_cost=packaging_cost,
            is_prepaid=True,
            discount_inr=0.0,
        )

        exp_profit_prepaid_discount = self.calculate_expected_profit(
            order_value=order_value,
            rto_prob=prob,
            gross_margin=gross_margin,
            forward_shipping=forward_shipping,
            reverse_shipping=reverse_shipping,
            packaging_cost=packaging_cost,
            is_prepaid=True,
            discount_inr=50.0,
        )

        # 1. RED PATH: Severe risk or hard fraud override
        if override_reason is not None or prob >= 0.65 or (exp_profit_cod < -50.0 and exp_profit_prepaid > 0):
            action_code = "STRICT_PREPAID_ONLY"
            recommended_action = "COD Block / Strict Prepaid Only"
            action_payload = {
                "display_message": "COD disabled due to high return risk profile. Order eligible strictly via prepaid checkout.",
                "suggested_discount_inr": 0,
                "allow_cod": False,
                "require_otp_verification": True,
                "action_color": "#F87171",
                "badge": "RED"
            }
            risk_tier = "High"

        # 2. ORANGE PATH: COD unprofitable or risky, but Prepaid conversion yields solid profit
        elif prob >= 0.35 or exp_profit_cod <= 20.0:
            action_code = "INCENTIVIZE_PREPAID"
            recommended_action = "Prepaid Conversion Incentive (₹50 UPI Discount)"
            action_payload = {
                "display_message": "Pay via UPI/Card now & get flat ₹50 instant discount + priority dispatch.",
                "suggested_discount_inr": 50,
                "allow_cod": False,
                "require_otp_verification": True,
                "action_color": "#FB923C",
                "badge": "ORANGE"
            }
            risk_tier = "Medium-High"

        # 3. YELLOW PATH: Borderline risk, recoverable via automated verification
        elif prob >= self.optimal_threshold:
            action_code = "VERIFY_ADDRESS_OTP"
            recommended_action = "Automated Verification & Address Fix"
            action_payload = {
                "display_message": "Trigger automated WhatsApp/SMS OTP confirmation and address validation to reduce undelivered returns.",
                "suggested_discount_inr": 0,
                "allow_cod": True,
                "require_otp_verification": True,
                "action_color": "#FBBF24",
                "badge": "YELLOW"
            }
            risk_tier = "Medium"

        # 4. GREEN PATH: Safe, highly profitable order
        else:
            action_code = "AUTO_SHIP"
            recommended_action = "Auto-Ship (Pre-Approved COD)"
            action_payload = {
                "display_message": "Order pre-approved for instant 1-click COD dispatch.",
                "suggested_discount_inr": 0,
                "allow_cod": True,
                "require_otp_verification": False,
                "action_color": "#34D399",
                "badge": "GREEN"
            }
            risk_tier = "Low"

        return {
            "action_code": action_code,
            "recommended_action": recommended_action,
            "action_payload": action_payload,
            "risk_tier": risk_tier,
            "expected_profit_cod": exp_profit_cod,
            "expected_profit_prepaid": exp_profit_prepaid,
            "expected_profit_prepaid_discount": exp_profit_prepaid_discount,
        }

    # ------------------------------------------------------------------
    # SINGLE & BATCH SCORING
    # ------------------------------------------------------------------

    def score_order(
        self,
        order: dict,
        gross_margin: float = 0.40,
        forward_shipping: float = 70.0,
        reverse_shipping: float = 90.0,
        packaging_cost: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Scores a single order dictionary and returns complete margin and risk metadata.
        """
        df_single = pd.DataFrame([order])
        df_scored = self.score_batch(
            df_single,
            gross_margin=gross_margin,
            forward_shipping=forward_shipping,
            reverse_shipping=reverse_shipping,
            packaging_cost=packaging_cost,
        )
        row = df_scored.iloc[0]

        prob = float(row["risk_probability"])
        raw_reason = row.get("rule_override_reason", None)

        try:
            override_reason = str(raw_reason) if pd.notna(raw_reason) else None
        except (TypeError, ValueError):
            override_reason = None

        if override_reason is not None and override_reason.strip() in ("", "nan", "None"):
            override_reason = None

        intervention = self.determine_profit_intervention(
            prob=prob,
            order_value=float(order.get("order_value", 1500)),
            gross_margin=gross_margin,
            forward_shipping=forward_shipping,
            reverse_shipping=reverse_shipping,
            packaging_cost=packaging_cost,
            override_reason=override_reason,
        )

        return {
            "risk_probability": prob,
            "risk_tier": intervention["risk_tier"],
            "expected_profit_cod": intervention["expected_profit_cod"],
            "expected_profit_prepaid": intervention["expected_profit_prepaid"],
            "expected_profit_prepaid_discount": intervention["expected_profit_prepaid_discount"],
            "action_code": intervention["action_code"],
            "recommended_action": intervention["recommended_action"],
            "action_payload": intervention["action_payload"],
            "optimal_threshold": self.optimal_threshold,
            "override_reason": override_reason,
            "top_risk_factors": [],
        }

    def score_batch(
        self,
        df_input: pd.DataFrame,
        gross_margin: float = 0.40,
        forward_shipping: float = 70.0,
        reverse_shipping: float = 90.0,
        packaging_cost: float = 40.0,
    ) -> pd.DataFrame:
        """
        Scores an entire batch of orders with single-pass vectorized execution.
        """
        df_processed = self._preprocess_order_df(df_input)

        # ML Model Inference
        if self.requires_scaling:
            X_eval = self.scaler.transform(df_processed)
        else:
            X_eval = df_processed.values

        base_probs = self.model.predict_proba(X_eval)[:, 1]

        df_output = df_input.copy()
        df_output["_base_prob"] = base_probs

        final_probs = []
        override_reasons = []
        action_codes = []
        recommended_actions = []
        risk_tiers = []
        exp_profits_cod = []
        exp_profits_prepaid = []
        action_colors = []
        display_messages = []

        for _, row in df_output.iterrows():
            row_dict = row.to_dict()
            base_p = float(row["_base_prob"])
            prob, reason = self._apply_rule_overrides_row(row, base_p)
            prob_rounded = round(prob, 4)

            order_val = float(row.get("order_value", 1500))

            inter = self.determine_profit_intervention(
                prob=prob_rounded,
                order_value=order_val,
                gross_margin=gross_margin,
                forward_shipping=forward_shipping,
                reverse_shipping=reverse_shipping,
                packaging_cost=packaging_cost,
                override_reason=reason,
            )

            final_probs.append(prob_rounded)
            override_reasons.append(reason)
            action_codes.append(inter["action_code"])
            recommended_actions.append(inter["recommended_action"])
            risk_tiers.append(inter["risk_tier"])
            exp_profits_cod.append(inter["expected_profit_cod"])
            exp_profits_prepaid.append(inter["expected_profit_prepaid"])
            action_colors.append(inter["action_payload"]["action_color"])
            display_messages.append(inter["action_payload"]["display_message"])

        df_output.drop(columns=["_base_prob"], inplace=True)

        df_output["risk_probability"] = final_probs
        df_output["risk_tier"] = risk_tiers
        df_output["expected_profit_cod"] = exp_profits_cod
        df_output["expected_profit_prepaid"] = exp_profits_prepaid
        df_output["action_code"] = action_codes
        df_output["recommended_action"] = recommended_actions
        df_output["checkout_display_message"] = display_messages
        df_output["action_color"] = action_colors
        df_output["rule_override_reason"] = override_reasons

        return df_output


# ----------------------------------------------------------------------
# SELF TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        engine = RiskEngine()
        print("RiskEngine loaded successfully!")
        print(f"  Winner Model : {engine.winner_name}")
        print(f"  Optimal Threshold : {engine.optimal_threshold:.2f}")

        sample_order = {
            "pincode_tier": "Tier 3",
            "pincode_rto_rate": 0.35,
            "payment_mode": "COD",
            "order_value": 3200,
            "discount_pct": 25.0,
            "category": "Apparel",
            "is_weekend_order": 1,
            "address_length": 25,
            "address_has_landmark": 0,
            "pin_matches_city": 1,
            "customer_tenure_days": 15,
            "customer_past_orders": 1,
            "customer_past_rto_count": 0,
            "customer_past_rto_rate": 0.0,
        }

        res = engine.score_order(sample_order)
        print("\nProfit-Maximized Scoring Result:")
        print(f"  Risk Probability : {res['risk_probability']*100:.1f}% ({res['risk_tier']})")
        print(f"  Expected COD Profit : ₹{res['expected_profit_cod']}")
        print(f"  Expected Prepaid Profit : ₹{res['expected_profit_prepaid']}")
        print(f"  Action Code : {res['action_code']}")
        print(f"  Intervention : {res['recommended_action']}")
        print(f"  Display Msg : {res['action_payload']['display_message']}")
        print(f"  Suggested Discount : ₹{res['action_payload']['suggested_discount_inr']}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Self test failed: {e}")
