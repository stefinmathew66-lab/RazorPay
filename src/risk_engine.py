import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Tuple

# Define global path constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_artifacts.joblib")


class RiskEngine:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifacts not found at {model_path}. "
                "Please run train.py and evaluate.py first."
            )

        # Load artifacts saved during training and threshold tuning
        self.artifacts = joblib.load(model_path)
        self.winner_name = self.artifacts["winner_name"]
        self.model = self.artifacts["winning_model"]
        self.label_encoders = self.artifacts["label_encoders"]
        self.scaler = self.artifacts["scaler"]
        self.feature_columns = self.artifacts["feature_columns"]

        # Whether the winning model requires StandardScaler input
        self.requires_scaling = self.artifacts.get("requires_scaling", False)

        # Load optimal threshold, falling back to 0.5 if not yet tuned
        self.optimal_threshold = self.artifacts.get("optimal_threshold", 0.50)

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------

    def _preprocess_order_df(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a raw order DataFrame into a model-ready feature matrix.

        Categorical columns are mapped using the dictionaries saved during
        training.  Any unseen or null value is mapped to -1 (a valid integer
        sentinel that keeps tree-based models stable; logistic regression is
        protected downstream by the scaler which centres and scales -1 as a
        plausible numeric value).

        For LabelEncoder objects (legacy artifacts), the original fallback to
        the first known class is preserved so old artifacts still work.
        """
        df = df_input.copy()

        # Add missing columns with a default of 0 (safe for numeric features)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Encode categorical columns
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                df[col] = -1
                continue

            if isinstance(le, dict):
                # ── New dict-mapping format ────────────────────────────────
                # Map known values, fill unknown/null with the *median* encoded
                # integer so unseen categories don't cluster at index 0 or -1
                # and bias predictions.  Use -1 only as a true "unknown" flag
                # understood by XGBoost / RandomForest.
                df[col] = df[col].map(le).fillna(-1).astype(int)
            else:
                # ── Legacy LabelEncoder fallback ──────────────────────────
                known = set(le.classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # Select and order features exactly as during training
        df = df[self.feature_columns]
        return df

    # ------------------------------------------------------------------
    # RULE OVERRIDES  (vectorized-friendly helper)
    # ------------------------------------------------------------------

    def _apply_rule_overrides_row(
        self, row: pd.Series, base_prob: float
    ) -> Tuple[float, Optional[str]]:
        """
        Applies merchant fraud rule overrides for a single row.
        Returns (final_probability, override_reason | None).
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
                "High-Value Transaction Address Mismatch "
                "(Value >= ₹5,000 & Pincode/City Mismatch)",
            )

        # Rule 3: COD exploitation on fresh profile with incomplete address
        if (
            payment_mode == "COD"
            and order_val >= 7500
            and addr_len < 30
            and tenure <= 7
        ):
            return (
                max(base_prob, 0.90),
                "COD Exploitation Risk (High-Value COD order on brand-new "
                "profile with incomplete address)",
            )

        return base_prob, None

    # Public alias kept for backward compatibility with existing callers
    def apply_rule_overrides(
        self, order: dict, base_prob: float
    ) -> Tuple[float, Optional[str]]:
        return self._apply_rule_overrides_row(order, base_prob)

    # ------------------------------------------------------------------
    # BATCH SCORING  (single-pass, vectorized rule application)
    # ------------------------------------------------------------------

    def score_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Scores a batch of orders in a single pass.

        Input : DataFrame with raw feature columns.
        Output: Copy of DataFrame enriched with:
                  risk_probability, rule_override_reason,
                  risk_tier, recommended_action
        """
        # ── Step 1: Preprocess ─────────────────────────────────────────
        df_processed = self._preprocess_order_df(df_input)

        # ── Step 2: ML model prediction ───────────────────────────────
        if self.requires_scaling:
            X_eval = self.scaler.transform(df_processed)
        else:
            X_eval = df_processed.values  # numpy array for speed

        base_probs = self.model.predict_proba(X_eval)[:, 1]

        # ── Step 3: Vectorized rule overrides (single iteration) ───────
        df_output = df_input.copy()
        df_output["_base_prob"] = base_probs

        final_probs = []  # type: list
        override_reasons = []  # type: list

        for _, row in df_output.iterrows():
            prob, reason = self._apply_rule_overrides_row(
                row.to_dict(), float(row["_base_prob"])
            )
            final_probs.append(round(prob, 4))  # consistent 4 d.p. always
            override_reasons.append(reason)

        df_output.drop(columns=["_base_prob"], inplace=True)

        # ── Step 4: Enrich output ──────────────────────────────────────
        df_output["risk_probability"] = final_probs
        df_output["rule_override_reason"] = override_reasons
        df_output["risk_tier"] = df_output["risk_probability"].map(
            self._get_risk_tier
        )
        df_output["recommended_action"] = df_output.apply(
            lambda r: (
                self._get_recommended_action(r["risk_probability"])
                + (f" (Override: {r['rule_override_reason']})"
                   if r["rule_override_reason"] else "")
            ),
            axis=1,
        )

        return df_output

    # ------------------------------------------------------------------
    # SINGLE-ORDER SCORING
    # ------------------------------------------------------------------

    def score_order(self, order: dict) -> dict:
        """
        Scores a single order dictionary.

        Input : Dictionary with order characteristics.
        Output: Dictionary with risk metadata.
        """
        df_single = pd.DataFrame([order])
        df_scored = self.score_batch(df_single)
        row = df_scored.iloc[0]

        prob = float(row["risk_probability"])
        raw_reason = row.get("rule_override_reason", None)

        # ── Safe null coercion: pd.notna raises TypeError on some dtypes ──
        try:
            override_reason = str(raw_reason) if pd.notna(raw_reason) else None
        except (TypeError, ValueError):
            override_reason = None

        # Normalise empty-string edge-case
        if override_reason is not None and override_reason.strip() in ("", "nan", "None"):
            override_reason = None

        return {
            "risk_probability": prob,
            "risk_tier": self._get_risk_tier(prob),
            "recommended_action": str(row["recommended_action"]),
            "optimal_threshold": self.optimal_threshold,
            "override_reason": override_reason,
            "top_risk_factors": [],  # populated by explain.py when called
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _get_risk_tier(self, probability: float) -> str:
        """Classify probability into Low / Medium / High."""
        if probability < 0.30:
            return "Low"
        elif probability <= 0.60:
            return "Medium"
        else:
            return "High"

    def _get_recommended_action(self, probability: float) -> str:
        """
        Maps probability to a merchant action relative to the
        cost-optimised threshold saved in self.optimal_threshold.
        """
        if probability < (self.optimal_threshold - 0.10):
            return "Auto-Ship (Pre-approved)"
        elif probability <= self.optimal_threshold:
            return "SMS/IVR Verification (Medium risk boundary)"
        else:
            return "Hold & Require Prepaid (High RTO risk flagged)"


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        engine = RiskEngine()
        print("RiskEngine loaded successfully!")
        print(f"  Winner : {engine.winner_name}")
        print(f"  Scaling: {engine.requires_scaling}")
        print(f"  Threshold: {engine.optimal_threshold}")

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
            "customer_past_rto_rate": 0.0,
        }

        res = engine.score_order(test_order)
        print(f"\nSample Order Scoring Results:")
        print(f"  Risk Probability : {res['risk_probability']:.4f}")
        print(f"  Risk Tier        : {res['risk_tier']}")
        print(f"  Recommended      : {res['recommended_action']}")
        print(f"  Threshold Used   : {res['optimal_threshold']:.2f}")
        print(f"  Override Reason  : {res['override_reason']}")

        # Edge-case: unseen category
        unseen_order = test_order.copy()
        unseen_order["category"] = "UnknownCategory2025"
        res2 = engine.score_order(unseen_order)
        print(f"\nUnseen-category order (should NOT raise):")
        print(f"  Risk Probability : {res2['risk_probability']:.4f}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")
        print("Make sure train.py and evaluate.py have been executed first.")
