"""
Razorpay RTO Risk-Ops & Profit Protection Engine
Automated Quality Assurance & Edge-Case Unit Test Suite
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure src imports resolve correctly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.risk_engine import RiskEngine
from src.explain import RiskExplainer


class TestRiskEngine(unittest.TestCase):
    """Comprehensive test suite verifying engine stability, edge cases, and business logic."""

    @classmethod
    def setUpClass(cls):
        """Initialize the risk engine and explainer once for all tests."""
        cls.engine = RiskEngine()
        cls.explainer = RiskExplainer(cls.engine)

    def test_01_engine_initialization(self):
        """Verify model artifacts and metadata load cleanly."""
        self.assertIsNotNone(self.engine.model, "Model object should be loaded")
        self.assertIsNotNone(self.engine.winner_name, "Winner model name must exist")
        self.assertIsInstance(self.engine.feature_columns, list, "Feature columns must be a list")
        self.assertGreater(len(self.engine.feature_columns), 0, "Feature columns list cannot be empty")
        self.assertIsInstance(self.engine.optimal_threshold, float, "Optimal threshold must be float")

    def test_02_normal_order_scoring(self):
        """Verify single order scoring returns all required financial and risk keys."""
        sample_order = {
            "pincode_tier": "Tier 1",
            "pincode_rto_rate": 0.12,
            "payment_mode": "Prepaid",
            "order_value": 2499.0,
            "discount_pct": 10.0,
            "category": "Apparel",
            "is_weekend_order": 0,
            "address_length": 65,
            "address_has_landmark": 1,
            "pin_matches_city": 1,
            "customer_tenure_days": 120,
            "customer_past_orders": 3,
            "customer_past_rto_count": 0,
            "customer_past_rto_rate": 0.0,
        }

        result = self.engine.score_order(sample_order)

        expected_keys = [
            "risk_probability",
            "risk_tier",
            "expected_profit_cod",
            "expected_profit_prepaid",
            "expected_profit_prepaid_discount",
            "action_code",
            "recommended_action",
            "action_payload",
            "optimal_threshold",
            "override_reason",
        ]

        for key in expected_keys:
            self.assertIn(key, result, f"Key '{key}' missing from score_order result")

        self.assertGreaterEqual(result["risk_probability"], 0.0)
        self.assertLessEqual(result["risk_probability"], 1.0)
        self.assertIn(result["action_code"], ["AUTO_SHIP", "VERIFY_ADDRESS_OTP", "INCENTIVIZE_PREPAID", "STRICT_PREPAID_ONLY"])

    def test_03_expected_profit_calculation(self):
        """Verify mathematical integrity of the margin-aware expected net profit formula."""
        # 100% success case (P(RTO) = 0): Profit = (1000 * 0.40) - 70 = 400 - 70 = 330
        profit_zero_risk = RiskEngine.calculate_expected_profit(
            order_value=1000.0,
            rto_prob=0.0,
            gross_margin=0.40,
            forward_shipping=70.0,
            reverse_shipping=90.0,
            packaging_cost=40.0,
            is_prepaid=False
        )
        self.assertEqual(profit_zero_risk, 330.0)

        # 100% failure case (P(RTO) = 1.0): Loss = -(70 + 90 + 40) = -200
        profit_max_risk = RiskEngine.calculate_expected_profit(
            order_value=1000.0,
            rto_prob=1.0,
            gross_margin=0.40,
            forward_shipping=70.0,
            reverse_shipping=90.0,
            packaging_cost=40.0,
            is_prepaid=False
        )
        self.assertEqual(profit_max_risk, -200.0)

    def test_04_repeat_offender_fraud_override(self):
        """Verify Rule 1: Repeat offenders with >= 66% RTO on >= 3 past orders are hard-blocked."""
        fraud_order = {
            "pincode_tier": "Tier 1",
            "pincode_rto_rate": 0.10,
            "payment_mode": "COD",
            "order_value": 2000.0,
            "customer_past_orders": 4,
            "customer_past_rto_count": 3,
            "customer_past_rto_rate": 0.75,  # 75% return rate on 4 past orders
        }

        result = self.engine.score_order(fraud_order)
        self.assertEqual(result["action_code"], "STRICT_PREPAID_ONLY")
        self.assertEqual(result["risk_probability"], 0.98)
        self.assertIsNotNone(result["override_reason"])
        self.assertIn("Repeat Offender", result["override_reason"])

    def test_05_high_value_address_mismatch_override(self):
        """Verify Rule 2: High value (>= ₹5,000) address mismatch triggers safeguard."""
        mismatch_order = {
            "pincode_tier": "Tier 2",
            "pincode_rto_rate": 0.20,
            "payment_mode": "COD",
            "order_value": 6500.0,  # >= ₹5,000
            "pin_matches_city": 0,  # Mismatch
        }

        result = self.engine.score_order(mismatch_order)
        self.assertGreaterEqual(result["risk_probability"], 0.85)
        self.assertIsNotNone(result["override_reason"])
        self.assertIn("Address Mismatch", result["override_reason"])

    def test_06_unseen_categories_and_dirty_inputs(self):
        """Verify the engine handles completely unknown categories, missing columns, and dirty strings gracefully."""
        dirty_order = {
            "pincode_tier": "Tier 999_UNKNOWN",  # Unseen category
            "category": "Interstellar_Vessel",    # Unseen category
            "order_value": "4500.50",             # String number
            "discount_pct": None,                 # Null value
            # All other 10 features deliberately omitted
        }

        # Must not raise any exceptions
        result = self.engine.score_order(dirty_order)
        self.assertIsInstance(result["risk_probability"], float)
        self.assertIn(result["action_code"], ["AUTO_SHIP", "VERIFY_ADDRESS_OTP", "INCENTIVIZE_PREPAID", "STRICT_PREPAID_ONLY"])

    def test_07_batch_scoring_throughput(self):
        """Verify batch scoring handles a DataFrame correctly with zero dropped rows."""
        data = [
            {"pincode_tier": "Tier 1", "order_value": 1500, "pincode_rto_rate": 0.05, "payment_mode": "Prepaid"},
            {"pincode_tier": "Tier 2", "order_value": 2800, "pincode_rto_rate": 0.25, "payment_mode": "COD"},
            {"pincode_tier": "Tier 3", "order_value": 4200, "pincode_rto_rate": 0.45, "payment_mode": "COD"},
            {"pincode_tier": "Tier 1", "order_value": 8500, "pincode_rto_rate": 0.15, "pin_matches_city": 0},
        ]
        df_in = pd.DataFrame(data)
        df_out = self.engine.score_batch(df_in)

        self.assertEqual(len(df_out), len(df_in), "Output rows must match input rows exactly")
        required_cols = ["risk_probability", "risk_tier", "expected_profit_cod", "expected_profit_prepaid", "action_code", "recommended_action"]
        for col in required_cols:
            self.assertIn(col, df_out.columns, f"Column '{col}' missing from batch output")

    def test_08_treeshap_explainability(self):
        """Verify local SHAP explanations compute and format properly."""
        sample_order = {
            "pincode_tier": "Tier 3",
            "pincode_rto_rate": 0.40,
            "payment_mode": "COD",
            "order_value": 3500,
            "discount_pct": 25.0,
            "category": "Apparel",
            "is_weekend_order": 1,
            "address_length": 25,
            "address_has_landmark": 0,
            "pin_matches_city": 1,
            "customer_tenure_days": 10,
            "customer_past_orders": 1,
            "customer_past_rto_count": 0,
            "customer_past_rto_rate": 0.0
        }

        reasons = self.explainer.explain_prediction(sample_order, top_n=3)
        self.assertIsInstance(reasons, list)
        self.assertLessEqual(len(reasons), 3)

        for r in reasons:
            self.assertIn("factor", r)
            self.assertIn("impact", r)
            self.assertIn("direction", r)
            self.assertIn(r["direction"], ["+", "-"])
            self.assertIsInstance(r["impact"], float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
