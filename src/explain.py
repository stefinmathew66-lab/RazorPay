import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Ensure matplotlib runs in headless mode (no GUI window popped)
import matplotlib
matplotlib.use('Agg')

try:
    from src.risk_engine import RiskEngine
except ModuleNotFoundError:
    from risk_engine import RiskEngine

class RiskExplainer:
    def __init__(self, engine: RiskEngine = None):
        if engine is None:
            engine = RiskEngine()
            
        self.engine = engine
        self.winner_name = engine.winner_name
        self.model = engine.model
        self.feature_columns = engine.feature_columns
        self.scaler = engine.scaler
        
        # Initialize the correct explainer depending on model architecture
        print(f"Initializing SHAP explainer for {self.winner_name}...")
        
        # Load a small background sample from test set to reference base values
        test_set_path = os.path.join(os.path.dirname(__file__), "..", "models", "test_set.joblib")
        if os.path.exists(test_set_path):
            test_data = joblib.load(test_set_path)
            # Use 100 samples for background references
            background_data = test_data["X_test"]
            background_scaled = test_data["X_test_scaled"]
        else:
            background_data = None
            background_scaled = None
            
        if self.winner_name == "Logistic Regression":
            if background_scaled is not None:
                self.explainer = shap.LinearExplainer(self.model, background_scaled)
            else:
                self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_columns))))
        else:
            # Tree based model (Random Forest / XGBoost)
            self.explainer = shap.TreeExplainer(self.model)
            
    def generate_global_importance(self, save_path=None):
        """Generates and saves the SHAP summary plot representing global feature importances."""
        if save_path is None:
            save_path = os.path.join(os.path.dirname(__file__), "..", "models", "global_importance.png")
            
        test_set_path = os.path.join(os.path.dirname(__file__), "..", "models", "test_set.joblib")
        if not os.path.exists(test_set_path):
            raise FileNotFoundError("Test dataset not found. Please run train.py first.")
            
        test_data = joblib.load(test_set_path)
        
        if self.winner_name == "Logistic Regression":
            X_eval = test_data["X_test_scaled"]
            # Convert scaled test data back to a pandas DataFrame for readable labels
            X_eval_df = pd.DataFrame(X_eval, columns=self.feature_columns)
        else:
            X_eval_df = test_data["X_test"]
            
        # Compute SHAP values
        shap_values = self.explainer(X_eval_df)
        
        # Plot
        plt.figure(figsize=(10, 6))
        # Customizing look
        plt.title(f"Global Risk Features Importance (SHAP values - {self.winner_name})", fontsize=14, pad=15)
        shap.summary_plot(shap_values, X_eval_df, show=False, plot_size=None)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Global SHAP feature importance plot saved to {save_path}")
        
    def explain_prediction(self, order: dict, top_n: int = 4) -> list[dict]:
        """
        Computes SHAP values for a single transaction dictionary.
        Returns: list of dicts with keys: 'factor', 'impact' (float probability change), 'direction' (str '+' or '-')
        """
        # Preprocess order using the risk engine helper
        df_single = self.engine._preprocess_order_df(pd.DataFrame([order]))
        
        if self.winner_name == "Logistic Regression":
            df_input = self.scaler.transform(df_single)
            df_input_df = pd.DataFrame(df_input, columns=self.feature_columns)
            # LinearExplainer values are log-odds.
            raw_shap = self.explainer.shap_values(df_input_df)
            # Approximate probability shift by scaling log-odds SHAP values
            # (Scale by 0.25 is standard logistic calibration slope at p=0.5)
            shap_array = raw_shap[0] * 0.22 
        else:
            df_input_df = df_single
            # TreeExplainer outputs SHAP values
            raw_shap = self.explainer.shap_values(df_input_df)
            
            # Handle list output for classification (SHAP lists values per class [0, 1] for RF)
            if isinstance(raw_shap, list):
                shap_array = raw_shap[1][0] # use class 1 (RTO class)
            elif len(raw_shap.shape) == 3: # Some versions return (nsamples, nfeatures, nclasses)
                shap_array = raw_shap[0, :, 1]
            else:
                shap_array = raw_shap[0] # XGBoost outputs class 1 probability directly in binary classification
                
        # Build human-readable explanations based on feature values
        explanations = []
        
        for idx, col in enumerate(self.feature_columns):
            val = order.get(col, df_single.iloc[0][col])
            impact = float(shap_array[idx])
            
            # Skip negligible contributions
            if abs(impact) < 0.005:
                continue
                
            # Formatting features into plain English descriptions
            factor_desc = ""
            if col == "payment_mode":
                factor_desc = f"Payment mode is {val}"
            elif col == "pincode_rto_rate":
                factor_desc = f"Area historical return rate is high ({val*100:.1f}%)" if impact > 0 else f"Area historical return rate is low ({val*100:.1f}%)"
            elif col == "customer_past_rto_rate":
                if val > 0:
                    factor_desc = f"Customer returns history ({val*100:.1f}%)"
                else:
                    factor_desc = "Zero returns history"
            elif col == "discount_pct":
                factor_desc = f"High discount applied ({val:.1f}%)" if val > 20 else f"Modest discount ({val:.1f}%)"
            elif col == "address_length":
                factor_desc = f"Short/incomplete delivery address ({val} chars)" if val < 40 else f"Detailed address length ({val} chars)"
            elif col == "address_has_landmark":
                factor_desc = "No local landmark provided" if val == 0 else "Local landmark specified"
            elif col == "pin_matches_city":
                factor_desc = "Pincode mismatches selected city" if val == 0 else "Pincode matches selected city"
            elif col == "customer_tenure_days":
                factor_desc = f"New/recent customer (Tenure: {val} days)" if val < 90 else f"Loyal customer (Tenure: {val} days)"
            elif col == "order_value":
                factor_desc = f"High transaction value (₹{val:,})" if val > 3000 else f"Modest transaction value (₹{val:,})"
            elif col == "is_weekend_order":
                factor_desc = "Weekend checkout order" if val == 1 else "Weekday checkout order"
            elif col == "pincode_tier":
                factor_desc = f"Delivery to {val} location"
            elif col == "category":
                factor_desc = f"Purchasing {val} product category"
            else:
                factor_desc = f"{col.replace('_', ' ').title()}: {val}"
                
            direction = "+" if impact > 0 else "-"
            
            explanations.append({
                "feature": col,
                "factor": factor_desc,
                "impact": abs(impact),
                "direction": direction
            })
            
        # Sort by impact descending
        explanations = sorted(explanations, key=lambda x: x["impact"], reverse=True)
        return explanations[:top_n]

if __name__ == "__main__":
    # Test script self-check
    try:
        explainer = RiskExplainer()
        print("Generating global importance plot...")
        explainer.generate_global_importance()
        
        test_order = {
            "pincode_tier": "Tier 3",
            "pincode_rto_rate": 0.42,
            "payment_mode": "COD",
            "order_value": 4500,
            "discount_pct": 55.0,
            "category": "Apparel",
            "is_weekend_order": 1,
            "address_length": 22,
            "address_has_landmark": 0,
            "pin_matches_city": 0,
            "customer_tenure_days": 12,
            "customer_past_orders": 1,
            "customer_past_rto_count": 1,
            "customer_past_rto_rate": 1.0
        }
        
        reasons = explainer.explain_prediction(test_order)
        print("\nPredictive Risk Factors for Test Order:")
        for r in reasons:
            sign = "+" if r["direction"] == "+" else "-"
            print(f" - {r['factor']} ({sign}{r['impact']*100:.1f}%)")
            
    except Exception as e:
        print(f"Error during self-test: {e}")
        print("Make sure train.py and evaluate.py have been executed first.")
