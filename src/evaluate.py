import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# -------------------------------------------------------------
# BUSINESS COST CONSTANTS
# -------------------------------------------------------------
# False Positive (FP) = Predict RTO (Block/Hold), but order was actually Delivered.
# Cost = Lost margin on the sale (approx ₹150).
FALSE_POSITIVE_COST = 150.0

# False Negative (FN) = Predict Delivered (Ship), but order actually returns to origin (RTO).
# Cost = Wasted round-trip shipping fee (forward + reverse logistics, approx ₹300).
FALSE_NEGATIVE_COST = 300.0

def tune_threshold():
    # 1. Load models and evaluation sets
    artifacts_path = os.path.join("models", "model_artifacts.joblib")
    test_set_path = os.path.join("models", "test_set.joblib")
    
    if not os.path.exists(artifacts_path) or not os.path.exists(test_set_path):
        raise FileNotFoundError("Model artifacts not found. Please run train.py first.")
        
    artifacts = joblib.load(artifacts_path)
    test_set = joblib.load(test_set_path)
    
    winner_name = artifacts["winner_name"]
    model = artifacts["winning_model"]
    
    X_test = test_set["X_test"]
    y_test = test_set["y_test"]
    X_test_scaled = test_set["X_test_scaled"]
    
    print(f"Loaded winning model: {winner_name}")
    
    # 2. Get prediction probabilities
    # Logistic Regression requires scaled features, while tree models use raw features
    if winner_name == "Logistic Regression":
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = model.predict_proba(X_test)[:, 1]
        
    # 3. Sweep thresholds
    thresholds = np.arange(0.10, 0.91, 0.05)
    sweep_results = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        # Calculate business cost
        total_cost = (fp * FALSE_POSITIVE_COST) + (fn * FALSE_NEGATIVE_COST)
        
        sweep_results.append({
            "threshold": round(t, 2),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "total_cost": total_cost
        })
        
    sweep_df = pd.DataFrame(sweep_results)
    
    # 4. Find optimal threshold (minimizes total cost)
    optimal_idx = sweep_df["total_cost"].idxmin()
    optimal_row = sweep_df.iloc[optimal_idx]
    optimal_threshold = optimal_row["threshold"]
    optimal_cost = optimal_row["total_cost"]
    
    # 5. Compare with naive 0.5 threshold
    default_row = sweep_df[sweep_df["threshold"] == 0.50].iloc[0]
    default_cost = default_row["total_cost"]
    
    savings = default_cost - optimal_cost
    pct_savings = (savings / default_cost) * 100 if default_cost > 0 else 0
    
    # 6. Save optimal threshold to model artifacts so risk engine can read it dynamically
    artifacts["optimal_threshold"] = float(optimal_threshold)
    joblib.dump(artifacts, artifacts_path)
    
    # Save sweep DataFrame to CSV for dashboard usage
    sweep_csv_path = os.path.join("models", "threshold_sweep.csv")
    sweep_df.to_csv(sweep_csv_path, index=False)
    
    # 7. Print Report
    print("\n" + "="*80)
    print(f"{'COST-SENSITIVE THRESHOLD TUNING REPORT':^80}")
    print("="*80)
    print(f"FP Cost (Lost Sale Margin):  ₹{FALSE_POSITIVE_COST:.2f}")
    print(f"FN Cost (Wasted Shipping):   ₹{FALSE_NEGATIVE_COST:.2f}")
    print(f"Cost Ratio (FN / FP):        {FALSE_NEGATIVE_COST / FALSE_POSITIVE_COST:.2f}x")
    print("-"*80)
    print(f"{'Thresh':<6} | {'FP (Lost)':<9} | {'FN (RTO)':<9} | {'Precision':<9} | {'Recall':<6} | {'Total Cost':<10}")
    print("-"*80)
    for _, r in sweep_df.iterrows():
        prefix = "👉" if r["threshold"] == optimal_threshold else "  "
        print(f"{prefix}{r['threshold']:<4} | {int(r['FP']):<9} | {int(r['FN']):<9} | {r['precision']:.4f}    | {r['recall']:.4f} | ₹{r['total_cost']:,.2f}")
        
    print("-"*80)
    print(f"Naive 0.50 Cost:           ₹{default_cost:,.2f}")
    print(f"Optimal {optimal_threshold:.2f} Cost:        ₹{optimal_cost:,.2f}")
    print(f"Net Financial Savings:     ₹{savings:,.2f} ({pct_savings:.1f}% reduction in loss)")
    
    # Business insight callout
    print("\n" + "="*80)
    print("👉 BUSINESS INSIGHT:")
    if FALSE_NEGATIVE_COST > FALSE_POSITIVE_COST:
        print(" Since wasted shipping (FN) is MORE EXPENSIVE than a lost sale (FP), the optimal")
        print(f" threshold shifted DOWN to {optimal_threshold:.2f} (from 0.50). This prompts us to flag")
        print(" more orders as high-risk, aggressively catching returns even if it creates a few")
        print(" more false alarms, saving the merchant significant logistics spend.")
    elif FALSE_POSITIVE_COST > FALSE_NEGATIVE_COST:
        print(" Since a lost sale (FP) is MORE EXPENSIVE than wasted shipping (FN), the optimal")
        print(f" threshold shifted UP. We tolerate more shipping risk to protect merchant sales volume.")
    else:
        print(" Costs are symmetric; the threshold balances precision and recall evenly.")
    print("="*80)
    
if __name__ == "__main__":
    tune_threshold()
