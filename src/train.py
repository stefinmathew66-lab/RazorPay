import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_and_evaluate():
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # 1. Load data
    data_path = os.path.join("data", "rto_orders.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 2. Drop identifiers and target-leakage debug features
    # pincode_id is categorical with 300 levels, we drop it since we have pincode_rto_rate and pincode_tier
    # _true_risk_prob is the true probability from generator (must drop to prevent leakage)
    X = df.drop(columns=["pincode_id", "_true_risk_prob", "is_rto"])
    y = df["is_rto"]
    
    # 3. Handle Categorical Columns
    categorical_cols = ["pincode_tier", "payment_mode", "category"]
    label_encoders = {}
    
    # Copy features to prevent SettingWithCopyWarning
    X = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {list(le.classes_)} -> {list(range(len(le.classes_)))}")
        
    # Save the order of features to ensure consistent production scoring
    feature_columns = list(X.columns)
    
    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set shape: {X_train.shape[0]} samples (RTO rate: {y_train.mean()*100:.2f}%)")
    print(f"Test set shape:  {X_test.shape[0]} samples (RTO rate: {y_test.mean()*100:.2f}%)")
    
    # 5. Fit Scaler (Required for Logistic Regression)
    # Fit on training set, transform both train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train Models
    print("\nTraining models...")
    
    # Model 1: Logistic Regression (interpretable linear baseline)
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    
    # Model 2: Random Forest (ensemble tree bagger)
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train) # Trees do not require scaling
    
    # Model 3: XGBoost Classifier (gradient booster)
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    
    # 7. Evaluate Models on Held-Out Test Set
    models = {
        "Logistic Regression": (log_reg, X_test_scaled),
        "Random Forest": (rf, X_test),
        "XGBoost": (xgb, X_test)
    }
    
    results = []
    trained_objects = {
        "Logistic Regression": log_reg,
        "Random Forest": rf,
        "XGBoost": xgb
    }
    
    for name, (model, X_eval) in models.items():
        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        # Check for potential data leakage
        leakage_warning = ""
        if prec > 0.95 or rec > 0.95:
            leakage_warning = "⚠️ POSSIBLE DATA LEAKAGE DETECTED (Precision/Recall > 95%)"
            
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "AUC-ROC": auc,
            "CM": cm,
            "Warning": leakage_warning
        })
        
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'MODEL COMPARISON REPORT (HELD-OUT TEST SET)':^80}")
    print("="*80)
    print(f"{'Model':<25} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'AUC-ROC':<7}")
    print("-"*80)
    for res in results:
        print(f"{res['Model']:<25} | {res['Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['F1-Score']:.4f} | {res['AUC-ROC']:.4f}")
        if res["Warning"]:
            print(f"  {res['Warning']}")
    print("="*80)
    
    # 8. Select Winner (highest AUC-ROC)
    best_res = max(results, key=lambda x: x["AUC-ROC"])
    winner_name = best_res["Model"]
    print(f"\nWinning Model based on AUC-ROC: {winner_name} (AUC: {best_res['AUC-ROC']:.4f})")
    
    # 9. Save artifacts to models/ directory
    # We save all models, label encoders, features, scaler, and the test set for threshold analysis.
    artifacts = {
        "log_reg": log_reg,
        "rf": rf,
        "xgb": xgb,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "winner_name": winner_name,
        "winning_model": trained_objects[winner_name]
    }
    
    # Save the collective artifacts dictionary
    artifacts_path = os.path.join("models", "model_artifacts.joblib")
    joblib.dump(artifacts, artifacts_path)
    
    # Save test set for use in threshold tuning
    test_set_path = os.path.join("models", "test_set.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test, "X_test_scaled": X_test_scaled}, test_set_path)
    
    print(f"Successfully saved all model artifacts to {artifacts_path}")
    print(f"Successfully saved test sets to {test_set_path}")

if __name__ == "__main__":
    train_and_evaluate()
