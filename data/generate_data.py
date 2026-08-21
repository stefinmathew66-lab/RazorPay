import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_orders=12000, seed=42):
    np.random.seed(seed)
    
    print(f"Generating {num_orders} synthetic e-commerce orders...")
    
    # 1. Generate Pincode level data
    num_pincodes = 300
    pincodes = [f"PIN_{100000 + i}" for i in range(num_pincodes)]
    
    # Intrinsic pincode RTO rate (Beta distribution: long tail of high risk)
    # Beta(2, 8) ranges mostly 0 to 0.5, with mean ~0.2
    pincode_base_rates = np.random.beta(2, 8, size=num_pincodes)
    # Scale base rate to range [0.03, 0.55]
    pincode_base_rates = 0.03 + pincode_base_rates * 0.52
    
    # Pincode tier mapping: Tier 1 (30%), Tier 2 (40%), Tier 3 (30%)
    pincode_tiers = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=num_pincodes, p=[0.3, 0.4, 0.3])
    
    pincode_df = pd.DataFrame({
        "pincode_id": pincodes,
        "pincode_rto_rate": pincode_base_rates,
        "pincode_tier": pincode_tiers
    })
    
    # 2. Generate Order characteristics
    # Payment mode: COD is 55% majority
    payment_modes = np.random.choice(["COD", "Prepaid"], size=num_orders, p=[0.55, 0.45])
    
    # Order value: Right-skewed lognormal distribution
    # Rescaled to be mostly between 300 and 6000, with rare higher orders
    order_values = np.random.lognormal(mean=7.5, sigma=0.7, size=num_orders)
    order_values = np.clip(order_values, 299, 15000).round()
    
    # Discount percentage: Beta(1.5, 3.5) * 80 -> heavier discounts correlate with impulsive buys
    discount_pcts = (np.random.beta(1.5, 3.5, size=num_orders) * 80).round(1)
    
    # Category distribution
    categories = np.random.choice(
        ["Apparel", "Electronics", "Beauty", "Footwear", "Home"], 
        size=num_orders, 
        p=[0.40, 0.15, 0.20, 0.15, 0.10]
    )
    
    # Weekend order: 30% are weekend purchases
    is_weekend_orders = np.random.choice([0, 1], size=num_orders, p=[0.70, 0.30])
    
    # 3. Address Quality characteristics
    # Address length: mostly 30 to 120 chars
    address_lengths = np.random.normal(loc=65, scale=25, size=num_orders).astype(int)
    address_lengths = np.clip(address_lengths, 10, 150)
    
    # Address landmark binary indicator
    address_has_landmarks = np.random.choice([0, 1], size=num_orders, p=[0.40, 0.60])
    
    # Pincode matches city (sanity check): 97% match, 3% mismatch
    pin_matches_cities = np.random.choice([1, 0], size=num_orders, p=[0.97, 0.03])
    
    # 4. Customer History
    # Customer tenure in days
    customer_tenures = np.random.randint(0, 730, size=num_orders)
    
    # Customer past orders (loosely correlated with tenure)
    # Lambda = tenure / 30 -> older customers have more orders
    past_order_lambdas = np.maximum(0.5, customer_tenures / 30.0)
    customer_past_orders = np.random.poisson(lam=past_order_lambdas).astype(int)
    customer_past_orders = np.clip(customer_past_orders, 0, 50)
    
    # Customer past RTO count and rate
    customer_past_rto_rates = []
    customer_past_rto_counts = []
    
    # Assign pincode mapping to orders to help with individual customer past RTO rates
    order_pincodes = np.random.choice(pincodes, size=num_orders)
    order_pincode_rates = pincode_df.set_index("pincode_id").loc[order_pincodes, "pincode_rto_rate"].values
    
    for i in range(num_orders):
        past_cnt = customer_past_orders[i]
        if past_cnt == 0:
            customer_past_rto_rates.append(0.0)
            customer_past_rto_counts.append(0)
        else:
            # Personal intrinsic rate is partially correlated with pincode base rate plus personal noise
            personal_rate = np.clip(np.random.normal(loc=order_pincode_rates[i], scale=0.1), 0, 0.8)
            # Sample historical RTOs
            rto_cnt = np.random.binomial(n=past_cnt, p=personal_rate)
            customer_past_rto_counts.append(rto_cnt)
            customer_past_rto_rates.append(float(rto_cnt) / past_cnt)
            
    customer_past_rto_counts = np.array(customer_past_rto_counts)
    customer_past_rto_rates = np.array(customer_past_rto_rates)
    
    # Assemble raw feature DataFrame
    df = pd.DataFrame({
        "pincode_id": order_pincodes,
        "payment_mode": payment_modes,
        "order_value": order_values,
        "discount_pct": discount_pcts,
        "category": categories,
        "is_weekend_order": is_weekend_orders,
        "address_length": address_lengths,
        "address_has_landmark": address_has_landmarks,
        "pin_matches_city": pin_matches_cities,
        "customer_tenure_days": customer_tenures,
        "customer_past_orders": customer_past_orders,
        "customer_past_rto_count": customer_past_rto_counts,
        "customer_past_rto_rate": customer_past_rto_rates
    })
    
    # Merge pincode features
    df = df.merge(pincode_df, on="pincode_id", how="left")
    
    # 5. Calibrate Target Generation Logic (Latent Risk Score -> Probability -> Sample is_rto)
    # Standardize factors to reasonable ranges for logit calculations
    is_cod = (df["payment_mode"] == "COD").astype(int)
    disc_val = df["discount_pct"] / 100.0
    addr_len_scaled = (100.0 - df["address_length"]) / 100.0 # short address increases risk
    tenure_scaled = (730.0 - df["customer_tenure_days"]) / 730.0 # low tenure increases risk
    category_risk_map = {
        "Apparel": 0.40,
        "Footwear": 0.30,
        "Beauty": 0.0,
        "Electronics": -0.10,
        "Home": -0.40
    }
    cat_risk = df["category"].map(category_risk_map).fillna(0.0)
    
    # Base coefficients
    # We will search for beta_0 (intercept) and beta_cod to hit:
    # COD RTO: ~37.5%, Prepaid RTO: ~7.5%, Overall: ~24%
    beta_0 = -4.5
    beta_cod = 2.0
    
    # Simple coordinate descent to calibrate parameters for the specific seed
    for iteration in range(30):
        # Compute logits
        logit = (
            beta_0 +
            3.50 * df["pincode_rto_rate"] +
            beta_cod * is_cod +
            1.10 * disc_val +
            0.30 * df["is_weekend_order"] +
            0.80 * addr_len_scaled -
            0.50 * df["address_has_landmark"] +
            1.40 * (1 - df["pin_matches_city"]) +
            0.60 * tenure_scaled +
            2.50 * df["customer_past_rto_rate"] +
            cat_risk
        )
        probs = 1.0 / (1.0 + np.exp(-logit))
        probs = np.clip(probs, 0.02, 0.92)
        
        # Calculate simulated rates based on expected probabilities
        pred_cod_rto = probs[df["payment_mode"] == "COD"].mean()
        pred_prepaid_rto = probs[df["payment_mode"] == "Prepaid"].mean()
        
        # Adjust coefficients based on errors
        cod_err = pred_cod_rto - 0.375
        prepaid_err = pred_prepaid_rto - 0.075
        
        # Intercept shifts prepaid rate (since beta_cod does not apply there)
        beta_0 -= prepaid_err * 2.0
        # COD coefficient shifts the gap
        beta_cod -= (cod_err - prepaid_err) * 2.0
        
    print(f"Calibrated Generator Parameters: Intercept={beta_0:.4f}, COD_Weight={beta_cod:.4f}")
    
    # Final logit with calibrated parameters
    logit = (
        beta_0 +
        3.50 * df["pincode_rto_rate"] +
        beta_cod * is_cod +
        1.10 * disc_val +
        0.30 * df["is_weekend_order"] +
        0.80 * addr_len_scaled -
        0.50 * df["address_has_landmark"] +
        1.40 * (1 - df["pin_matches_city"]) +
        0.60 * tenure_scaled +
        2.50 * df["customer_past_rto_rate"] +
        cat_risk
    )
    true_risk_probs = 1.0 / (1.0 + np.exp(-logit))
    true_risk_probs = np.clip(true_risk_probs, 0.02, 0.92)
    
    df["_true_risk_prob"] = true_risk_probs
    df["is_rto"] = np.random.binomial(n=1, p=true_risk_probs)
    
    # Re-order columns for clarity
    col_order = [
        "pincode_id", "pincode_tier", "pincode_rto_rate",
        "payment_mode", "order_value", "discount_pct", "category", "is_weekend_order",
        "address_length", "address_has_landmark", "pin_matches_city",
        "customer_tenure_days", "customer_past_orders", "customer_past_rto_count", "customer_past_rto_rate",
        "_true_risk_prob", "is_rto"
    ]
    df = df[col_order]
    
    return df

def run_calibration_report(df):
    print("\n" + "="*40)
    print("      DATA QUALITY & CALIBRATION REPORT")
    print("="*40)
    total_rows = len(df)
    print(f"Total Rows: {total_rows}")
    
    # RTO Rates
    overall_rto = df["is_rto"].mean() * 100
    cod_rto = df[df["payment_mode"] == "COD"]["is_rto"].mean() * 100
    prepaid_rto = df[df["payment_mode"] == "Prepaid"]["is_rto"].mean() * 100
    
    print(f"Overall RTO Rate: {overall_rto:.2f}% (Target: 20-25%)")
    print(f"COD RTO Rate:     {cod_rto:.2f}% (Target: 35-40%)")
    print(f"Prepaid RTO Rate: {prepaid_rto:.2f}% (Target: 5-10%)")
    
    # Feature correlations
    print("\nFeature Correlations with RTO Target:")
    numeric_cols = [
        "pincode_rto_rate", "order_value", "discount_pct", "is_weekend_order", 
        "address_length", "address_has_landmark", "pin_matches_city",
        "customer_tenure_days", "customer_past_orders", "customer_past_rto_rate", "is_rto"
    ]
    corr = df[numeric_cols].corr()["is_rto"].sort_values(ascending=False)
    for col, val in corr.items():
        if col != "is_rto":
            print(f" - {col:<24}: {val:+.4f}")
            
    print("="*40)

if __name__ == "__main__":
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    
    # Generate data
    df = generate_synthetic_data()
    
    # Save to CSV
    output_path = os.path.join("data", "rto_orders.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved synthetic data to {output_path}")
    
    # Run calibration check
    run_calibration_report(df)
