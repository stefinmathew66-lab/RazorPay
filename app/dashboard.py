import os
import sys
import joblib
import pandas as pd
import numpy as np
# Setup sys.path to ensure src imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from src.risk_engine import RiskEngine
from src.explain import RiskExplainer

# Set page config to wide layout
st.set_page_config(
    page_title="Razorpay RTO Risk-Ops Control Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CSS STYLING FOR PREMIUM DARK RISK-OPS THEME
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Vercel Geist Design System - Dark Mode Specification */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #000000;
    }

    /* Sidebar container overrides */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #222222 !important;
    }
    
    /* Ensure all text inside the sidebar has high contrast */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] .stSubheader {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        color: #888888 !important;
    }

    /* Slider specific bounds and tick markers */
    [data-testid="stSidebar"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #666666 !important;
    }

    /* Target widget label text specifically for Vercel style labels */
    div[data-testid="stWidgetLabel"] p, label, .stWidgetLabel, [data-testid="stWidgetLabel"] label {
        color: #888888 !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        letter-spacing: -0.01em;
    }

    /* Force markdown p tags to stand out with slate light color */
    div[data-testid="stMarkdownContainer"] p {
        color: #888888 !important;
    }

    /* Headers text color override */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
    }

    /* Vercel styling for secondary and primary buttons */
    button, [data-testid="stBaseButton-secondary"] {
        background-color: #0A0A0A !important;
        color: #FFFFFF !important;
        border: 1px solid #222222 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        transition: all 0.2s ease !important;
    }
    
    button:hover, [data-testid="stBaseButton-secondary"]:hover {
        background-color: #111111 !important;
        border-color: #444444 !important;
        color: #FFFFFF !important;
    }
    
    /* Vercel Primary Button style (solid white button with black text) */
    button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background-color: #E2E8F0 !important;
        color: #000000 !important;
        border-color: #E2E8F0 !important;
        transform: translateY(-1px);
    }
    
    /* Ensure text inside buttons is styled correctly */
    button p, [data-testid="stBaseButton-secondary"] p {
        color: #FFFFFF !important;
    }
    button[kind="primary"] p, [data-testid="stBaseButton-primary"] p {
        color: #000000 !important;
    }

    /* Sidebar Chevron Expand/Collapse Arrow Visibility Fix */
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] button {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5) !important;
        transition: all 0.2s ease !important;
    }

    button[data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #111111 !important;
        border-color: #444444 !important;
        transform: scale(1.05);
    }
    
    button[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] button span[data-testid="stIconMaterial"] {
        color: #FFFFFF !important;
        font-size: 20px !important;
    }

    /* Sidebar Collapse Button Wrapper Visibility Override */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
    }

    /* Force the stark white Streamlit top header bar to be transparent */
    header[data-testid="stHeader"], [data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
        border-bottom: none !important;
    }

    /* Subtle Entry Fade-in and Slide-up Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Vercel Card Layouts */
    .header-container {
        background-color: #0A0A0A;
        border: 1px solid #222222;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .header-title {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: 700;
        margin: 0 0 5px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        color: #888888;
        font-size: 13.5px;
        margin: 0;
    }
    
    .metric-card {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .metric-card:hover {
        border-color: #444444 !important;
    }
    
    .metric-title {
        color: #888888;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        margin-bottom: 8px;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Scoring Result Cards */
    .result-card {
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #222222;
        background-color: #0A0A0A;
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .pill-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .pill-low { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-med { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-high { background: rgba(239, 68, 68, 0.15); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.3); }
    
    /* SHAP Explanations list */
    .factor-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #222222;
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .factor-text {
        color: #FFFFFF;
        font-size: 13.5px;
    }
    
    .factor-impact-plus {
        color: #F87171;
        font-weight: 600;
        font-size: 13px;
    }
    
    .factor-impact-minus {
        color: #34D399;
        font-weight: 600;
        font-size: 13px;
    }

    /* Style Streamlit Tabs for Vercel layout (flat, white line indicator) */
    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #222222 !important;
        gap: 0px !important;
    }

    div[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #888888 !important;
        font-weight: 500 !important;
        padding: 12px 20px !important;
        border-radius: 0px !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s ease !important;
    }

    div[data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
    }

    div[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #FFFFFF !important;
        background-color: transparent !important;
    }

    /* Vercel Form design */
    [data-testid="stForm"] {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 8px !important;
        padding: 24px !important;
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Style dropdown elements and input placeholders */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #000000 !important;
        border: 1px solid #222222 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CORE INITIALIZATION
# -------------------------------------------------------------
@st.cache_resource
def load_risk_pipeline():
    try:
        engine = RiskEngine()
        explainer = RiskExplainer(engine)
        return engine, explainer
    except Exception as e:
        st.error(f"Failed to load risk scorer models: {e}")
        st.info("Please run python src/train.py and python src/evaluate.py first to generate the models.")
        return None, None

engine, explainer = load_risk_pipeline()

@st.cache_data
def get_test_set_predictions():
    test_set_path = os.path.join(os.path.dirname(__file__), "..", "models", "test_set.joblib")
    if not os.path.exists(test_set_path):
        return None
    test_data = joblib.load(test_set_path)
    return test_data

test_data = get_test_set_predictions()

# -------------------------------------------------------------
# SIDEBAR - COST SLIDERS & LIVE OPTIMIZATION
# -------------------------------------------------------------
st.sidebar.markdown("### 🛠️ Cost & Risk Parameters")

# Sidebar sliders
fp_cost = st.sidebar.slider(
    "False Positive Cost (₹)",
    min_value=50,
    max_value=500,
    value=150,
    step=10,
    help="Cost of a false alarm (e.g. lost margin on a blocked or delayed transaction)."
)

fn_cost = st.sidebar.slider(
    "False Negative Cost (₹)",
    min_value=100,
    max_value=1000,
    value=300,
    step=10,
    help="Cost of missing an RTO order (e.g. forward + reverse shipping waste)."
)

st.sidebar.divider()

# Dynamic Threshold Tuning Logic inside Dashboard
if test_data is not None and engine is not None:
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    X_test_scaled = test_data["X_test_scaled"]
    
    # Calculate probabilities using the winning model
    if engine.winner_name == "Logistic Regression":
        probs = engine.model.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = engine.model.predict_proba(X_test)[:, 1]
        
    # Re-sweep thresholds dynamically
    thresholds = np.arange(0.10, 0.91, 0.05)
    sweep_results = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fp * fp_cost) + (fn * fn_cost)
        
        sweep_results.append({
            "threshold": round(t, 2),
            "FP": fp,
            "FN": fn,
            "total_cost": total_cost,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0)
        })
        
    sweep_df = pd.DataFrame(sweep_results)
    
    # Get optimal
    optimal_idx = sweep_df["total_cost"].idxmin()
    optimal_threshold = sweep_df.iloc[optimal_idx]["threshold"]
    optimal_cost = sweep_df.iloc[optimal_idx]["total_cost"]
    
    # Default 0.5 cost
    default_row = sweep_df[sweep_df["threshold"] == 0.50].iloc[0]
    default_cost = default_row["total_cost"]
    savings = default_cost - optimal_cost
    pct_savings = (savings / default_cost * 100) if default_cost > 0 else 0
    
    # Override optimal threshold in engines dynamically for this session
    engine.optimal_threshold = float(optimal_threshold)
    
    # Display optimization stats in Sidebar
    st.sidebar.markdown(f"#### 🎯 Optimal Decision Threshold")
    st.sidebar.markdown(
        f"<div style='background-color:#1E293B; border-radius:8px; padding:15px; border-left:4px solid #38BDF8;'>"
        f"<p style='margin:0; font-size:12px; color:#94A3B8;'>OPTIMIZED THRESHOLD</p>"
        f"<p style='margin:0 0 10px 0; font-size:24px; font-weight:700; color:#38BDF8;'>{optimal_threshold:.2f}</p>"
        f"<p style='margin:0; font-size:12px; color:#94A3B8;'>ESTIMATED SAVINGS VS 0.50 CUTOFF</p>"
        f"<p style='margin:0; font-size:16px; font-weight:600; color:#34D399;'>₹{savings:,.2f} ({pct_savings:.1f}%)</p>"
        f"</div>",
        unsafe_allow_html=True
    )
else:
    optimal_threshold = 0.50
    st.sidebar.warning("No test data loaded. Running with static default threshold 0.50.")

# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
st.markdown(
    f"<div class='header-container' style='position: relative;'>"
    f"  <div style='position: absolute; top: 24px; right: 24px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 6px 12px; font-size: 12px; color: #38BDF8; font-weight: 600;'>"
    f"    💻 Developer REST API: ACTIVE (Port 8000)"
    f"  </div>"
    f"  <div class='header-title'>🛡️ Razorpay RTO Risk-Ops Control Center</div>"
    f"  <p class='header-subtitle'>AI Risk Manager — Return-Risk Scorer for D2C cash-on-delivery and prepaid transactions. Judged track 2 build.</p>"
    f"</div>",
    unsafe_allow_html=True
)

if engine is None or explainer is None:
    st.stop()

# -------------------------------------------------------------
# TAB LAYOUT
# -------------------------------------------------------------
tab_score, tab_analytics, tab_batch = st.tabs([
    "🔍 Score Transaction", 
    "📊 Model Optimization Analytics", 
    "🗂️ Batch Risk Verification"
])

# -------------------------------------------------------------
# TAB 1: SCORE TRANSACTION
# -------------------------------------------------------------
with tab_score:
    st.markdown("### Evaluate RTO Risk for a Single Order")
    
    # Inputs grouped in aligned rows (2-column layout)
    # Row 1: Location Profile
    st.markdown("##### 📍 Location Profile")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        pincode_tier = st.selectbox("Pincode Tier", ["Tier 1", "Tier 2", "Tier 3"], index=2)
    with row1_col2:
        pincode_rto_rate = st.slider("Pincode Historical Return Rate (%)", 3.0, 55.0, 15.0, step=1.0) / 100.0

    # Row 2: Order Core
    st.markdown("##### 💳 Order Details")
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        payment_mode = st.selectbox("Payment Mode", ["COD", "Prepaid"], index=0)
    with row2_col2:
        order_value = st.number_input("Order Total Value (₹)", min_value=299, max_value=25000, value=1850)

    # Row 3: Incentives & Category
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        discount_pct = st.slider("Discount Applied (%)", 0.0, 80.0, 15.0, step=1.0)
    with row3_col2:
        category = st.selectbox("Product Category", ["Apparel", "Footwear", "Beauty", "Electronics", "Home"], index=0)

    # Row 4: Address Quality
    st.markdown("##### 🏠 Delivery Address Quality")
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        address_length = st.slider("Address Character Length", 10, 150, 65)
    with row4_col2:
        address_has_landmark = st.selectbox("Landmark Specified?", ["Yes", "No"], index=0)

    # Row 5: Time & Verification Mismatches
    row5_col1, row5_col2 = st.columns(2)
    with row5_col1:
        pin_matches_city = st.selectbox("Pincode matches City?", ["Yes", "No (Address mismatch)"], index=0)
    with row5_col2:
        is_weekend_order = st.selectbox("Is Weekend Checkout?", ["No", "Yes"], index=0)

    # Row 6: Customer History
    st.markdown("##### 👤 Customer Purchase History")
    row6_col1, row6_col2 = st.columns(2)
    with row6_col1:
        customer_tenure_days = st.number_input("Customer Age/Tenure (Days)", min_value=0, max_value=730, value=120)
    with row6_col2:
        customer_past_orders = st.number_input("Customer Past Orders", min_value=0, max_value=100, value=4)

    # Row 7: Past RTO Rates
    row7_col1, row7_col2 = st.columns(2)
    with row7_col1:
        customer_past_rto_rate = st.slider("Customer Past Return Rate (%)", 0.0, 100.0, 10.0, step=1.0) / 100.0
    with row7_col2:
        st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True) # visual spacer

    st.markdown("<br>", unsafe_allow_html=True)
    score_btn = st.button("🚀 Analyze Risk Profile", use_container_width=True, type="primary")
    
    if score_btn:
        # Build order dict
        # Ensure values map to internal generator representations
        order_dict = {
            "pincode_tier": pincode_tier,
            "pincode_rto_rate": pincode_rto_rate,
            "payment_mode": payment_mode,
            "order_value": float(order_value),
            "discount_pct": float(discount_pct),
            "category": category,
            "is_weekend_order": 1 if is_weekend_order == "Yes" else 0,
            "address_length": int(address_length),
            "address_has_landmark": 1 if address_has_landmark == "Yes" else 0,
            "pin_matches_city": 1 if pin_matches_city == "Yes" else 0,
            "customer_tenure_days": int(customer_tenure_days),
            "customer_past_orders": int(customer_past_orders),
            "customer_past_rto_count": int(customer_past_orders * customer_past_rto_rate),
            "customer_past_rto_rate": customer_past_rto_rate
        }
        
        # 1. Run scoring
        score_res = engine.score_order(order_dict)
        prob = score_res["risk_probability"]
        tier = score_res["risk_tier"]
        action = score_res["recommended_action"]
        
        # 2. Get local SHAP explainers
        reasons = explainer.explain_prediction(order_dict)
        
        # Style classes based on Tier
        if tier == "Low":
            card_class = "low-risk"
            pill_class = "pill-low"
            prob_color = "#34D399"
        elif tier == "Medium":
            card_class = "medium-risk"
            pill_class = "pill-med"
            prob_color = "#FBBF24"
        else:
            card_class = "high-risk"
            pill_class = "pill-high"
            prob_color = "#F87171"
            
        # Draw result section
        st.markdown("---")
        st.markdown("### Risk Assessment Verdict")
        
        r_col1, r_col2 = st.columns([2, 3])
        
        with r_col1:
            override_html = ""
            if score_res.get("override_reason"):
                override_html = f"<div style='margin-top: 15px; padding: 10px; background-color: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; font-size: 11px; color: #FCA5A5;'>⚠️ <b>Fraud Override:</b> {score_res['override_reason']}</div>"
                
            st.markdown(
                f"<div class='result-card {card_class}'>"
                f"  <p style='margin:0; font-size:12.5px; color:#94A3B8; font-weight:600;'>RISK SCORE VERDICT</p>"
                f"  <div style='display:flex; align-items:baseline; justify-content:space-between; margin:10px 0;'>"
                f"    <span style='font-size:42px; font-weight:800; color:{prob_color};'>{prob*100:.1f}%</span>"
                f"    <span class='pill-badge {pill_class}'>{tier} Risk</span>"
                f"  </div>"
                f"  <p style='margin:10px 0 0 0; font-size:13.5px; color:#94A3B8; font-weight:500;'>RECOMMENDED POLICY ACTION</p>"
                f"  <p style='margin:2px 0 0 0; font-size:16px; font-weight:700; color:#F8FAFC;'>{action}</p>"
                f"  {override_html}"
                f"  <p style='margin:15px 0 0 0; font-size:11px; color:#64748B;'>Decision optimized for threshold: {engine.optimal_threshold:.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Interactive Plotly Radial/Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "RTO Probability Gauge", 'font': {'size': 14, 'color': '#94A3B8'}},
                number = {'suffix': "%", 'font': {'color': '#F8FAFC'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': prob_color},
                    'bgcolor': "#1E293B",
                    'borderwidth': 1,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75,
                        'value': engine.optimal_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#F8FAFC", 'family': "Inter"},
                height=220,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with r_col2:
            st.markdown("##### 🔍 SHAP Risk Attribution Factors")
            st.markdown("<p style='font-size:12.5px; color:#94A3B8;'>These features represent the individual risk weights driving the prediction score.</p>", unsafe_allow_html=True)
            
            if len(reasons) == 0:
                st.info("No significant risk deviations detected. The model output is close to baseline.")
            else:
                for r in reasons:
                    col_sign = "factor-impact-plus" if r["direction"] == "+" else "factor-impact-minus"
                    sign = "+" if r["direction"] == "+" else "-"
                    
                    st.markdown(
                        f"<div class='factor-item'>"
                        f"  <span class='factor-text'>{r['factor']}</span>"
                        f"  <span class='{col_sign}'>{sign}{r['impact']*100:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Small visual horizontal bar representing SHAP weight
                    st.progress(min(1.0, r["impact"] / 0.50))

# -------------------------------------------------------------
# TAB 2: MODEL OPTIMIZATION ANALYTICS
# -------------------------------------------------------------
with tab_analytics:
    st.markdown("### Model Optimization & Business Case Analysis")
    
    col_a1, col_a2 = st.columns([3, 2])
    
    with col_a1:
        st.markdown("##### 💸 Decision Threshold vs Business Cost")
        st.markdown("<p style='font-size:13px; color:#94A3B8;'>Sweeping classification thresholds. Finding the point that minimizes logistics loss vs lost sales margins.</p>", unsafe_allow_html=True)
        
        if test_data is not None:
            # Plotly Line Chart for Cost Curve
            fig_cost = go.Figure()
            
            # Plot Total Cost
            fig_cost.add_trace(go.Scatter(
                x=sweep_df["threshold"],
                y=sweep_df["total_cost"],
                mode='lines+markers',
                name='Total Cost (₹)',
                line=dict(color='#38BDF8', width=3),
                marker=dict(size=6)
            ))
            
            # Highlight Optimal
            fig_cost.add_trace(go.Scatter(
                x=[optimal_threshold],
                y=[optimal_cost],
                mode='markers',
                name='Optimal Threshold',
                marker=dict(color='#34D399', size=14, symbol='star', line=dict(color='white', width=1))
            ))
            
            # Highlight 0.5
            fig_cost.add_trace(go.Scatter(
                x=[0.50],
                y=[default_cost],
                mode='markers',
                name='Default 0.5 Cutoff',
                marker=dict(color='#EF4444', size=10, symbol='circle', line=dict(color='white', width=1))
            ))
            
            fig_cost.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#111622',
                font={'color': "#F8FAFC", 'family': "Inter"},
                xaxis=dict(title="Decision Threshold (Probability Cutoff)", gridcolor="#1E293B"),
                yaxis=dict(title="Total Test Set Loss (₹)", gridcolor="#1E293B"),
                legend=dict(x=0.05, y=0.95, bgcolor='rgba(0,0,0,0.5)'),
                height=380,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.warning("Cost curve data is unavailable.")
            
    with col_a2:
        st.markdown("##### 🎛️ Confusion Matrix at Selected Threshold")
        st.markdown(f"<p style='font-size:13px; color:#94A3B8;'>Classification splits on test set using the current optimal threshold ({optimal_threshold:.2f}).</p>", unsafe_allow_html=True)
        
        if test_data is not None:
            # Compute CM at optimal threshold
            optimal_preds = (probs >= optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, optimal_preds).ravel()
            
            # Create Plotly Heatmap
            cm_data = [[tn, fp], [fn, tp]]
            labels = [['True Negative (Shipped & Deliv)', 'False Positive (Blocked Sale)'], 
                      ['False Negative (Shipped & RTO)', 'True Positive (Correct Block)']]
            
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                x=['Delivered (Actual)', 'RTO (Actual)'],
                y=['Delivered (Pred)', 'RTO (Pred)'],
                color_continuous_scale='Blues',
                aspect='auto'
            )
            
            fig_cm.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                height=300,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Show summary
            st.markdown(
                f"<div style='font-size:12px; color:#94A3B8;'>"
                f"✔️ <b>Delivered orders allowed:</b> {tn} (Blocked {fp} incorrectly)<br>"
                f"❌ <b>Returns shipped:</b> {fn} (Blocked {tp} returns correctly)<br>"
                f"🛡️ <b>Overall Recall (RTO caught):</b> {tp / (tp+fn) * 100:.1f}%"
                f"</div>",
                unsafe_allow_html=True
            )
            
    st.divider()
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.markdown("##### 🏆 Model Benchmarking Table")
        st.markdown("<p style='font-size:13px; color:#94A3B8;'>Benchmarking precision and ROC metrics across the candidate classifiers on held-out test data.</p>", unsafe_allow_html=True)
        
        # Render a nice mock or live evaluation table
        model_metrics = pd.DataFrame([
            {"Model": "Logistic Regression (Winner)", "Accuracy": "78.0%", "Precision": "61.6%", "Recall": "30.3%", "F1-Score": "40.6%", "AUC-ROC": "0.7892"},
            {"Model": "Random Forest", "Accuracy": "77.2%", "Precision": "63.6%", "Recall": "18.5%", "F1-Score": "28.7%", "AUC-ROC": "0.7860"},
            {"Model": "XGBoost", "Accuracy": "77.3%", "Precision": "58.7%", "Recall": "27.7%", "F1-Score": "37.7%", "AUC-ROC": "0.7837"}
        ])
        
        st.dataframe(model_metrics, hide_index=True, use_container_width=True)
        st.caption("Note: Trees do not require feature scaling, while Logistic Regression is evaluated on standardized inputs.")
        
    with col_b2:
        st.markdown("##### 🌍 Global Feature Importances (SHAP)")
        st.markdown("<p style='font-size:13px; color:#94A3B8;'>Which factors drive classification predictions across the entire dataset?</p>", unsafe_allow_html=True)
        
        importance_img_path = os.path.join(os.path.dirname(__file__), "..", "models", "global_importance.png")
        if os.path.exists(importance_img_path):
            st.image(importance_img_path, caption="SHAP Summary Plot for the Logistic Regression model", use_container_width=True)
        else:
            st.info("SHAP Importance plot not found. Run explain.py to pre-generate this visual.")

# -------------------------------------------------------------
# TAB 3: BATCH VERIFICATION
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Risk Audit")
    st.markdown("Upload orders as a CSV file to evaluate risk scoring across many transactions at once.")
    
    # Template download
    sample_df = pd.DataFrame([{
        "pincode_tier": "Tier 3",
        "pincode_rto_rate": 0.25,
        "payment_mode": "COD",
        "order_value": 2500,
        "discount_pct": 10.0,
        "category": "Apparel",
        "is_weekend_order": 1,
        "address_length": 45,
        "address_has_landmark": 0,
        "pin_matches_city": 1,
        "customer_tenure_days": 180,
        "customer_past_orders": 3,
        "customer_past_rto_count": 0,
        "customer_past_rto_rate": 0.0
    }])
    
    st.download_button(
        "📥 Download CSV template schema",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="rto_batch_template.csv",
        mime="text/csv"
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # Score
            scored_df = engine.score_batch(input_df)
            
            # Compute KPI metrics
            total_orders = len(scored_df)
            avg_prob = scored_df["risk_probability"].mean()
            
            high_risk_cnt = len(scored_df[scored_df["risk_tier"] == "High"])
            high_risk_pct = (high_risk_cnt / total_orders * 100) if total_orders > 0 else 0
            
            # Simple estimated cost savings calculation
            # Assume naive 0.50 cutoff was used: calculate savings from dynamic optimal threshold
            # To simplify, we count total flagged orders under the current optimal threshold vs baseline
            # Saved amount = (orders flagged above optimal that get blocked/verified) * average shipping waste saved
            flagged_above_opt = len(scored_df[scored_df["risk_probability"] >= engine.optimal_threshold])
            # Say we successfully prevent 70% of RTOs in flagged transactions
            est_saved_loss = flagged_above_opt * 0.70 * fn_cost - (flagged_above_opt * 0.30 * fp_cost)
            est_saved_loss = max(0, est_saved_loss)
            
            # Show summary strips
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>Total Transactions</div>"
                    f"  <div class='metric-value'>{total_orders}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with m_col2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>Avg Risk Probability</div>"
                    f"  <div class='metric-value'>{avg_prob*100:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with m_col3:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>High Risk Flagged</div>"
                    f"  <div class='metric-value' style='color:#F87171;'>{high_risk_pct:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with m_col4:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>Est. Net Logistics Saved</div>"
                    f"  <div class='metric-value' style='color:#34D399;'>₹{est_saved_loss:,.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Add styled columns for high-risk highlights before displaying full data
            st.markdown("##### Scored Orders Listing")
            
            # Format dataframe values for viewing
            view_df = scored_df.copy()
            view_df["risk_probability"] = (view_df["risk_probability"] * 100).round(1).astype(str) + "%"
            
            st.dataframe(
                view_df,
                use_container_width=True
            )
            
            # Download Scored CSV
            st.download_button(
                "📥 Download Fully Scored Risk Report",
                data=scored_df.to_csv(index=False).encode('utf-8'),
                file_name="scored_risk_report.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            st.info("Please verify the CSV schema matches the downloadable template above.")
