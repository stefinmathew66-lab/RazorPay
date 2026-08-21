"""
Razorpay RTO Risk-Ops & Profit Protection Engine
Executive Control Center & Dynamic Checkout Profit Arbitrage Dashboard
"""

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
    page_title="Razorpay RTO Risk-Ops & Profit Protection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CSS STYLING FOR PREMIUM VERCEL DARK THEME
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Modern Premium Typography - Plus Jakarta Sans & Inter */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp, 
    div[data-testid="stMarkdownContainer"], 
    p, h1, h2, h3, h4, h5, h6, 
    label, input, select, textarea, button {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
    }
    
    /* Preserve Streamlit Material Icons font */
    span[data-testid="stIconMaterial"], 
    [data-testid="stIconMaterial"], 
    .material-symbols-rounded, 
    .material-symbols-outlined,
    [data-testid="stIconMaterial"] * {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    }
    
    html, body, .stApp {
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
        color: #CCCCCC !important;
    }

    /* Slider specific bounds and tick markers */
    [data-testid="stSidebar"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] {
        color: #FFFFFF !important;
    }

    /* Target widget label text specifically for Vercel style labels with high contrast */
    div[data-testid="stWidgetLabel"] p, label, .stWidgetLabel, [data-testid="stWidgetLabel"] label {
        color: #CCCCCC !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        letter-spacing: -0.01em;
    }

    /* Force all markdown container text, spans, and captions to stand out in light slate */
    div[data-testid="stMarkdownContainer"], 
    div[data-testid="stMarkdownContainer"] span, 
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCaptionContainer"],
    .stCaption,
    .stMarkdown p {
        color: #CCCCCC !important;
    }

    /* Headers text color override with strong visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Give section subheaders inside the forms a clean left accent border */
    div[data-testid="stMarkdownContainer"] h5 {
        border-left: 2px solid #38BDF8 !important;
        padding-left: 10px !important;
        font-weight: 600 !important;
        margin-top: 18px !important;
        margin-bottom: 12px !important;
    }

    /* Styles for sidebar optimal threshold card */
    .threshold-card {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        border-left: 4px solid #38BDF8 !important;
    }
    
    [data-testid="stSidebar"] .threshold-card p,
    .threshold-card p {
        margin: 0 !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stSidebar"] .threshold-card p.card-label,
    .threshold-card p.card-label {
        color: #888888 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stSidebar"] .threshold-card p.threshold-val,
    .threshold-card p.threshold-val {
        color: #38BDF8 !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin: 0 0 10px 0 !important;
    }

    [data-testid="stSidebar"] .threshold-card p.savings-val,
    .threshold-card p.savings-val {
        color: #34D399 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    /* Style default Streamlit Deploy button to be a white button with black text */
    [data-testid="stAppDeployButton"] button,
    button[data-testid="stBaseButton-header"],
    [data-testid="stAppDeployButton"] [data-testid="stBaseButton-header"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stAppDeployButton"] button:hover,
    button[data-testid="stBaseButton-header"]:hover,
    [data-testid="stAppDeployButton"] [data-testid="stBaseButton-header"]:hover {
        background-color: #E2E8F0 !important;
        border-color: #E2E8F0 !important;
        color: #000000 !important;
        transform: translateY(-1px);
    }
    
    [data-testid="stAppDeployButton"] button p,
    button[data-testid="stBaseButton-header"] p,
    [data-testid="stAppDeployButton"] button span {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Vercel styling for secondary and primary buttons with bottom fade-in hover effect */
    button, [data-testid="stBaseButton-secondary"] {
        background-color: #0A0A0A !important;
        color: #FFFFFF !important;
        border: 1px solid #222222 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    button:hover, [data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(0deg, rgba(255, 255, 255, 0.18) 0%, rgba(255, 255, 255, 0.04) 55%, #0A0A0A 100%) !important;
        border-color: #444444 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4), 0 -2px 10px rgba(255, 255, 255, 0.1) inset !important;
        transform: translateY(-1px) !important;
    }
    
    /* Vercel Primary Button style (solid white button with bottom fade highlight) */
    button[kind="primary"], [data-testid="stBaseButton-primary"], .stButton button[kind="primary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: -0.01em !important;
        padding: 10px 24px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(255, 255, 255, 0.1) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover, .stButton button[kind="primary"]:hover {
        background: linear-gradient(0deg, #FFFFFF 0%, #E2E8F0 100%) !important;
        color: #000000 !important;
        border-color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.35), 0 -6px 14px rgba(255, 255, 255, 0.6) inset !important;
    }
    
    /* Ensure text inside buttons is styled correctly */
    button p, [data-testid="stBaseButton-secondary"] p {
        color: #FFFFFF !important;
    }
    button[kind="primary"] p, [data-testid="stBaseButton-primary"] p,
    button[kind="primary"] span, [data-testid="stBaseButton-primary"] span,
    button[kind="primary"] div, [data-testid="stBaseButton-primary"] div {
        color: #000000 !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: -0.01em !important;
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

    /* Subtle Entry Fade-in Animations */
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
        font-size: 24px;
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
    
    .pill-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-yellow { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-orange { background: rgba(251, 146, 60, 0.15); color: #FB923C; border: 1px solid rgba(251, 146, 60, 0.3); }
    .pill-red { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    
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

    /* Segmented Pill Control Bar for Main Navigation Tabs */
    div[data-baseweb="tab-list"] {
        background-color: #0A0A0A !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 6px !important;
        display: inline-flex !important;
        width: max-content !important;
        border-bottom: none !important;
    }

    div[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border-radius: 6px !important;
        border: 1px solid transparent !important;
        border-bottom: none !important;
        padding: 8px 18px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-baseweb="tab"] p,
    div[data-baseweb="tab"] span {
        color: #94A3B8 !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border-color: #334155 !important;
    }

    div[data-baseweb="tab"]:hover p,
    div[data-baseweb="tab"]:hover span {
        color: #FFFFFF !important;
    }

    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.12) !important;
        border: 1px solid #38BDF8 !important;
        border-bottom: 1px solid #38BDF8 !important;
        color: #38BDF8 !important;
        font-weight: 600 !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.15) !important;
    }

    div[data-baseweb="tab"][aria-selected="true"] p,
    div[data-baseweb="tab"][aria-selected="true"] span {
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }

    /* Tab Indicator and Border Strip Removal */
    div[data-baseweb="tab-highlight"], 
    div[data-baseweb="tab-border"],
    div[data-testid="stTabs"] hr {
        display: none !important;
    }

    /* Style dropdown elements and input placeholders */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #000000 !important;
        border: 1px solid #222222 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
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
# SIDEBAR - DYNAMIC MERCHANT ECONOMICS SANDBOX
# -------------------------------------------------------------
st.sidebar.markdown("### 🛠️ Merchant Economics Sandbox")
st.sidebar.markdown("<p style='font-size:12.5px; color:#888888;'>Tune unit economics to optimize net merchant profit rather than arbitrary classification cutoffs.</p>", unsafe_allow_html=True)

gross_margin_pct = st.sidebar.slider(
    "Average Product Gross Margin (%)",
    min_value=10.0,
    max_value=80.0,
    value=40.0,
    step=2.0,
    help="Gross product margin before shipping and fulfillment costs."
)
gross_margin = gross_margin_pct / 100.0

forward_shipping = st.sidebar.slider(
    "Forward Shipping Cost (₹)",
    min_value=30.0,
    max_value=250.0,
    value=70.0,
    step=5.0,
    help="Outbound shipping cost charged by logistics carrier."
)

reverse_shipping = st.sidebar.slider(
    "Reverse Shipping Cost (₹)",
    min_value=40.0,
    max_value=300.0,
    value=90.0,
    step=5.0,
    help="Return freight fee incurred on undelivered/rejected COD packages."
)

packaging_cost = st.sidebar.slider(
    "Packaging & Handling Loss (₹)",
    min_value=10.0,
    max_value=150.0,
    value=40.0,
    step=5.0,
    help="Damaged packing, re-warehousing, and deadweight handling loss."
)

cac_penalty = st.sidebar.slider(
    "CAC Opportunity Cost (₹)",
    min_value=50.0,
    max_value=500.0,
    value=150.0,
    step=10.0,
    help="Customer acquisition cost lost when falsely blocking a legitimate buyer."
)

st.sidebar.divider()

# -------------------------------------------------------------
# DYNAMIC NET PROFIT THRESHOLD OPTIMIZATION
# -------------------------------------------------------------
if test_data is not None and engine is not None:
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    X_test_scaled = test_data["X_test_scaled"]

    # Calculate model probabilities on test set
    if engine.winner_name == "Logistic Regression":
        test_probs = engine.model.predict_proba(X_test_scaled)[:, 1]
    else:
        test_probs = engine.model.predict_proba(X_test)[:, 1]

    # Re-sweep thresholds based on Merchant Expected Net Profit
    thresholds = np.arange(0.10, 0.91, 0.05)
    sweep_results = []
    
    # Approximate mean order value from test set or standard ₹1,850
    avg_order_value = float(X_test["order_value"].mean()) if "order_value" in X_test.columns else 1850.0

    for t in thresholds:
        preds = (test_probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        # Profit Economics:
        # TN (Delivered COD): Earn gross margin - forward shipping
        profit_tn = tn * ((avg_order_value * gross_margin) - forward_shipping)
        # FP (Blocked good sale): Lost margin / CAC penalty
        loss_fp = fp * cac_penalty
        # FN (Undelivered RTO): Lost shipping + return freight + packaging
        loss_fn = fn * (forward_shipping + reverse_shipping + packaging_cost)
        # TP (Intercepted RTO converted to Prepaid with incentive):
        # We recover ~70% as prepaid sales with ₹50 incentive discount
        recovered_tp = tp * 0.70 * (((avg_order_value - 50.0) * gross_margin) - forward_shipping)

        net_profit = profit_tn + recovered_tp - loss_fp - loss_fn

        sweep_results.append({
            "threshold": round(t, 2),
            "net_profit": round(net_profit, 2),
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "TN": tn,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0)
        })

    sweep_df = pd.DataFrame(sweep_results)

    # Optimal Threshold is the point of Peak Net Profit
    optimal_idx = sweep_df["net_profit"].idxmax()
    optimal_threshold = float(sweep_df.iloc[optimal_idx]["threshold"])
    optimal_profit = float(sweep_df.iloc[optimal_idx]["net_profit"])

    # Default naive 0.50 cutoff profit
    default_row = sweep_df[sweep_df["threshold"] == 0.50].iloc[0]
    default_profit = float(default_row["net_profit"])
    profit_lift = optimal_profit - default_profit
    profit_lift_pct = (profit_lift / abs(default_profit) * 100) if default_profit != 0 else 0.0

    # Dynamically update optimal threshold on the engine for this session
    engine.optimal_threshold = optimal_threshold

    # Sidebar Optimal Card
    st.sidebar.markdown(f"#### 🎯 Profit-Maximizing Threshold")
    st.sidebar.markdown(
        f"<div class='threshold-card'>"
        f"  <p class='card-label'>OPTIMIZED CUTOFF</p>"
        f"  <p class='threshold-val'>{optimal_threshold:.2f}</p>"
        f"  <p class='card-label'>ESTIMATED PROFIT LIFT VS 0.50</p>"
        f"  <p class='savings-val'>+₹{profit_lift:,.2f} (+{profit_lift_pct:.1f}%)</p>"
        f"</div>",
        unsafe_allow_html=True
    )
else:
    optimal_threshold = 0.20
    optimal_profit = 0.0
    profit_lift = 0.0
    profit_lift_pct = 0.0

# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
st.markdown(
    f"<div class='header-container' style='position: relative;'>"
    f"  <div style='position: absolute; top: 24px; right: 24px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 6px 12px; font-size: 12px; color: #38BDF8; font-weight: 600;'>"
    f"    💻 Developer REST API: ACTIVE (Port 8000)"
    f"  </div>"
    f"  <div class='header-title'>🛡️ Razorpay RTO Risk-Ops & Profit Protection Engine</div>"
    f"  <p class='header-subtitle'>Autonomous Profit Arbitrage & Conversion Protection — Maximizing D2C Merchant GMV & Margins.</p>"
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
# TAB 1: SINGLE ORDER PROFIT & RISK SIMULATOR
# -------------------------------------------------------------
with tab_score:
    st.markdown("### Single Order Profit & Risk Simulator")
    
    # Grouped Inputs in 2-column layout
    st.markdown("##### 📍 Location Profile")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        pincode_tier = st.selectbox("Pincode Tier", ["Tier 1", "Tier 2", "Tier 3"], index=2)
    with row1_col2:
        pincode_rto_rate = st.slider("Pincode Historical Return Rate (%)", 3.0, 55.0, 22.0, step=1.0) / 100.0

    st.markdown("##### 💳 Order Details")
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        payment_mode = st.selectbox("Selected Checkout Payment Mode", ["COD", "Prepaid"], index=0)
    with row2_col2:
        order_value = st.number_input("Order Total Value (₹)", min_value=299, max_value=25000, value=2850)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        discount_pct = st.slider("Cart Discount Applied (%)", 0.0, 80.0, 15.0, step=1.0)
    with row3_col2:
        category = st.selectbox("Product Category", ["Apparel", "Footwear", "Beauty", "Electronics", "Home"], index=0)

    st.markdown("##### 🏠 Delivery Address Quality")
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        address_length = st.slider("Address Character Length", 10, 150, 45)
    with row4_col2:
        address_has_landmark = st.selectbox("Landmark Specified?", ["Yes", "No"], index=1)

    row5_col1, row5_col2 = st.columns(2)
    with row5_col1:
        pin_matches_city = st.selectbox("Pincode matches City?", ["Yes", "No (Address mismatch)"], index=0)
    with row5_col2:
        is_weekend_order = st.selectbox("Is Weekend Checkout?", ["No", "Yes"], index=0)

    st.markdown("##### 👤 Customer Purchase History")
    row6_col1, row6_col2 = st.columns(2)
    with row6_col1:
        customer_tenure_days = st.number_input("Customer Account Age (Days)", min_value=0, max_value=730, value=45)
    with row6_col2:
        customer_past_orders = st.number_input("Customer Past Orders Completed", min_value=0, max_value=100, value=2)

    row7_col1, row7_col2 = st.columns(2)
    with row7_col1:
        customer_past_rto_rate = st.slider("Customer Historical Return Rate (%)", 0.0, 100.0, 0.0, step=1.0) / 100.0
    with row7_col2:
        st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    score_btn = st.button("Analyze Profit & Risk Profile", use_container_width=True, type="primary")

    if score_btn:
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

        # 1. Run profit-arbitrage scoring
        verdict = engine.score_order(
            order=order_dict,
            gross_margin=gross_margin,
            forward_shipping=forward_shipping,
            reverse_shipping=reverse_shipping,
            packaging_cost=packaging_cost,
        )

        prob = verdict["risk_probability"]
        tier = verdict["risk_tier"]
        action = verdict["recommended_action"]
        action_payload = verdict["action_payload"]
        exp_profit_cod = verdict["expected_profit_cod"]
        exp_profit_prepaid = verdict["expected_profit_prepaid"]
        exp_profit_discount = verdict["expected_profit_prepaid_discount"]

        # 2. Get local SHAP explainers
        reasons = explainer.explain_prediction(order_dict)

        badge_class = f"pill-{action_payload.get('badge', 'green').lower()}"
        prob_color = action_payload.get("action_color", "#34D399")

        st.markdown("---")
        st.markdown("### Profit & Risk Decision Verdict")

        col_v1, col_v2 = st.columns([2, 3])

        with col_v1:
            override_html = ""
            if verdict.get("override_reason"):
                override_html = f"<div style='margin-top: 15px; padding: 10px; background-color: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; font-size: 11px; color: #FCA5A5;'>⚠️ <b>Fraud Override:</b> {verdict['override_reason']}</div>"

            # Result Card
            st.markdown(
                f"<div class='result-card'>"
                f"  <p style='margin:0; font-size:12px; color:#888888; font-weight:600; letter-spacing:0.05em;'>EXPECTED RTO PROBABILITY</p>"
                f"  <div style='display:flex; align-items:baseline; justify-content:space-between; margin:10px 0;'>"
                f"    <span style='font-size:38px; font-weight:800; color:{prob_color};'>{prob*100:.1f}%</span>"
                f"    <span class='pill-badge {badge_class}'>{tier} Risk ({action_payload.get('badge', 'GREEN')})</span>"
                f"  </div>"
                f"  <p style='margin:12px 0 0 0; font-size:12px; color:#888888; font-weight:600; letter-spacing:0.05em;'>PROFIT-MAXIMIZING ACTION</p>"
                f"  <p style='margin:2px 0 0 0; font-size:16px; font-weight:700; color:#FFFFFF;'>{action}</p>"
                f"  <div style='margin-top:12px; padding:12px; background:rgba(255,255,255,0.03); border:1px solid #222222; border-radius:6px; font-size:13px; color:#CCCCCC;'>"
                f"    👉 <b>Checkout Intervention:</b> {action_payload['display_message']}"
                f"  </div>"
                f"  {override_html}"
                f"</div>",
                unsafe_allow_html=True
            )

            # Profit Comparison Box
            st.markdown(
                f"<div class='result-card' style='padding: 18px;'>"
                f"  <p style='margin:0 0 10px 0; font-size:12px; color:#888888; font-weight:600;'>EXPECTED NET PROFIT ARBITRAGE</p>"
                f"  <div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1E293B;'>"
                f"    <span style='color:#CCCCCC; font-size:13px;'>Expected Profit (Fulfilled via COD):</span>"
                f"    <span style='font-weight:700; color:{'#34D399' if exp_profit_cod >= 0 else '#F87171'};'>₹{exp_profit_cod:,.2f}</span>"
                f"  </div>"
                f"  <div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1E293B;'>"
                f"    <span style='color:#CCCCCC; font-size:13px;'>Expected Profit (Prepaid Converted):</span>"
                f"    <span style='font-weight:700; color:#34D399;'>₹{exp_profit_prepaid:,.2f}</span>"
                f"  </div>"
                f"  <div style='display:flex; justify-content:space-between; padding:8px 0;'>"
                f"    <span style='color:#CCCCCC; font-size:13px;'>Expected Profit (Prepaid with ₹50 Discount):</span>"
                f"    <span style='font-weight:700; color:#38BDF8;'>₹{exp_profit_discount:,.2f}</span>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "RTO Probability Gauge", 'font': {'size': 13, 'color': '#888888'}},
                number = {'suffix': "%", 'font': {'color': '#FFFFFF'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#333333"},
                    'bar': {'color': prob_color},
                    'bgcolor': "#111111",
                    'borderwidth': 1,
                    'bordercolor': "#222222",
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [20, 40], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [40, 65], 'color': 'rgba(251, 146, 60, 0.15)'},
                        {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
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
                font={'color': "#FFFFFF", 'family': "Plus Jakarta Sans"},
                height=200,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_v2:
            st.markdown("##### 🔍 Financial Risk Attribution Drivers")
            st.markdown("<p style='font-size:12.5px; color:#888888;'>SHAP attributions showing features pulling risk upward (+Risk) vs downward (-Risk).</p>", unsafe_allow_html=True)

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
                    st.progress(min(1.0, max(0.0, abs(r["impact"]) / 0.50)))

# -------------------------------------------------------------
# TAB 2: MODEL OPTIMIZATION ANALYTICS
# -------------------------------------------------------------
with tab_analytics:
    st.markdown("### Profit Curve Optimization & Business Case Analysis")

    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        st.markdown("##### 📈 Net Merchant Profit (₹) vs Decision Cutoff Threshold")
        st.markdown("<p style='font-size:13px; color:#888888;'>Sweeping classification thresholds. Finding the exact cutoff that maximizes cumulative merchant earnings.</p>", unsafe_allow_html=True)

        if test_data is not None:
            fig_profit = go.Figure()

            # Net Profit Curve
            fig_profit.add_trace(go.Scatter(
                x=sweep_df["threshold"],
                y=sweep_df["net_profit"],
                mode='lines+markers',
                name='Net Merchant Profit (₹)',
                line=dict(color='#38BDF8', width=3),
                marker=dict(size=6)
            ))

            # Highlight Peak Profit Threshold
            fig_profit.add_trace(go.Scatter(
                x=[optimal_threshold],
                y=[optimal_profit],
                mode='markers',
                name='Peak Profit Threshold',
                marker=dict(color='#34D399', size=14, symbol='star', line=dict(color='white', width=1))
            ))

            # Highlight Naive 0.5 Cutoff
            fig_profit.add_trace(go.Scatter(
                x=[0.50],
                y=[default_profit],
                mode='markers',
                name='Naive 0.50 Cutoff',
                marker=dict(color='#EF4444', size=10, symbol='circle', line=dict(color='white', width=1))
            ))

            fig_profit.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#0A0A0A',
                font={'color': "#FFFFFF", 'family': "Plus Jakarta Sans"},
                xaxis=dict(title="Classification Cutoff Threshold", gridcolor="#222222"),
                yaxis=dict(title="Cumulative Portfolio Profit (₹)", gridcolor="#222222"),
                legend=dict(x=0.05, y=0.95, bgcolor='rgba(0,0,0,0.6)'),
                height=380,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            st.plotly_chart(fig_profit, use_container_width=True)
        else:
            st.warning("Test dataset not available.")

    with col_a2:
        st.markdown("##### 🎛️ Confusion Matrix & Financial Translation")
        st.markdown(f"<p style='font-size:13px; color:#888888;'>Classification performance translated to currency gained vs lost at threshold {optimal_threshold:.2f}.</p>", unsafe_allow_html=True)

        if test_data is not None:
            opt_preds = (test_probs >= optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, opt_preds).ravel()

            cm_data = [[tn, fp], [fn, tp]]
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
                height=260,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown(
                f"<div style='font-size:12.5px; color:#CCCCCC; padding:8px; background:#0A0A0A; border:1px solid #222222; border-radius:6px;'>"
                f"✅ <b>Delivered COD orders secured:</b> {tn} (Earned ₹{tn * ((avg_order_value * gross_margin) - forward_shipping):,.0f})<br>"
                f"🛡️ <b>RTO Returns intercepted:</b> {tp} (Prevented ₹{tp * (forward_shipping + reverse_shipping + packaging_cost):,.0f} freight waste)<br>"
                f"❌ <b>Unavoidable Returns:</b> {fn} | <b>False Alarms:</b> {fp}"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("##### 🏆 Model Benchmarking Table")
        st.markdown("<p style='font-size:13px; color:#888888;'>Held-out validation metrics across candidate architectures.</p>", unsafe_allow_html=True)
        
        bench_df = pd.DataFrame([
            {"Model": "XGBoost (Winner)", "Accuracy": "77.4%", "Precision": "60.4%", "Recall": "31.7%", "F1-Score": "41.6%", "AUC-ROC": "0.7955"},
            {"Model": "Logistic Regression", "Accuracy": "76.4%", "Precision": "55.9%", "Recall": "32.7%", "F1-Score": "41.3%", "AUC-ROC": "0.7809"},
            {"Model": "Random Forest", "Accuracy": "75.6%", "Precision": "55.0%", "Recall": "21.8%", "F1-Score": "31.2%", "AUC-ROC": "0.7786"}
        ])
        st.dataframe(bench_df, hide_index=True, use_container_width=True)

    with col_b2:
        st.markdown("##### 🌍 Global Feature Importances (SHAP)")
        st.markdown("<p style='font-size:13px; color:#888888;'>Key drivers of return behavior across the transaction database.</p>", unsafe_allow_html=True)
        importance_img_path = os.path.join(os.path.dirname(__file__), "..", "models", "global_importance.png")
        if os.path.exists(importance_img_path):
            st.image(importance_img_path, caption="SHAP Summary Plot for the Winning XGBoost Model", use_container_width=True)

# -------------------------------------------------------------
# TAB 3: BATCH AUDIT & PORTFOLIO PROFIT ANALYSIS
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Audit & Portfolio Profit Analysis")
    st.markdown("<p style='font-size:13px; color:#888888;'>Upload bulk orders to simulate automated profit-arbitrage routing across your entire order pipeline.</p>", unsafe_allow_html=True)

    # Template schema
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

            # Defensive Numeric Coercion
            categorical_cols = {"pincode_tier", "payment_mode", "category"}
            for col in input_df.columns:
                if col not in categorical_cols:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

            # Run Scoring with dynamic sandbox economics
            scored_df = engine.score_batch(
                df_input=input_df,
                gross_margin=gross_margin,
                forward_shipping=forward_shipping,
                reverse_shipping=reverse_shipping,
                packaging_cost=packaging_cost,
            )

            total_orders = len(scored_df)
            total_gmv = scored_df["order_value"].sum()

            # Protected Net Revenue: Delivered orders gross profit
            auto_ship_orders = scored_df[scored_df["action_code"] == "AUTO_SHIP"]
            protected_revenue = auto_ship_orders["expected_profit_cod"].sum()

            # Saved Logistics Waste: Intercepted high-risk COD orders
            intercepted_orders = scored_df[scored_df["action_code"].isin(["INCENTIVIZE_PREPAID", "STRICT_PREPAID_ONLY"])]
            saved_waste = len(intercepted_orders) * (forward_shipping + reverse_shipping + packaging_cost)

            # Recovered GMV via Prepaid Conversion: Revenue from intercepted orders converted to prepaid
            recovered_gmv = intercepted_orders["order_value"].sum() * 0.70

            # Net Profit Lift:
            total_baseline_profit = scored_df["expected_profit_cod"].sum()
            # With our interventions: auto_ship keeps COD profit, intercepted converts to prepaid profit
            total_optimized_profit = auto_ship_orders["expected_profit_cod"].sum() + (intercepted_orders["expected_profit_prepaid"].sum() * 0.70)
            net_lift = total_optimized_profit - total_baseline_profit
            net_lift_pct = (net_lift / abs(total_baseline_profit) * 100) if total_baseline_profit != 0 else 0.0

            # 4 Executive KPI Cards
            k_col1, k_col2, k_col3, k_col4 = st.columns(4)

            with k_col1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>Protected Net Revenue</div>"
                    f"  <div class='metric-value' style='color:#38BDF8;'>₹{protected_revenue:,.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with k_col2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>Saved Logistics Waste</div>"
                    f"  <div class='metric-value' style='color:#34D399;'>₹{saved_waste:,.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with k_col3:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>Recovered Prepaid GMV</div>"
                    f"  <div class='metric-value' style='color:#FBBF24;'>₹{recovered_gmv:,.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with k_col4:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"  <div class='metric-title'>Net Profit Lift</div>"
                    f"  <div class='metric-value' style='color:{'#34D399' if net_lift >= 0 else '#F87171'};'>+₹{net_lift:,.2f} (+{net_lift_pct:.1f}%)</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Enriched Profit & Risk Action Plan")

            view_df = scored_df[[
                "order_value", "risk_probability", "risk_tier", "action_code", 
                "expected_profit_cod", "expected_profit_prepaid", "recommended_action", "checkout_display_message"
            ]].copy()
            view_df["risk_probability"] = (view_df["risk_probability"] * 100).round(1).astype(str) + "%"

            st.dataframe(view_df, use_container_width=True)

            st.download_button(
                "📥 Download Enriched Profit & Risk Action Plan (CSV)",
                data=scored_df.to_csv(index=False).encode('utf-8'),
                file_name="enriched_profit_risk_plan.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            st.info("Please verify the CSV format against the template schema.")
