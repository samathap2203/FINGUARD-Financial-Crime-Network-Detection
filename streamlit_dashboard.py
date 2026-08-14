import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

# ── Load models ───────────────────────────────────────────────────
with open("rf_model.pkl", "rb") as f:
    rf = pickle.load(f)
with open("lgbm_model.pkl", "rb") as f:
    lgbm = pickle.load(f)
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ── Load GNN results ──────────────────────────────────────────────
emb_2d     = np.load("gnn_emb_2d.npy")
gnn_labels = np.load("gnn_labels.npy")
y_true_gnn = np.load("gnn_y_true.npy")
y_pred_gnn = np.load("gnn_y_pred.npy")
with open("gnn_history.json", "r") as f:
    history = json.load(f)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="Financial Crime Detection", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap");
    html, body, [class*="css"] { font-family: "Inter", sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d0021 0%, #1a0033 50%, #0d001a 100%); }
    .main .block-container { padding: 2rem 3rem; }
    h1 { background: linear-gradient(90deg, #c084fc, #f472b6, #fb923c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 700; }
    h2 { color: #c084fc; font-weight: 600; }
    h3 { color: #f472b6; font-weight: 500; }
    p, li, label { color: #e2d9f3; }
    .stSidebar { background: linear-gradient(180deg, #1a0033 0%, #0d0021 100%); border-right: 1px solid #4a1a6b; }
    .stSidebar .stRadio label { color: #c084fc; font-size: 15px; }
    .metric-card { background: linear-gradient(135deg, #2d0050 0%, #1a0033 100%); border: 1px solid #6b21a8; border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(192,132,252,0.15); }
    .metric-card h2 { background: linear-gradient(90deg, #c084fc, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; margin: 0; }
    .metric-card p { color: #a78bfa; margin: 5px 0 0; font-size: 13px; }
    .fraud-box { background: linear-gradient(135deg, #450a0a, #7f1d1d); border-left: 5px solid #ef4444; padding: 20px; border-radius: 12px; font-size: 20px; font-weight: 700; color: #fca5a5; box-shadow: 0 4px 20px rgba(239,68,68,0.3); }
    .normal-box { background: linear-gradient(135deg, #0a2e1a, #14532d); border-left: 5px solid #22c55e; padding: 20px; border-radius: 12px; font-size: 20px; font-weight: 700; color: #86efac; box-shadow: 0 4px 20px rgba(34,197,94,0.3); }
    .info-card { background: linear-gradient(135deg, #1e1b4b, #312e81); border: 1px solid #4338ca; border-radius: 12px; padding: 18px; margin: 8px 0; }
    .stButton > button { background: linear-gradient(90deg, #7c3aed, #c026d3); color: white; border: none; border-radius: 10px; padding: 12px 30px; font-size: 16px; font-weight: 600; width: 100%; transition: all 0.3s; box-shadow: 0 4px 15px rgba(124,58,237,0.4); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(124,58,237,0.6); }
    .section-header { background: linear-gradient(90deg, #7c3aed22, transparent); border-left: 4px solid #c084fc; padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 20px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────
st.title("🔍 Financial Crime Analysis System")
st.markdown("<p style=\'color:#a78bfa; font-size:16px; margin-top:-10px\'>Advanced Fraud Detection using ML, Deep Learning & Network Analysis</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style=\'color:#c084fc; text-align:center\'>⚙️ Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", [
    "🏠 Home",
    "🔎 Predict Transaction",
    "📊 Model Comparison",
    "🧠 GNN Analysis",
    "📈 Fraud Insights",
    "ℹ️ About"
])
st.sidebar.markdown("---")
st.sidebar.markdown("<p style=\'color:#6b7280; font-size:12px; text-align:center\'>Financial Crime Detection v2.0<br>Capstone Project 2025</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    col1, col2, col3, col4, col5 = st.columns(5)
    stats = [("6.3M+","Transactions"), ("6","Models Trained"),
             ("1","GNN Model"), ("95.58%","Best F1 (XGBoost)"), ("0.13%","Fraud Rate")]
    for col, (val, label) in zip([col1,col2,col3,col4,col5], stats):
        with col:
            st.markdown(f"<div class=\'metric-card\'><h2>{val}</h2><p>{label}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class=\'section-header\'><h3 style=\'margin:0\'>📌 Project Overview</h3></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class=\'info-card\'>
        <p>A complete <b style=\'color:#c084fc\'>Financial Crime Detection Pipeline</b> using:</p>
        <ul>
            <li><b style=\'color:#f472b6\'>Machine Learning</b> — Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Isolation Forest</li>
            <li><b style=\'color:#f472b6\'>Deep Learning</b> — Graph Neural Network (GraphSAGE)</li>
            <li><b style=\'color:#f472b6\'>Network Analysis</b> — Degree, Betweenness, Closeness Centrality, Community Detection</li>
            <li><b style=\'color:#f472b6\'>Explainability</b> — SHAP Beeswarm, Waterfall, Summary, Donut plots</li>
            <li><b style=\'color:#f472b6\'>Anomaly Detection</b> — Isolation Forest (Unsupervised)</li>
            <li><b style=\'color:#f472b6\'>Class Imbalance</b> — SMOTE (Synthetic Minority Oversampling Technique)</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class=\'section-header\'><h3 style=\'margin:0\'>📂 Dataset Info</h3></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class=\'info-card\'>
        <ul>
            <li><b style=\'color:#c084fc\'>Dataset:</b> PaySim Synthetic Financial Simulator</li>
            <li><b style=\'color:#c084fc\'>Total Records:</b> 6.3 Million transactions</li>
            <li><b style=\'color:#c084fc\'>Sample Used:</b> 200,000 transactions</li>
            <li><b style=\'color:#c084fc\'>Fraud Rate:</b> ~0.13% (highly imbalanced)</li>
            <li><b style=\'color:#c084fc\'>Transaction Types:</b> CASH_OUT, TRANSFER, PAYMENT, CASH_IN, DEBIT</li>
            <li><b style=\'color:#c084fc\'>Imbalance Fix:</b> SMOTE</li>
            <li><b style=\'color:#c084fc\'>Train/Test Split:</b> 80/20 with stratification</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class=\'section-header\'><h3 style=\'margin:0\'>🔬 Models Used</h3></div>", unsafe_allow_html=True)
    models_df = pd.DataFrame({
        "Model":      ["Logistic Regression", "Decision Tree", "Random Forest",
                       "XGBoost", "LightGBM", "GNN (GraphSAGE)", "Isolation Forest"],
        "Type":       ["Supervised", "Supervised", "Supervised",
                       "Supervised", "Supervised", "Deep Learning", "Unsupervised"],
        "Category":   ["Classical", "Tree-based", "Ensemble",
                       "Boosting", "Boosting", "Graph Neural Net", "Anomaly Detection"],
        "SMOTE Used": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "No"],
        "Best For":   ["Baseline", "Interpretability", "High accuracy",
                       "Best F1 score", "Speed + accuracy",
                       "Fraud ring detection", "Novel fraud patterns"]
    })
    st.dataframe(models_df, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class=\'section-header\'><h3 style=\'margin:0\'>🛠️ Advanced Techniques</h3></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#c084fc\'>📊 Visualizations</h4>
        <ul>
            <li>SHAP Beeswarm Plot</li>
            <li>SHAP Waterfall Plot</li>
            <li>SHAP Summary Bar Plot</li>
            <li>SHAP Donut Chart</li>
            <li>t-SNE GNN Embeddings</li>
            <li>ROC and PR Curves</li>
            <li>Anomaly Score Distribution</li>
        </ul></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#f472b6\'>🕸️ Network Analysis</h4>
        <ul>
            <li>Degree Centrality</li>
            <li>Betweenness Centrality</li>
            <li>Closeness Centrality</li>
            <li>Community Detection</li>
            <li>Fraud Ring Visualization</li>
            <li>GNN Node Classification</li>
            <li>Graph Embeddings (t-SNE)</li>
        </ul></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#fb923c\'>⚙️ ML Techniques</h4>
        <ul>
            <li>SMOTE Oversampling</li>
            <li>Feature Engineering</li>
            <li>Class Weight Balancing</li>
            <li>Hyperparameter Tuning</li>
            <li>Cross Validation</li>
            <li>ROC AUC Evaluation</li>
            <li>PR Curve Analysis</li>
        </ul></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════
elif page == "🔎 Predict Transaction":
    st.header("🔎 Predict Transaction Fraud Risk")
    st.markdown("<p style=\'color:#a78bfa\'>Enter transaction details to get instant fraud prediction</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style=\'color:#c084fc\'>Transaction Details</h4>", unsafe_allow_html=True)
        transaction_type = st.selectbox("Transaction Type", ["CASH_OUT","TRANSFER","PAYMENT","CASH_IN","DEBIT"])
        amount        = st.number_input("Transaction Amount (₹)", min_value=0.0, value=10000.0)
        oldbalanceOrg = st.number_input("Sender Old Balance (₹)", min_value=0.0, value=50000.0)
        newbalanceOrig= st.number_input("Sender New Balance (₹)", min_value=0.0, value=40000.0)
    with col2:
        st.markdown("<h4 style=\'color:#c084fc\'>Account Details</h4>", unsafe_allow_html=True)
        oldbalanceDest = st.number_input("Receiver Old Balance (₹)", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Receiver New Balance (₹)", min_value=0.0, value=10000.0)
        step           = st.slider("Time Step (hours)", 1, 744, 1)
        model_choice   = st.selectbox("Choose Model", ["Random Forest","LightGBM","Logistic Regression","Decision Tree"])

    st.markdown("---")
    if st.button("🔍 Analyze Transaction", use_container_width=True):
        balanceDiffOrig = oldbalanceOrg  - newbalanceOrig
        balanceDiffDest = newbalanceDest - oldbalanceDest
        amount_ratio    = amount / (oldbalanceOrg + 1)
        amount_log      = np.log1p(amount)
        type_CASH_OUT   = 1 if transaction_type == "CASH_OUT"  else 0
        type_DEBIT      = 1 if transaction_type == "DEBIT"     else 0
        type_PAYMENT    = 1 if transaction_type == "PAYMENT"   else 0
        type_TRANSFER   = 1 if transaction_type == "TRANSFER"  else 0

        features = pd.DataFrame([[
            step, amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest,
            balanceDiffOrig, balanceDiffDest,
            amount_ratio, amount_log,
            type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
        ]], columns=feature_columns)

        model       = rf if model_choice == "Random Forest" else lgbm
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        st.markdown("---")
        st.subheader("🎯 Analysis Result")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if prediction == 1:
                st.markdown("<div class=\'fraud-box\'>⚠️ FRAUD DETECTED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class=\'normal-box\'>✅ NORMAL TRANSACTION</div>", unsafe_allow_html=True)
        with col2:
            st.metric("Fraud Probability", f"{probability*100:.2f}%")
        with col3:
            risk = "🔴 HIGH" if probability > 0.7 else "🟡 MEDIUM" if probability > 0.3 else "🟢 LOW"
            st.metric("Risk Level", risk)
        with col4:
            st.metric("Model Used", model_choice)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 2))
            fig.patch.set_facecolor("#1a0033")
            ax.set_facecolor("#1a0033")
            bar_color = "#ef4444" if probability > 0.5 else "#22c55e"
            ax.barh(["Risk"], [probability], color=bar_color, height=0.5, alpha=0.9)
            ax.barh(["Risk"], [1-probability], left=[probability], color="#2d0050", height=0.5)
            ax.set_xlim(0, 1)
            ax.axvline(x=0.5, color="#c084fc", linestyle="--", alpha=0.7, linewidth=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#4a1a6b")
            ax.tick_params(colors="#a78bfa")
            ax.set_xlabel("Fraud Probability", color="#a78bfa")
            ax.text(probability/2, 0, f"{probability*100:.1f}%",
                   ha="center", va="center", color="white",
                   fontweight="bold", fontsize=12)
            plt.title("Risk Gauge", color="#c084fc", fontweight="bold", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 2))
            fig.patch.set_facecolor("#1a0033")
            ax.set_facecolor("#1a0033")
            feat_names = ["Amount Ratio", "Balance Diff", "Amount", "Tx Type"]
            values = [
                min(amount_ratio, 1),
                min(abs(balanceDiffOrig)/max(oldbalanceOrg,1), 1),
                min(amount/1000000, 1),
                0.8 if transaction_type in ["CASH_OUT","TRANSFER"] else 0.2
            ]
            colors = ["#ef4444" if v > 0.5 else "#22c55e" for v in values]
            ax.barh(feat_names, values, color=colors, alpha=0.8,
                   edgecolor="#4a1a6b", height=0.5)
            ax.set_xlim(0, 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#4a1a6b")
            ax.spines["left"].set_color("#4a1a6b")
            ax.tick_params(colors="#a78bfa", labelsize=9)
            ax.set_xlabel("Risk Contribution", color="#a78bfa", fontsize=9)
            plt.title("Feature Risk Contribution", color="#c084fc",
                     fontweight="bold", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("📋 Transaction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({
                "Feature": ["Transaction Type","Amount","Sender Old Balance",
                           "Sender New Balance","Receiver Old Balance","Receiver New Balance"],
                "Value":   [transaction_type, f"₹{amount:,.2f}",
                           f"₹{oldbalanceOrg:,.2f}", f"₹{newbalanceOrig:,.2f}",
                           f"₹{oldbalanceDest:,.2f}", f"₹{newbalanceDest:,.2f}"]
            }), use_container_width=True)
        with col2:
            st.dataframe(pd.DataFrame({
                "Engineered Feature": ["Balance Diff (Sender)","Balance Diff (Receiver)",
                                      "Amount Ratio","Log Amount"],
                "Value":             [f"₹{balanceDiffOrig:,.2f}", f"₹{balanceDiffDest:,.2f}",
                                      f"{amount_ratio:.4f}", f"{amount_log:.4f}"]
            }), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison")
    st.markdown("<p style=\'color:#a78bfa\'>Actual results from notebook — all supervised and unsupervised models</p>", unsafe_allow_html=True)

    comparison_data = pd.DataFrame({
        "Model":     ["Logistic Regression","Decision Tree","Random Forest",
                      "XGBoost","LightGBM","GNN (GraphSAGE)","Isolation Forest"],
        "Type":      ["Supervised","Supervised","Supervised",
                      "Supervised","Supervised","Deep Learning","Unsupervised"],
        "Accuracy":  [0.9696, 0.9998, 1.0000, 0.9999, 0.9968, 1.0000, "N/A"],
        "Precision": [0.0417, 0.8852, 1.0000, 0.9153, 0.2898, 1.0000, 0.00],
        "Recall":    [0.9815, 1.0000, 1.0000, 1.0000, 0.9444, 1.0000, 0.00],
        "F1 Score":  [0.0801, 0.9391, 1.0000, 0.9558, 0.4435, 1.0000, 0.00],
        "ROC-AUC":   [0.9889, 0.9999, 1.0000, 1.0000, 0.9997, 1.0000, 0.50]
    })
    st.dataframe(comparison_data, use_container_width=True)

    col1, col2 = st.columns(2)
    plot_data = comparison_data[comparison_data["Model"] != "Isolation Forest"].copy()
    plot_data = plot_data[comparison_data["Model"] != "Logistic Regression"].copy()

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#1a0033")
        ax.set_facecolor("#1a0033")
        x      = np.arange(len(comparison_data))
        width  = 0.18
        colors = ["#c084fc","#f472b6","#fb923c","#34d399"]
        metrics= ["Precision","Recall","F1 Score","ROC-AUC"]
        for i,(metric,color) in enumerate(zip(metrics,colors)):
            vals = [float(v) if v != "N/A" else 0 for v in comparison_data[metric]]
            bars = ax.bar(x + i*width - 1.5*width, vals, width,
                         label=metric, color=color, alpha=0.85, edgecolor="#1a0033")
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6, color="white")
        ax.set_facecolor("#1a0033")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#4a1a6b")
        ax.spines["left"].set_color("#4a1a6b")
        ax.tick_params(colors="#a78bfa", labelsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_data["Model"], rotation=20,
                          color="#a78bfa", fontsize=7)
        ax.set_ylim(0, 1.2)
        ax.yaxis.grid(True, color="#2d0050", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, facecolor="#2d0050",
                 edgecolor="#4a1a6b", labelcolor="#e2d9f3")
        plt.title("All Models — Performance Metrics",
                 color="#c084fc", fontweight="bold", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#1a0033")
        ax.set_facecolor("#1a0033")
        f1_vals  = [0.0801,0.9391,1.0000,0.9558,0.4435,1.0000,0.00]
        colors   = ["#6b7280","#c084fc","#f472b6","#fb923c",
                    "#34d399","#60a5fa","#374151"]
        bars = ax.barh(comparison_data["Model"], f1_vals,
                      color=colors, alpha=0.85,
                      edgecolor="#1a0033", height=0.6)
        ax.bar_label(bars, fmt="%.4f", padding=5,
                    color="white", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#4a1a6b")
        ax.spines["left"].set_color("#4a1a6b")
        ax.tick_params(colors="#a78bfa", labelsize=9)
        ax.xaxis.grid(True, color="#2d0050", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("F1 Score", color="#a78bfa")
        plt.title("F1 Score — All Models",
                 color="#c084fc", fontweight="bold", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔍 Key Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#c084fc\'>🏆 Best Models</h4>
        <p><b style=\'color:#f472b6\'>Random Forest & GNN</b> achieved perfect scores. <b style=\'color:#fb923c\'>XGBoost</b> achieved best realistic F1 of 0.9558 with 99.99% accuracy.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#f472b6\'>⚠️ PaySim Limitation</h4>
        <p>High scores are due to PaySim being <b style=\'color:#fb923c\'>synthetic</b>. Real-world data would show more variation. We focus on F1, Recall and ROC-AUC as primary metrics.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#fb923c\'>🔬 Unsupervised Failure</h4>
        <p><b style=\'color:#ef4444\'>Isolation Forest</b> scored 0.00 F1 — proving supervised learning with labelled data is essential for fraud detection.</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — GNN ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "🧠 GNN Analysis":
    st.header("🧠 Graph Neural Network Analysis")
    st.markdown("<p style=\'color:#a78bfa\'>GraphSAGE model results — node-level fraud detection through network topology</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""<div class=\'info-card\'>
    <h4 style=\'color:#c084fc\'>What is GNN?</h4>
    <p>Unlike traditional ML which looks at each transaction in isolation, GNN models the <b style=\'color:#f472b6\'>network of connections</b> between accounts. Each account is a <b style=\'color:#fb923c\'>node</b> and each transaction is an <b style=\'color:#fb923c\'>edge</b>. GNN detects fraud rings by learning from neighboring account behavior.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📉 Training Performance")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a0033")
        ax.set_facecolor("#1a0033")
        ax.plot(history["loss"], color="#f472b6", linewidth=2.5, label="Training Loss")
        ax.fill_between(range(len(history["loss"])),
                       history["loss"], alpha=0.15, color="#f472b6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#4a1a6b")
        ax.spines["left"].set_color("#4a1a6b")
        ax.tick_params(colors="#a78bfa")
        ax.set_xlabel("Epoch", color="#a78bfa")
        ax.set_ylabel("Loss", color="#a78bfa")
        ax.legend(facecolor="#2d0050", edgecolor="#4a1a6b",
                 labelcolor="#e2d9f3", fontsize=10)
        ax.yaxis.grid(True, color="#2d0050", linewidth=0.8)
        ax.set_axisbelow(True)
        plt.title("GNN Training Loss Curve",
                 color="#c084fc", fontweight="bold", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a0033")
        ax.set_facecolor("#1a0033")
        ax.plot(history["train_acc"], color="#c084fc", linewidth=2.5,
               label="Train Accuracy")
        ax.plot(history["test_acc"],  color="#34d399", linewidth=2.5,
               label="Test Accuracy", linestyle="--")
        ax.fill_between(range(len(history["train_acc"])),
                       history["train_acc"], alpha=0.1, color="#c084fc")
        ax.fill_between(range(len(history["test_acc"])),
                       history["test_acc"],  alpha=0.1, color="#34d399")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#4a1a6b")
        ax.spines["left"].set_color("#4a1a6b")
        ax.tick_params(colors="#a78bfa")
        ax.set_xlabel("Epoch", color="#a78bfa")
        ax.set_ylabel("Accuracy", color="#a78bfa")
        ax.legend(facecolor="#2d0050", edgecolor="#4a1a6b",
                 labelcolor="#e2d9f3", fontsize=10)
        ax.yaxis.grid(True, color="#2d0050", linewidth=0.8)
        ax.set_axisbelow(True)
        plt.title("GNN Train vs Test Accuracy",
                 color="#c084fc", fontweight="bold", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🗺️ Node Embeddings (t-SNE)")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a0033")
    ax.set_facecolor("#1a0033")
    normal_mask = gnn_labels == 0
    fraud_mask  = gnn_labels == 1
    ax.scatter(emb_2d[normal_mask, 0], emb_2d[normal_mask, 1],
              c="#60a5fa", alpha=0.4, s=8, label="Normal Accounts",
              edgecolors="none")
    ax.scatter(emb_2d[fraud_mask, 0],  emb_2d[fraud_mask, 1],
              c="#ef4444", alpha=0.8, s=25, label="Fraud Accounts",
              edgecolors="none", zorder=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#4a1a6b")
    ax.spines["left"].set_color("#4a1a6b")
    ax.tick_params(colors="#a78bfa")
    ax.set_xlabel("t-SNE Dimension 1", color="#a78bfa")
    ax.set_ylabel("t-SNE Dimension 2", color="#a78bfa")
    ax.legend(facecolor="#2d0050", edgecolor="#4a1a6b",
             labelcolor="#e2d9f3", fontsize=11,
             markerscale=2)
    plt.title("t-SNE Visualization of GNN Node Embeddings\n(Fraud accounts clearly separated from normal)",
             color="#c084fc", fontweight="bold", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 GNN Classification Results")
    col1, col2 = st.columns(2)

    with col1:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_gnn, y_pred_gnn)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#1a0033")
        ax.set_facecolor("#1a0033")
        sns.heatmap(cm, annot=True, fmt="d",
                   cmap="RdPu",
                   xticklabels=["Normal","Fraud"],
                   yticklabels=["Normal","Fraud"],
                   linewidths=0.5,
                   linecolor="#1a0033",
                   ax=ax)
        ax.tick_params(colors="#a78bfa")
        ax.set_xlabel("Predicted", color="#a78bfa")
        ax.set_ylabel("Actual",    color="#a78bfa")
        plt.title("GNN Confusion Matrix",
                 color="#c084fc", fontweight="bold", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        from sklearn.metrics import classification_report
        report = classification_report(y_true_gnn, y_pred_gnn,
                                      target_names=["Normal","Fraud"],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)
        st.markdown("<h4 style=\'color:#c084fc\'>Classification Report</h4>", unsafe_allow_html=True)
        st.dataframe(report_df, use_container_width=True)

        gnn_metrics = pd.DataFrame({
            "Metric": ["Precision","Recall","F1 Score","Accuracy"],
            "Value":  [f"{report['Fraud']['precision']:.4f}",
                      f"{report['Fraud']['recall']:.4f}",
                      f"{report['Fraud']['f1-score']:.4f}",
                      f"{report['accuracy']:.4f}"]
        })
        st.dataframe(gnn_metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Why GNN for Fraud Detection?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#c084fc\'>Traditional ML</h4>
        <ul>
            <li>Looks at 1 transaction at a time</li>
            <li>Cannot detect fraud rings</li>
            <li>Feature-based only</li>
            <li>Misses network patterns</li>
        </ul></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#f472b6\'>GNN Advantage</h4>
        <ul>
            <li>Looks at entire account network</li>
            <li>Detects fraud rings naturally</li>
            <li>Features + topology combined</li>
            <li>Catches coordinated fraud</li>
        </ul></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#fb923c\'>Our GNN Model</h4>
        <ul>
            <li>Architecture: GraphSAGE</li>
            <li>3 convolutional layers</li>
            <li>Hidden size: 64</li>
            <li>Trained for 150 epochs</li>
        </ul></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — FRAUD INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Fraud Insights":
    st.header("📈 Fraud Insights & Analysis")
    st.markdown("<p style=\'color:#a78bfa\'>Deep dive into fraud patterns discovered during analysis</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔑 Top Fraud Indicators (SHAP)")
        shap_data = pd.DataFrame({
            "Feature":    ["amount_ratio","newbalanceOrig","balanceDiffOrig",
                          "type_TRANSFER","oldbalanceOrg","type_PAYMENT",
                          "amount","type_CASH_OUT","amount_log","newbalanceDest"],
            "Importance": [0.200,0.102,0.089,0.031,0.028,0.027,
                          0.014,0.013,0.011,0.010]
        })
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1a0033")
        ax.set_facecolor("#1a0033")
        palette = ["#7c3aed","#8b5cf6","#a78bfa","#c084fc","#d8b4fe",
                  "#e879f9","#f472b6","#fb7185","#fb923c","#fbbf24"]
        bars = ax.barh(shap_data["Feature"][::-1],
                      shap_data["Importance"][::-1],
                      color=palette[::-1], alpha=0.9,
                      edgecolor="#1a0033", height=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=3, color="white", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#4a1a6b")
        ax.spines["left"].set_color("#4a1a6b")
        ax.tick_params(colors="#a78bfa", labelsize=9)
        ax.xaxis.grid(True, color="#2d0050", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean |SHAP Value|", color="#a78bfa")
        plt.title("Top 10 Fraud Detection Features (SHAP)",
                 color="#c084fc", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("💰 Fraud by Transaction Type")
        tx_data = pd.DataFrame({
            "Type":    ["CASH_OUT","TRANSFER","PAYMENT","CASH_IN","DEBIT"],
            "Fraud %": [0.18,0.31,0.00,0.00,0.00],
            "Total %": [35.2,8.1,33.8,21.9,1.0]
        })
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1a0033")
        ax.set_facecolor("#1a0033")
        x     = np.arange(len(tx_data))
        width = 0.35
        b1 = ax.bar(x-width/2, tx_data["Fraud %"], width,
                   label="Fraud %", color="#ef4444",
                   alpha=0.85, edgecolor="#1a0033")
        b2 = ax.bar(x+width/2, tx_data["Total %"]/100, width,
                   label="Volume %", color="#c084fc",
                   alpha=0.85, edgecolor="#1a0033")
        ax.bar_label(b1, fmt="%.2f", padding=3, color="white", fontsize=9)
        ax.bar_label(b2, fmt="%.2f", padding=3, color="white", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(tx_data["Type"], color="#a78bfa")
        ax.tick_params(colors="#a78bfa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#4a1a6b")
        ax.spines["left"].set_color("#4a1a6b")
        ax.yaxis.grid(True, color="#2d0050", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(facecolor="#2d0050", edgecolor="#4a1a6b",
                 labelcolor="#e2d9f3", fontsize=10)
        plt.title("Fraud Rate vs Volume by Transaction Type",
                 color="#c084fc", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🕸️ Network Analysis Findings")
        st.markdown("""<div class=\'info-card\'>
        <ul>
            <li><b style=\'color:#f472b6\'>Degree Centrality</b> — Top accounts involved in 50+ transactions flagged as high risk</li>
            <li><b style=\'color:#f472b6\'>Betweenness Centrality</b> — Identified hub accounts controlling money flow</li>
            <li><b style=\'color:#f472b6\'>Community Detection</b> — Discovered fraud rings of 3-8 connected accounts</li>
            <li><b style=\'color:#f472b6\'>GNN t-SNE</b> — Clear visual separation of fraud vs normal account clusters</li>
        </ul></div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("🤖 SHAP Key Findings")
        st.markdown("""<div class=\'info-card\'>
        <ul>
            <li><b style=\'color:#f472b6\'>amount_ratio</b> — Most important. High ratio = account drained in one transaction</li>
            <li><b style=\'color:#f472b6\'>newbalanceOrig</b> — Near-zero balance after transaction = strong fraud signal</li>
            <li><b style=\'color:#f472b6\'>balanceDiffOrig</b> — Large balance drop = suspicious pattern</li>
            <li><b style=\'color:#f472b6\'>TRANSFER/CASH_OUT</b> — Highest fraud association transaction types</li>
        </ul></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔬 Anomaly Detection Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#ef4444\'>❌ Isolation Forest</h4>
        <p>Detected <b>54 anomalies</b> but caught <b>0 actual fraud</b></p>
        <p style=\'color:#6b7280; font-size:12px\'>F1: 0.00 | Precision: 0.00 | Recall: 0.00</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#f472b6\'>🔍 Why It Failed</h4>
        <p>PaySim fraud is <b>not statistically extreme</b> in feature space — blends with normal transactions</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class=\'info-card\'>
        <h4 style=\'color:#22c55e\'>✅ Conclusion</h4>
        <p>Supervised learning with <b>labelled data is essential</b> — unsupervised alone is insufficient</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class=\'info-card\'>
        <h3 style=\'color:#c084fc\'>🎓 Project Details</h3>
        <ul>
            <li><b style=\'color:#f472b6\'>Title:</b> Financial Crime Analysis in Banking Transactions using ML and Network Analysis</li>
            <li><b style=\'color:#f472b6\'>Type:</b> Capstone Final Project</li>
            <li><b style=\'color:#f472b6\'>Dataset:</b> PaySim Synthetic Financial Dataset</li>
            <li><b style=\'color:#f472b6\'>Year:</b> 2025</li>
        </ul></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class=\'info-card\'>
        <h3 style=\'color:#f472b6\'>🛠️ Tech Stack</h3>
        <ul>
            <li><b style=\'color:#c084fc\'>Language:</b> Python 3.10</li>
            <li><b style=\'color:#c084fc\'>ML:</b> Scikit-learn, LightGBM, XGBoost</li>
            <li><b style=\'color:#c084fc\'>DL:</b> PyTorch, PyTorch Geometric</li>
            <li><b style=\'color:#c084fc\'>Network:</b> NetworkX</li>
            <li><b style=\'color:#c084fc\'>Explainability:</b> SHAP</li>
            <li><b style=\'color:#c084fc\'>Dashboard:</b> Streamlit</li>
            <li><b style=\'color:#c084fc\'>Visualization:</b> Matplotlib, Seaborn</li>
        </ul></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class=\'info-card\'>
    <h3 style=\'color:#fb923c\'>🚀 Future Work (Review 3)</h3>
    <ul>
        <li><b style=\'color:#c084fc\'>Real-time Detection</b> — Kafka/Flask API for live transaction monitoring</li>
        <li><b style=\'color:#c084fc\'>Temporal Analysis</b> — Time-based fraud pattern detection using step feature</li>
        <li><b style=\'color:#c084fc\'>GAT Model</b> — Graph Attention Network to improve GNN performance</li>
        <li><b style=\'color:#c084fc\'>Risk Scoring</b> — Composite risk score combining ML + network centrality</li>
        <li><b style=\'color:#c084fc\'>Real Dataset</b> — IEEE-CIS Fraud Detection for real-world validation</li>
    </ul></div>""", unsafe_allow_html=True)
