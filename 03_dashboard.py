import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="MCDM Model Selector", layout="wide", page_icon="🏆")

# Helper functions for calculations
def run_topsis_dynamic(df, weights, impacts):
    matrix = df.to_numpy(dtype=float)
    m, n = matrix.shape
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    
    ideal = []
    anti_ideal = []
    for j in range(n):
        if impacts[j] == 1:
            ideal.append(weighted_matrix[:, j].max())
            anti_ideal.append(weighted_matrix[:, j].min())
        else:
            ideal.append(weighted_matrix[:, j].min())
            anti_ideal.append(weighted_matrix[:, j].max())
            
    d_plus = np.sqrt(((weighted_matrix - np.array(ideal))**2).sum(axis=1))
    d_minus = np.sqrt(((weighted_matrix - np.array(anti_ideal))**2).sum(axis=1))
    scores = d_minus / (d_plus + d_minus)
    
    res = pd.DataFrame({"TOPSIS Score": scores}, index=df.index)
    res["Rank"] = res["TOPSIS Score"].rank(ascending=False).astype(int)
    return res.sort_values(by="Rank")

# Title and Description
st.title("🏆 MCDM-ModelSelect: Constrained ML Model Selection Dashboard")
st.markdown("""
This production-ready dashboard applies **Multi-Criteria Decision-Making (MCDM)** to select the most optimal Machine Learning model. 
By integrating hardware, infrastructure, and software engineering constraints (Speed & Size) alongside statistical metrics, we escape the 'accuracy-only' vacuum.
""")

# Load initial datasets
try:
    df_matrix = pd.read_csv("decision_matrix.csv", index_col=0)
except FileNotFoundError:
    st.error("❌ 'decision_matrix.csv' not found. Please run the training script first!")
    st.stop()

# Layout layout split: Sidebar for controls
st.sidebar.header("⚙️ Operational Scenario & Weights")
st.sidebar.markdown("Select a deployment profile or switch to manual configuration to adjust constraints.")

# Define Scenario Presets to eliminate Entropy sample-size limitations
scenarios = {
    "Standard Balanced (Entropy-Driven)": [0.30, 0.30, 0.10, 0.20, 0.10], # Default fallback weights
    "Edge / IoT Device Profile (Low Latency & Lightweight)": [0.15, 0.15, 0.10, 0.40, 0.20], # Prioritizes Inference and Size
    "Enterprise Cloud Profile (High Accuracy & Throughput)": [0.45, 0.35, 0.10, 0.05, 0.05], # Prioritizes Metrics
    "Custom / Manual Configuration": None
}

selected_scenario = st.sidebar.selectbox("Choose Deployment Profile:", list(scenarios.keys()))

# Handle Weights based on chosen scenario
if selected_scenario != "Custom / Manual Configuration":
    preset_weights = scenarios[selected_scenario]
    w_acc = preset_weights[0]
    w_f1 = preset_weights[1]
    w_train = preset_weights[2]
    w_inf = preset_weights[3]
    w_size = preset_weights[4]
    
    # Visual cues for locked presets
    st.sidebar.disabled = True
    st.sidebar.info(f"🔒 Weights are locked to match the **{selected_scenario}** engineering spec.")
else:
    st.sidebar.markdown("### Manual Fine-Tuning")
    w_acc = st.sidebar.slider("Accuracy Weight", 0.0, 1.0, 0.30, 0.05)
    w_f1 = st.sidebar.slider("F1-Score Weight", 0.0, 1.0, 0.30, 0.05)
    w_train = st.sidebar.slider("Train Time Weight (Min)", 0.0, 1.0, 0.10, 0.05)
    w_inf = st.sidebar.slider("Inference Time Weight (Min)", 0.0, 1.0, 0.20, 0.05)
    w_size = st.sidebar.slider("Model Size Weight (Min)", 0.0, 1.0, 0.10, 0.05)

# Normalize weights so they sum up to 1.0
user_weights = np.array([w_acc, w_f1, w_train, w_inf, w_size])
weight_sum = user_weights.sum()
if weight_sum == 0:
    user_weights = np.ones(5) / 5
else:
    user_weights = user_weights / weight_sum

# Display current normalized weights in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Normalized Weights Applied:")
for col, w in zip(df_matrix.columns, user_weights):
    st.sidebar.write(f"**{col}:** {round(w * 100, 1)}%")

# Main Dashboard Content Split into Tabs
tab1, tab2 = st.tabs(["📊 Performance Matrix & Rankings", "📈 Decision Insights"])

with tab1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("1. Empirical Machine Learning Performance Matrix")
        st.dataframe(df_matrix.style.highlight_max(subset=["Accuracy", "F1-Score"], color="#2E7D32")
                     .highlight_min(subset=["Train Time (s)", "Inference Time (s)", "Model Size (KB)"], color="#2E7D32"))
        st.caption("Green highlights represent the best performer for that specific criterion.")
        st.caption("**Hardware Note:** Benchmark executed on standard CPU infrastructure. Inference values represent collective batch execution latency.")
        
    with col2:
        st.subheader("2. Real-Time TOPSIS Rankings")
        impacts = [1, 1, -1, -1, -1]
        df_ranked = run_topsis_dynamic(df_matrix, user_weights, impacts)
        st.dataframe(df_ranked)

with tab2:
    st.subheader("💡 Visualizing the Winner")
    chart_data = df_ranked.sort_values(by="TOPSIS Score", ascending=True)
    st.bar_chart(data=chart_data, y="TOPSIS Score", use_container_width=True)
    
    st.success(f"🚀 **Deployment Recommendation:** Under the selected profile (**{selected_scenario}**), the optimal model to provision is **{df_ranked.index[0]}** with a TOPSIS score of **{round(df_ranked['TOPSIS Score'].iloc[0], 4)}**!")