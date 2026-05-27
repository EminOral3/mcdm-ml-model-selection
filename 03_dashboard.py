import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="MCDM Model Selector", layout="wide", page_icon="🏆")

# Helper functions for calculations (re-used for dynamic user input)
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
st.title("🏆 Meta-MCDM: Optimizing Machine Learning Model Selection")
st.markdown("""
This production-ready dashboard applies **Multi-Criteria Decision-Making (MCDM)** to select the most optimal Machine Learning model. 
Instead of looking blindly at accuracy, we evaluate models based on software engineering constraints like **Speed** and **Storage Size**.
""")

# Load initial datasets
try:
    df_matrix = pd.read_csv("decision_matrix.csv", index_col=0)
except FileNotFoundError:
    st.error("❌ 'decision_matrix.csv' not found. Please run the training script first!")
    st.stop()

# Layout layout split: Sidebar for controls, main panel for visuals
st.sidebar.header("⚙️ User-Defined Criteria Weights")
st.sidebar.markdown("Adjust the importance of each metric manually to trigger real-time TOPSIS re-ranking.")

# Sliders for dynamic weighting
w_acc = st.sidebar.slider("Accuracy Weight", 0.0, 1.0, 0.30, 0.05)
w_f1 = st.sidebar.slider("F1-Score Weight", 0.0, 1.0, 0.30, 0.05)
w_train = st.sidebar.slider("Train Time Weight (Min)", 0.0, 1.0, 0.10, 0.05)
w_inf = st.sidebar.slider("Inference Time Weight (Min)", 0.0, 1.0, 0.20, 0.05)
w_size = st.sidebar.slider("Model Size Weight (Min)", 0.0, 1.0, 0.10, 0.05)

# Normalize user weights so they sum up to 1.0
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
        st.subheader("1. Raw Machine Learning Performance Matrix")
        st.dataframe(df_matrix.style.highlight_max(subset=["Accuracy", "F1-Score"], color="#2E7D32")
                     .highlight_min(subset=["Train Time (s)", "Inference Time (s)", "Model Size (KB)"], color="#2E7D32"))
        st.caption("Green highlights represent the best performer for that specific criterion.")
        
    with col2:
        st.subheader("2. Real-Time TOPSIS Rankings")
        # Run dynamic TOPSIS based on sidebar sliders
        impacts = [1, 1, -1, -1, -1]
        df_ranked = run_topsis_dynamic(df_matrix, user_weights, impacts)
        st.dataframe(df_ranked)

with tab2:
    st.subheader("💡 Visualizing the Winner")
    # Horizontal Bar chart for TOPSIS Scores
    chart_data = df_ranked.sort_values(by="TOPSIS Score", ascending=True)
    st.bar_chart(data=chart_data, y="TOPSIS Score", use_container_width=True)
    
    st.info(f"🚀 **Current Champion:** Based on your operational preferences, the best model to deploy is **{df_ranked.index[0]}** with a TOPSIS evaluation score of **{round(df_ranked['TOPSIS Score'].iloc[0], 4)}**!")