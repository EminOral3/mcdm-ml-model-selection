# 🏆 Meta-MCDM: Optimizing Machine Learning Model Selection via Shannon Entropy & TOPSIS

An end-to-end production-ready framework that applies **Multi-Criteria Decision-Making (MCDM)** to select the most optimal Machine Learning model under strict software engineering and hardware constraints. 

Instead of evaluating models blindly based on prediction metrics (like Accuracy), this project introduces an operational engineering approach by factoring in **Training Speed, Live Inference Latency, and Disk Storage footprints**.

---

## 🧠 Mathematical Background

### 1. Shannon Entropy (Objective Weighting)
To remove human subjectivity from weight assignments, we analyze the objective diversification degree of each criterion. The entropy $E_j$ for the $j$-th criterion is calculated as:

$$E_j = -k \sum_{i=1}^{m} p_{ij} \ln(p_{ij})$$

Where $p_{ij}$ is the normalized decision matrix element and $k = \frac{1}{\ln(m)}$. The final criterion weight $w_j$ is given by:

$$w_j = \frac{1 - E_j}{\sum (1 - E_j)}$$

### 2. TOPSIS Method (Ranking)
Alternatives are ranked based on their geometric distance to the **Positive Ideal Solution ($A^+$)** and **Anti-Ideal Solution ($A^-$)**. The relative closeness performance score ($C_i^*$) is computed as:

$$C_i^* = \frac{D_i^-}{D_i^+ + D_i^-}$$

Where $C_i^* \in [0, 1]$. A score closer to 1 indicates a more robust and optimal model.

---

## 📊 Evaluation Criteria & Impacts

| Criterion | Metric Type | Operational Goal | Optimization Impact |
| :--- | :---: | :---: | :---: |
| **Accuracy** | Statistical | Maximize Predictive Power | $+1$ |
| **F1-Score** | Statistical | Maximize Balance (Precision/Recall) | $+1$ |
| **Train Time (s)** | Computational | Minimize Infrastructure Cost | $-1$ |
| **Inference Time (s)** | Computational | Minimize User-Facing Latency | $-1$ |
| **Model Size (KB)** | Infrastructure | Minimize Edge Device / Memory Load | $-1$ |

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/EminOral3/mcdm-ml-model-selection.git
cd mcdm-ml-model-selection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Execution Order
Run the entire pipeline step-by-step:
```bash
# Step 1: Train models & generate raw decision matrix
python 01_train_models.py

# Step 2: Compute Entropy weights & TOPSIS mathematical ranking
python 02_mcdm_analysis.py

# Step 3: Spin up the interactive visual dashboard
streamlit run 03_dashboard.py
```

---

## 🏆 Industry Case Study Result Summary

When evaluating models on standard enterprise data under objective weights, the framework compromises raw metrics to optimize hardware constraints:

- **Decision Tree** emerges as the Rank 1 Champion due to its microsecond inference latency and negligible model size (~3 KB), despite having slightly lower accuracy than bulky ensembles.
- **K-Nearest Neighbors (KNN)** is dynamically penalized to Last Place because its lazy-learning nature causes massive runtime inference delays.

Developed as a benchmark project combining Data Science principles with Operational Research methodologies.
