[🔗 Live Demo Link Here] : https://mcdm-ml-model-selection-7bmxutzymsbwcxx8chbrzx.streamlit.app/

#  MCDM-ModelSelect: Framework for Hardware and Operationally Constrained ML Model Selection

A production-ready engineering framework that applies **Multi-Criteria Decision-Making (MCDM)** to select the most optimal Machine Learning model under strict software engineering, deployment, and hardware constraints.

Instead of evaluating models blindly based on prediction metrics (like Accuracy), this framework introduces an operational engineering approach by factoring in **Training Speed, Live Inference Latency (Batch Inference), and Disk Storage footprints**.

---

##  Architectural Workflow & Structural Limitations

The framework operates in three sequential phases:
1. **ML Pipeline (`01_train_models.py`):** Trains 5 distinct algorithms on a tabular classification dataset, capturing empirical performance metrics.
2. **MCDM Engine (`02_mcdm_analysis.py`):** Computes weight distributions and evaluates alternatives via Vector-Normalized **TOPSIS**.
3. **Interactive Dashboard (`03_dashboard.py`):** A **Streamlit** application allowing stakeholders to pivot between automated statistical profiles and explicit operational constraints in real-time.

### ⚠️ Methodological Scope & Limitations (Read Before Deployment)
* **Dataset-Specific Rankings:** The default benchmark rankings generated herein are highly specific to the underlying tabular dataset and hardware configuration. On complex, high-dimensional, or unstructured datasets, deep ensembles (e.g., XGBoost) are expected to heavily dominate statistical metrics, shifting the mathematical ideal solution.
* **The Shannon Entropy Boundary:** While Shannon Entropy provides an objective view of mathematical variance across criteria, it is highly sensitive to outlier behaviors when applied to a small sample of alternatives (e.g., 5-7 models). Removing an extreme alternative (like KNN's inference latency) dynamically shifts the entire weight topology. To mitigate this structural circular dependency, this framework introduces **Operational Scenario Profiles** alongside purely statistical weighting.

---

## 🔬 Empirical Measurement Methodology

To ensure production-grade reproducibility, metrics are captured under the following strict boundaries:
* **Hardware Benchmark Environment:** Evaluated on a local CPU execution thread (Standard Core Architecture, system RAM bounds apply). No GPU acceleration is utilized during training or inference.
* **Inference Latency:** Measured as **Batch Inference Time** over the entire test subset (114 samples simultaneously), rather than single-sample stream processing. This represents a batch-prediction microservice scenario.
* **Model Size on Disk:** Represented via serialized `pickle` payload footprints. *Note on Instance-Based Learning:* For non-parametric models like KNN, this size natively scales with the training data volume since the model acts as a direct data repository.

---

## 📐 Mathematical Framework

### 1. Shannon Entropy (Statistical Weighting Variance)
To observe the statistical diversification degree of each criterion, the entropy $E_j$ for the $j$-th criterion is calculated as:

$$E_j = -k \sum_{i=1}^{m} p_{ij} \ln(p_{ij})$$

Where $p_{ij}$ is the normalized decision matrix element and $k = \frac{1}{\ln(m)}$. The final automated weight $w_j$ is given by:

$$w_j = \frac{1 - E_j}{\sum (1 - E_j)}$$

### 2. TOPSIS Method (Closeness Ranking)
Alternatives are ranked based on their geometric distance to the **Positive Ideal Solution ($A^+$)** and **Anti-Ideal Solution ($A^-$)**. The relative closeness performance score ($C_i^*$) is computed as:

$$C_i^* = \frac{D_i^-}{D_i^+ + D_i^-}$$

Where $C_i^* \in [0, 1]$. A score closer to 1 indicates a more robust and operationally viable model.

---

## 📊 Evaluation Criteria & Impacts

| Criterion | Metric Type | Operational Goal | Optimization Impact |
| :--- | :---: | :---: | :---: |
| **Accuracy** | Statistical | Maximize Predictive Power | $+1$ |
| **F1-Score** | Statistical | Maximize Balance (Precision/Recall) | $+1$ |
| **Train Time (s)** | Computational | Minimize Compute/Infrastructure Cost | $-1$ |
| **Inference Time (s)** | Computational | Minimize User-Facing System Latency | $-1$ |
| **Model Size (KB)** | Infrastructure | Minimize Edge Device / Memory Footprint | $-1$ |

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

# Step 2: Compute weights & TOPSIS mathematical ranking
python 02_mcdm_analysis.py

# Step 3: Spin up the interactive visual dashboard
streamlit run 03_dashboard.py
```

---

Developed as an engineering benchmark project combining Data Science metrics with Operational Research methodologies.
