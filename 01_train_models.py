import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def load_and_preprocess_data():
    """
    Loads a reliable, built-in classification dataset from scikit-learn
    to avoid any 404 network errors, mimicking a real production workflow.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Scale features since models like KNN and Logistic Regression are scale-sensitive
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Load and split the built-in production dataset
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# 2. Define competing machine learning models (MCDM Alternatives)
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

decision_matrix_data = []

print("🚀 Training models and collecting MCDM criteria metrics...\n")

for name, model in models.items():
    # --- Measure Training Time ---
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train
    
    # --- Measure Inference Time ---
    start_inf = time.time()
    preds = model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    
    # --- Calculate Performance Metrics ---
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # --- Measure Model Storage Size on Disk ---
    filename = f"{name.replace(' ', '_')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    model_size_kb = os.path.getsize(filename) / 1024
    os.remove(filename)  # Clean up the temporary file
    
    # Append results to the decision matrix list
    decision_matrix_data.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Train Time (s)": round(training_time, 4),
        "Inference Time (s)": round(inference_time, 4),
        "Model Size (KB)": round(model_size_kb, 2)
    })
    print(f"✅ {name} evaluation completed.")

# 3. Convert to DataFrame and save the final Decision Matrix
df_decision_matrix = pd.DataFrame(decision_matrix_data).set_index("Model")
print("\n📊 GENERATED MCDM DECISION MATRIX:")
print(df_decision_matrix)

# Save the matrix for the next step (MCDM calculation script)
df_decision_matrix.to_csv("decision_matrix.csv")
print("\n💾 'decision_matrix.csv' saved successfully!")