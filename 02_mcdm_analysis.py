import pandas as pd
import numpy as np

def calculate_entropy_weights(df):
    """
    Calculates objective criteria weights using Shannon Entropy Method.
    Assumes all values in df are non-negative.
    """
    matrix = df.to_numpy(dtype=float)
    m, n = matrix.shape
    
    # Step 1: Normalize the decision matrix
    # Avoid division by zero by adding a tiny epsilon if sum is 0
    column_sums = matrix.sum(axis=0)
    column_sums[column_sums == 0] = 1e-9
    p_ij = matrix / column_sums
    
    # Step 2: Compute Entropy measures (E_j)
    # Use np.log with handling for p_ij == 0 (0 * log(0) = 0)
    eps = 1e-12
    k = 1.0 / np.log(m)
    entropy = -k * np.sum(p_ij * np.log(p_ij + eps), axis=0)
    
    # Step 3: Compute Diversification (d_j) and Weights (w_j)
    diversification = 1.0 - entropy
    div_sum = diversification.sum()
    
    if div_sum == 0:
        weights = np.ones(n) / n
    else:
        weights = diversification / div_sum
        
    return pd.Series(weights, index=df.columns)

def run_topsis(df, weights, impacts):
    """
    Ranks alternatives using the TOPSIS method based on calculated weights and impacts.
    impacts: list of 1 (maximize) and -1 (minimize)
    """
    matrix = df.to_numpy(dtype=float)
    m, n = matrix.shape
    
    # Step 1: Linear Vector Normalization
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    
    # Step 2: Weighted Normalized Decision Matrix
    weighted_matrix = norm_matrix * weights.to_numpy()
    
    # Step 3: Determine Ideal (A+) and Anti-Ideal (A-) Solutions
    ideal_solution = []
    anti_ideal_solution = []
    
    for j in range(n):
        if impacts[j] == 1:  # Maximize
            ideal_solution.append(weighted_matrix[:, j].max())
            anti_ideal_solution.append(weighted_matrix[:, j].min())
        else:  # Minimize
            ideal_solution.append(weighted_matrix[:, j].min())
            anti_ideal_solution.append(weighted_matrix[:, j].max())
            
    ideal_solution = np.array(ideal_solution)
    anti_ideal_solution = np.array(anti_ideal_solution)
    
    # Step 4: Calculate Separation Measures from Ideal and Anti-Ideal
    d_plus = np.sqrt(((weighted_matrix - ideal_solution)**2).sum(axis=1))
    d_minus = np.sqrt(((weighted_matrix - anti_ideal_solution)**2).sum(axis=1))
    
    # Step 5: Calculate Relative Closeness to the Ideal Solution (Performance Score)
    scores = d_minus / (d_plus + d_minus)
    
    result_df = pd.DataFrame({
        "TOPSIS Score": scores
    }, index=df.index)
    
    result_df["Rank"] = result_df["TOPSIS Score"].rank(ascending=False).astype(int)
    return result_df.sort_values(by="Rank")

if __name__ == "__main__":
    # Load the decision matrix generated in Step 1
    try:
        df_matrix = pd.read_csv("decision_matrix.csv", index_col=0)
        # In case index column is read as data, set it properly
        if "Model" in df_matrix.columns:
            df_matrix = df_matrix.set_index("Model")
    except FileNotFoundError:
        print("❌ 'decision_matrix.csv' not found! Please run Step 1 first.")
        exit()
        
    print("📊 Loaded Decision Matrix:")
    print(df_matrix)
    print("\n" + "="*50 + "\n")
    
    # 1. Calculate Weights using Entropy
    print("🧠 Calculating objective weights via Shannon Entropy...")
    weights = calculate_entropy_weights(df_matrix)
    print("\nCalculated Kriteria Weights:")
    for criterion, w in weights.items():
        print(f" - {criterion}: {round(w * 100, 2)}%")
        
    print("\n" + "="*50 + "\n")
    
    # 2. Run TOPSIS Analysis
    # Define impacts: Accuracy (1), F1-Score (1), Train Time (-1), Inference Time (-1), Model Size (-1)
    impacts = [1, 1, -1, -1, -1]
    
    print("🏆 Ranking models using TOPSIS...")
    final_rankings = run_topsis(df_matrix, weights, impacts)
    
    print("\n🥇 FINAL MODEL RANKINGS:")
    print(final_rankings)
    
    # Save the final results for dashboard visualization
    final_rankings.to_csv("mcdm_results.csv")
    print("\n💾 'mcdm_results.csv' saved successfully!")