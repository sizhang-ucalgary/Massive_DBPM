import time
import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def main():
    parser = argparse.ArgumentParser(description="Decision Tree Policy Miner")
    parser.add_argument("input_policy", help="Path to input policy (.npy)")
    parser.add_argument("original_policy", help="Path to original policy (.npy)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for DT")
    args = parser.parse_args()

    # Load Data
    pp = np.load(args.input_policy)
    op = np.load(args.original_policy)
    
    start_time = time.time()

    # Data Extraction Helper
    def extract_data(tensor):
        if tensor.dtype == np.uint8:
            # Bit-packed Mode B
            g = (tensor >> 4).astype(np.int32)
            d = (tensor & 0x0F).astype(np.int32)
            mask = (g + d > 0)
            a_idx, u_idx, v_idx = np.where(mask)
            # Resolve to majority decision (Deny-by-Default)
            y = np.where(g[a_idx, u_idx, v_idx] > d[a_idx, u_idx, v_idx], 1, 0).astype(np.int32)
            X = np.stack([a_idx, u_idx, v_idx], axis=1).astype(np.int32)
        else:
            # Standard Mode A
            a_idx, u_idx, v_idx = np.where(tensor != 2)
            y = tensor[a_idx, u_idx, v_idx].astype(np.int32)
            X = np.stack([a_idx, u_idx, v_idx], axis=1).astype(np.int32)
        return X, y

    X_train, y_train = extract_data(pp)
    X_test, y_test = extract_data(op)

    # Scikit-learn Decision Tree
    model = DecisionTreeClassifier(random_state=args.random_state)
    print("\n--- PHASE 1: Training ---")
    model.fit(X_train, y_train)

    print("\n--- PHASE 2: Testing ---")
    y_pred = model.predict(X_test)

    elapsed_time = time.time() - start_time

    # Metrics Computation
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    tn = int(np.sum((y_pred == 0) & (y_test == 0)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Standardized Output
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Final Domains: {model.tree_.n_leaves}")
    print(f"Compression Time: {elapsed_time:.2f} s")
    print("-" * 40)

if __name__ == '__main__':
    main()
