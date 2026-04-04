import csv
from collections import defaultdict

# Data structures to store F1 scores grouped by method and alpha
partial_data = defaultdict(lambda: defaultdict(list))
noise_data = defaultdict(lambda: defaultdict(list))

# Path to the results CSV file
csv_path = 'Results_Skewness.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        policy_type = row['type']
        method = row['method']
        # Handle alpha values which may have trailing dots (e.g., '0.1.')
        alpha = float(row['alpha'].rstrip('.'))
        f1 = float(row['f1'])
        
        # Filter and group data by policy type and method
        if policy_type == 'Partial':
            if method in ['GC', 'DT', 'MLP']:
                partial_data[method][alpha].append(f1)
        elif policy_type == 'Noise':
            if method in ['MDL', 'DT', 'MLP']:
                noise_data[method][alpha].append(f1)

def print_trends(title, data, methods):
    """Prints a formatted table of average F1-scores by alpha for specified methods."""
    # Collect all unique alpha values across all relevant methods
    all_alphas = set()
    for m in methods:
        if m in data:
            all_alphas.update(data[m].keys())
    
    if not all_alphas:
        print(f"\nNo data found for {title}")
        return
        
    alphas = sorted(list(all_alphas))
    
    print(f"\n{title}")
    header = "Method/Alpha | " + " | ".join(f"{a:.1f}" for a in alphas)
    print(header)
    print("-" * len(header))
    
    for m in methods:
        if m in data:
            row_items = []
            for a in alphas:
                scores = data[m].get(a)
                if scores:
                    avg_f1 = sum(scores) / len(scores)
                    row_items.append(f"{avg_f1:.4f}")
                else:
                    row_items.append("  N/A ")
            row_str = f"{m:<12} | " + " | ".join(row_items)
            print(row_str)
        else:
            print(f"{m:<12} | " + " | ".join(["  N/A "] * len(alphas)))

# Display average F1-score trends as requested
print_trends("Partial Policies: Average F1-score Trends by alpha", partial_data, ['GC', 'DT', 'MLP'])
print_trends("Noise Policies: Average F1-score Trends by alpha", noise_data, ['MDL', 'DT', 'MLP'])
