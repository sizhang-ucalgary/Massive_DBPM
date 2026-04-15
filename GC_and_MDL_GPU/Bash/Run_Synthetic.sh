#!/bin/bash

# --- Configuration ---
DATA_DIR="Synthetic"
CSV_RESULT="Results_Synthetic.csv"

# Script Paths
GC_SCRIPT="gc_compressor.py"
MDL_SCRIPT="mdl_compressor.py"
DT_SCRIPT="dt_sklearn.py"
MLP_SCRIPT="mlp_torch.py"

# Initialize CSV Header
# Metrics: [Accuracy, Precision, Recall, F1, Model Complexity, Time]
echo "filename,type,ps,pn,method,accuracy,precision,recall,f1,complexity,time" > "$CSV_RESULT"

# Extraction Function
extract_metrics() {
    echo "$1" | awk -F': ' '
        /Accuracy:/ {acc=$2}
        /Precision:/ {prec=$2}
        /Recall:/ {rec=$2}
        /F1-Score:/ {f1=$2}
        /Final Domains:/ {gsub(/ /,"",$2); dom=$2}
        /Compression Time:/ {gsub(/ s/,"",$2); t=$2}
        END {print acc "," prec "," rec "," f1 "," dom "," t}
    '
}

# Helper to run 10 times and print each result
run_benchmark() {
    local script=$1
    local f=$2
    local orig=$3
    local method_name=$4
    local type=$5
    local ps=$6
    local pn=$7
    
    echo "  [+] Benchmarking $method_name on $(basename "$f") (10 runs)..."
    for run in {1..10}; do
        raw=$(python3 "$script" "$f" "$orig" 2>/dev/null)
        m=$(extract_metrics "$raw")
        # Print each run result to CSV
        echo "$(basename "$f"),$type,$ps,$pn,$method_name,$m" >> "$CSV_RESULT"
        echo -n "." # Progress indicator
    done
    echo "" # Newline after dots
}

echo "===================================================="
echo "[*] Massive DBPM Synthetic Benchmark Suite"
echo "===================================================="
echo "    Data Directory: $DATA_DIR"
echo "    Output File:    $CSV_RESULT"
echo "----------------------------------------------------"

shopt -s nullglob

# --- 2. Noisy Setting: [MDL vs DT vs MLP] ---
for f in "$DATA_DIR"/noise_policy_*.npy; do
    fname=$(basename "$f")
    ps=$(echo "$fname" | sed -n 's/.*ps_\([0-9.]*\).*/\1/p')
    pn=$(echo "$fname" | sed -n 's/.*pn_\([0-9.]*\).*/\1/p')
    
    orig="${f/noise_policy/original_policy}"
    if [ ! -f "$orig" ]; then
        echo "[!] Original file not found for $fname. Skipping."
        continue
    fi

    echo ">>> Processing Noisy Data: $fname (ps=$ps, pn=$pn)"

    run_benchmark "$MDL_SCRIPT" "$f" "$orig" "MDL" "Noisy" "$ps" "$pn"
    run_benchmark "$DT_SCRIPT" "$f" "$orig" "DT" "Noisy" "$ps" "$pn"
    run_benchmark "$MLP_SCRIPT" "$f" "$orig" "MLP" "Noisy" "$ps" "$pn"
done

# --- 1. Clean Setting: [GC vs DT vs MLP] ---
for f in "$DATA_DIR"/partial_policy_*.npy; do
    fname=$(basename "$f")
    ps=$(echo "$fname" | sed -n 's/.*ps_\([0-9.]*\).*/\1/p')
    pn="0.0"
    
    orig="${f/partial_policy/original_policy}"
    if [ ! -f "$orig" ]; then
        echo "[!] Original file not found for $fname. Skipping."
        continue
    fi

    echo ">>> Processing Clean Data: $fname (ps=$ps, pn=$pn)"

    run_benchmark "$GC_SCRIPT" "$f" "$orig" "GC" "Clean" "$ps" "$pn"
    run_benchmark "$DT_SCRIPT" "$f" "$orig" "DT" "Clean" "$ps" "$pn"
    run_benchmark "$MLP_SCRIPT" "$f" "$orig" "MLP" "Clean" "$ps" "$pn"
done

shopt -u nullglob
echo "----------------------------------------------------"
echo "[+] Done. All measurements saved to $CSV_RESULT"
echo "===================================================="
