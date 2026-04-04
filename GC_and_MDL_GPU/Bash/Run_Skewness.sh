#!/bin/bash

# --- Configuration ---
DATA_DIR="Skewness"
CSV_RESULT="Results_Skewness.csv"

GC_SCRIPT="gc_compressor.py"
MDL_SCRIPT="mdl_compressor.py"
DT_SCRIPT="dt_sklearn.py"
MLP_SCRIPT="mlp_torch.py"

# Initialize CSV with Headers (Overwrites existing file)
echo "type,ps,pn,alpha,method,accuracy,precision,recall,f1,domains,time" > "$CSV_RESULT"

# Execution Function: Runs a script, parses labeled output, and appends to CSV
run_benchmark() {
    local type=$1      # Partial or Noise
    local p_file=$2    # Processed policy file path
    local o_file=$3    # Original policy file path
    local ps=$4
    local pn=$5
    local alpha=$6
    local script=$7
    local method=$8

    echo "[$(date +%T)] Running $method: ps=$ps, pn=$pn, alpha=$alpha..."

    # 1. Run the script and capture full output
    local raw_output=$(python3 "$script" "$p_file" "$o_file")

    # 2. Use awk to extract values from labeled text (ignores dashes and headers)
    local metrics=$(echo "$raw_output" | awk -F': ' '
        /Accuracy:/ {acc=$2}
        /Precision:/ {prec=$2}
        /Recall:/ {rec=$2}
        /F1-Score:/ {f1=$2}
        /Final Domains:/ {dom=$2}
        /Compression Time:/ {gsub(/ s/,"",$2); t=$2}
        END {print acc "," prec "," rec "," f1 "," dom "," t}
    ')
    
    # 3. Append to CSV
    echo "$type,$ps,$pn,$alpha,$method,$metrics" >> "$CSV_RESULT"
}

echo "----------------------------------------------------"
echo "[*] Starting Benchmarks ..."
echo "----------------------------------------------------"

# --- 1. Process Partial Folder ---
for f in "$DATA_DIR/Partial/partial_policy_"*.npy; do
    [[ -e "$f" ]] || continue
    
    ps=$(echo "$f" | grep -oP 'ps_\K[0-9.]+')
    alpha=$(echo "$f" | grep -oP 'alpha_\K[0-9.]+')
    [ -z "$alpha" ] && alpha="1.0"
    pn="0.0"
    
    orig="${f/partial_policy/original_policy}"

    run_benchmark "Partial" "$f" "$orig" "$ps" "$pn" "$alpha" "$GC_SCRIPT" "GC"
    run_benchmark "Partial" "$f" "$orig" "$ps" "$pn" "$alpha" "$DT_SCRIPT" "DT"
    run_benchmark "Partial" "$f" "$orig" "$ps" "$pn" "$alpha" "$MLP_SCRIPT" "MLP"
done

# --- 2. Process Noise Folder ---
for f in "$DATA_DIR/Noise/noise_policy_"*.npy; do
    [[ -e "$f" ]] || continue

    ps=$(echo "$f" | grep -oP 'ps_\K[0-9.]+')
    pn=$(echo "$f" | grep -oP 'pn_\K[0-9.]+')
    alpha=$(echo "$f" | grep -oP 'alpha_\K[0-9.]+')
    [ -z "$alpha" ] && alpha="1.0"
    
    orig="${f/noise_policy/original_policy}"

    run_benchmark "Noise" "$f" "$orig" "$ps" "$pn" "$alpha" "$MDL_SCRIPT" "MDL"
    run_benchmark "Noise" "$f" "$orig" "$ps" "$pn" "$alpha" "$DT_SCRIPT" "DT"
    run_benchmark "Noise" "$f" "$orig" "$ps" "$pn" "$alpha" "$MLP_SCRIPT" "MLP"
done

echo "----------------------------------------------------"
echo "[+] Done! Metrics saved to $CSV_RESULT"
echo "----------------------------------------------------"
