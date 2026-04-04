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
echo "filename,type,ps,pn,method,accuracy,precision,recall,f1,domains,time" > "$CSV_RESULT"

# Extraction Function
extract_metrics() {
    echo "$1" | awk -F': ' '
        /Accuracy:/ {acc=$2}
        /Precision:/ {prec=$2}
        /Recall:/ {rec=$2}
        /F1-Score:/ {f1=$2}
        /Final Domains:/ {dom=$2}
        /Compression Time:/ {gsub(/ s/,"",$2); t=$2}
        END {print acc "," prec "," rec "," f1 "," dom "," t}
    '
}

echo "----------------------------------------------------"
echo "[*] Starting Test Benchmarks on $DATA_DIR folder..."
echo "----------------------------------------------------"

shopt -s nullglob

# --- 1. Process Partial Policies ---
for f in "$DATA_DIR"/partial_policy_*.npy; do
    fname=$(basename "$f")
    ps=$(echo "$fname" | grep -oP 'ps_\K[0-9.]+')
    pn="0.0"
    
    # Locate corresponding original file
    orig="${f/partial_policy/original_policy}"
    
    if [ ! -f "$orig" ]; then
        echo "[!] Original file not found for $fname. Skipping."
        continue
    fi

    # Run GC
    echo "[Partial] Running GC on $fname..."
    raw=$(python3 "$GC_SCRIPT" "$f" "$orig")
    metrics=$(extract_metrics "$raw")
    echo "$fname,Partial,$ps,$pn,GC,$metrics" >> "$CSV_RESULT"

    # Run DT
    echo "[Partial] Running DT on $fname..."
    raw=$(python3 "$DT_SCRIPT" "$f" "$orig")
    metrics=$(extract_metrics "$raw")
    echo "$fname,Partial,$ps,$pn,DT,$metrics" >> "$CSV_RESULT"

    # Run MLP
    echo "[Partial] Running MLP on $fname..."
    raw=$(python3 "$MLP_SCRIPT" "$f" "$orig")
    metrics=$(extract_metrics "$raw")
    echo "$fname,Partial,$ps,$pn,MLP,$metrics" >> "$CSV_RESULT"
done

# --- 2. Process Noise Policies ---
for f in "$DATA_DIR"/noise_policy_*.npy; do
    fname=$(basename "$f")
    ps=$(echo "$fname" | grep -oP 'ps_\K[0-9.]+')
    pn=$(echo "$fname" | grep -oP 'pn_\K[0-9.]+')
    
    orig="${f/noise_policy/original_policy}"
    
    if [ ! -f "$orig" ]; then
        echo "[!] Original file not found for $fname. Skipping."
        continue
    fi

    # Run MDL
    echo "[Noise] Running MDL on $fname..."
    raw=$(python3 "$MDL_SCRIPT" "$f" "$orig")
    metrics=$(extract_metrics "$raw")
    echo "$fname,Noise,$ps,$pn,MDL,$metrics" >> "$CSV_RESULT"

    # Run DT
    echo "[Noise] Running DT on $fname..."
    raw=$(python3 "$DT_SCRIPT" "$f" "$orig")
    metrics=$(extract_metrics "$raw")
    echo "$fname,Noise,$ps,$pn,DT,$metrics" >> "$CSV_RESULT"

    # Run MLP
    echo "[Noise] Running MLP on $fname..."
    raw=$(python3 "$MLP_SCRIPT" "$f" "$orig")
    metrics=$(extract_metrics "$raw")
    echo "$fname,Noise,$ps,$pn,MLP,$metrics" >> "$CSV_RESULT"
done

shopt -u nullglob
echo "----------------------------------------------------"
echo "[+] Done! Results saved to $CSV_RESULT"
echo "----------------------------------------------------"
