#!/bin/bash

# --- Configuration ---
DATA_DIR="SEAndroid"
CSV_RESULT="Results_SEAndroid.csv"

GC_SCRIPT="gc_compressor.py"
MDL_SCRIPT="mdl_compressor.py"

# Initialize CSV Header (Including ps and pn extracted from filenames)
echo "filename,type,ps,pn,method,accuracy,precision,recall,f1,domains,time" > "$CSV_RESULT"

echo "[*] Processing SEAndroid Dataset..."

# Function to extract metrics from Python output
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

shopt -s nullglob

# --- 1. Process PARTIAL policies ---
for partial in "$DATA_DIR"/partial_policy_*.npy; do
    # Extract parameter: ps (using grep -oP for matching)
    ps=$(echo "$partial" | grep -oP 'ps_\K[0-9.]+')
    pn="0.0"
    
    # Locate corresponding original file by prefix replacement
    original="${partial/partial_policy_/original_policy_}"
    fname=$(basename "$partial")
    
    if [ -f "$original" ]; then
        echo "    [$(date +%T)] Running GC on $fname (ps=$ps)..."
        raw_output=$(python3 "$GC_SCRIPT" "$partial" "$original")
        metrics=$(extract_metrics "$raw_output")
        echo "$fname,Partial,$ps,$pn,GC,$metrics" >> "$CSV_RESULT"
    else
        echo "    [!] Warning: Original file not found for $fname"
    fi
done

# --- 2. Process NOISE policies ---
for noise in "$DATA_DIR"/noise_policy_*.npy; do
    # Extract parameters: ps and pn
    ps=$(echo "$noise" | grep -oP 'ps_\K[0-9.]+')
    pn=$(echo "$noise" | grep -oP 'pn_\K[0-9.]+')
    
    original="${noise/noise_policy_/original_policy_}"
    fname=$(basename "$noise")
    
    if [ -f "$original" ]; then
        echo "    [$(date +%T)] Running MDL on $fname (ps=$ps, pn=$pn)..."
        raw_output=$(python3 "$MDL_SCRIPT" "$noise" "$original")
        metrics=$(extract_metrics "$raw_output")
        echo "$fname,Noise,$ps,$pn,MDL,$metrics" >> "$CSV_RESULT"
    else
        echo "    [!] Warning: Original file not found for $fname"
    fi
done

shopt -u nullglob
echo "----------------------------------------------------"
echo "[+] Done. Results saved to $CSV_RESULT"
