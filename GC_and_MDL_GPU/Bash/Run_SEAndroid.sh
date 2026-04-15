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

# --- 2. Process NOISE policies ---
for noise in "$DATA_DIR"/noise_policy_*.npy; do
    # Extract parameters: ps and pn
    ps=$(echo "$noise" | grep -oP 'ps_\K[0-9.]+')
    pn=$(echo "$noise" | grep -oP 'pn_\K[0-9.]+')
    
    original="${noise/noise_policy_/original_policy_}"
    fname=$(basename "$noise")
    
    if [ -f "$original" ]; then
        echo "    [$(date +%T)] Running MDL on $fname (ps=$ps, pn=$pn, 10 runs)..."
        
        for run in {1..10}; do
            raw=$(python3 "$MDL_SCRIPT" "$noise" "$original" 2>/dev/null)
            m=$(extract_metrics "$raw")
            # Print each run result to CSV immediately
            echo "$fname,Noise,$ps,$pn,MDL,$m" >> "$CSV_RESULT"
            echo -n "."
        done
        echo ""
    else
        echo "    [!] Warning: Original file not found for $fname"
    fi
done

# --- 1. Process PARTIAL policies ---
for partial in "$DATA_DIR"/partial_policy_*.npy; do
    # Extract parameter: ps
    ps=$(echo "$partial" | grep -oP 'ps_\K[0-9.]+')
    pn="0.0"
    
    original="${partial/partial_policy_/original_policy_}"
    fname=$(basename "$partial")
    
    if [ -f "$original" ]; then
        echo "    [$(date +%T)] Running GC on $fname (ps=$ps, 10 runs)..."
        
        for run in {1..10}; do
            raw=$(python3 "$GC_SCRIPT" "$partial" "$original" 2>/dev/null)
            m=$(extract_metrics "$raw")
            # Print each run result to CSV immediately
            echo "$fname,Partial,$ps,$pn,GC,$m" >> "$CSV_RESULT"
            echo -n "."
        done
        echo ""
    else
        echo "    [!] Warning: Original file not found for $fname"
    fi
done

shopt -u nullglob
echo "----------------------------------------------------"
echo "[+] Done. Results saved to $CSV_RESULT"
