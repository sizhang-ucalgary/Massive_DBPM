#!/bin/bash

# --- Configuration ---
DATA_DIR="Scalability"
CSV_RESULT="Results_Scalability.csv"

GC_SCRIPT="gc_compressor.py"
MDL_SCRIPT="mdl_compressor.py"


# Initialize CSV Header
echo "filename,type,k,m,n,ps,pn,method,accuracy,precision,recall,f1,domains,time,status" > "$CSV_RESULT"

# --- Extraction Function ---
extract_metrics() {
    # Parses labeled output and removes ' s' from time
    echo "$1" | awk -F': ' '
        /Accuracy:/ {acc=$2}
        /Precision:/ {prec=$2}
        /Recall:/ {rec=$2}
        /F1-Score:/ {f1=$2}
        /Final Domains:/ {dom=$2}
        /Compression Time:/ {gsub(/ s/,"",$2); t=$2}
        END {
            if(acc=="") print "0,0,0,0,0,0"
            else print acc "," prec "," rec "," f1 "," dom "," t
        }
    '
}

echo "[*] Starting Scalability Benchmarks"

shopt -s nullglob

# --- 2. Process NOISE policies ---
for noise in "$DATA_DIR"/noise_policy_*.npy; do
    fname=$(basename "$noise")
    original="${noise/noise_policy_/original_policy_}"
    
    # Extract metadata
    k_val=$(echo "$fname" | grep -oP 'k\K[0-9]+')
    m_val=$(echo "$fname" | grep -oP 'm\K[0-9]+')
    n_val=$(echo "$fname" | grep -oP 'n\K[0-9]+')
    ps_val=$(echo "$fname" | grep -oP 'ps_\K[0-9.]+')
    pn_val=$(echo "$fname" | grep -oP 'pn_\K[0-9.]+')

    if (( n_val > 80000 )); then
        echo "[Noise] Skipping MDL on $fname (N=$n_val > 80000)..."
        echo "$fname,Noise,$k_val,$m_val,$n_val,$ps_val,$pn_val,MDL,0,0,0,0,0,0,Skipped" >> "$CSV_RESULT"
        continue
    fi

    echo "[Noise] Running MDL on $fname (N=$n_val)..."
    raw_output=$(python3 "$MDL_SCRIPT" "$noise" "$original")
    metrics=$(extract_metrics "$raw_output")
    echo "$fname,Noise,$k_val,$m_val,$n_val,$ps_val,$pn_val,MDL,$metrics,Success" >> "$CSV_RESULT"
done

# --- 1. Process PARTIAL policies ---
for partial in "$DATA_DIR"/partial_policy_*.npy; do
    fname=$(basename "$partial")
    original="${partial/partial_policy_/original_policy_}"
    
    # Extract metadata using Perl-regex
    k_val=$(echo "$fname" | grep -oP 'k\K[0-9]+')
    m_val=$(echo "$fname" | grep -oP 'm\K[0-9]+')
    n_val=$(echo "$fname" | grep -oP 'n\K[0-9]+')
    ps_val=$(echo "$fname" | grep -oP 'ps_\K[0-9.]+')
    pn_val="0.0"

    echo "[Partial] Running GC on $fname (N=$n_val)..."
    raw_output=$(python3 "$GC_SCRIPT" "$partial" "$original")
    metrics=$(extract_metrics "$raw_output")
    echo "$fname,Partial,$k_val,$m_val,$n_val,$ps_val,$pn_val,GC,$metrics,Success" >> "$CSV_RESULT"
done

shopt -u nullglob
echo "----------------------------------------------------"
echo "[+] Done. Results saved to $CSV_RESULT"
