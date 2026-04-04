#!/bin/bash

# --- Configuration ---
OUT_DIR="Scalability"
PYTHON_SCRIPT="policy_generator.py"

K=1
PS_PARTIAL=0.4
PS_NOISE=0.2
PN_NOISE=0.2

# Create output directory
mkdir -p "$OUT_DIR"

echo "[*] Generating Scalability Datasets for Benchmarks..."

# List of (m, n) pairs from Synthetic_Results.csv
SIZES=(
    "100 10000"
    "200 20000"
    "300 30000"
    "400 40000"
    "500 50000"
    "600 60000"
    "700 70000"
    "800 80000"
    "900 90000"
    "1000 100000"
)

# 1. Generate Partial Policies (ps=0.4, pn=0.0)
echo "[1/2] Generating Partial Policies..."
for size in "${SIZES[@]}"; do
    set -- $size
    m=$1
    n=$2
    echo "    - m=$m, n=$n, ps=$PS_PARTIAL, pn=0.0"
    python3 "$PYTHON_SCRIPT" "$K" "$m" "$n" "$PS_PARTIAL" 0.0 --dir "$OUT_DIR"
done

# 2. Generate Noise Policies (ps=0.2, pn=0.2)
echo "[2/2] Generating Noise Policies..."
for size in "${SIZES[@]}"; do
    set -- $size
    m=$1
    n=$2
    echo "    - m=$m, n=$n, ps=$PS_NOISE, pn=$PN_NOISE"
    python3 "$PYTHON_SCRIPT" "$K" "$m" "$n" "$PS_NOISE" "$PN_NOISE" --dir "$OUT_DIR"
done

echo "[+] Done. Datasets generated in $OUT_DIR/"
