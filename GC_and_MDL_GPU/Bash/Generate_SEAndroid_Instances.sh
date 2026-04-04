#!/bin/bash

# --- Configuration ---
INPUT_DATA="seandroid_dataset.pkl"
OUT_DIR="SEAndroid"
PYTHON_SCRIPT="seandroid_policy_converter.py"

N=3000

# Create output directory
mkdir -p "$OUT_DIR"

if [ ! -f "$INPUT_DATA" ]; then
    echo "[!] Error: $INPUT_DATA not found. Please ensure the dataset file is in the script directory."
    exit 1
fi

echo "[*] Generating SEAndroid Datasets for Benchmarks..."

# 1. Generate Partial Policies (Varying ps, pn=0.0)
echo "[1/2] Generating Partial Policies (n=$N)..."
for ps in 0.3 0.5 0.7 0.9; do
    echo "    - ps=$ps, pn=0.0"
    python3 "$PYTHON_SCRIPT" "$N" "$ps" 0.0 "$INPUT_DATA" "$OUT_DIR"
done

# 2. Generate Noise Policies (ps=0.1, Varying pn)
echo "[2/2] Generating Noise Policies (n=$N)..."
for pn in 0.1 0.2 0.3 0.4; do
    echo "    - ps=0.1, pn=$pn"
    python3 "$PYTHON_SCRIPT" "$N" 0.1 "$pn" "$INPUT_DATA" "$OUT_DIR"
done

echo "[+] Done. Datasets generated in $OUT_DIR/"
