#!/bin/bash

# Configuration
INPUT_DIR="CODASPY2024_DBPM_Dataset"
RAWDATA_DIR="raw_data"
OUTPUT_DIR="output"
TIMEOUT=300

# Ensure the root raw data directory exists
mkdir -p "$RAWDATA_DIR"

# List of Graph Coloring (GC) methods
GC_METHODS=("RS" "LF" "SL" "RSI" "LFI" "SLI" "CSB" "CSD" "SLF" "GIS")

# List of MaxSAT methods
MAXSAT_METHODS=("BE_NF_MD_LI")


echo "Starting Sequential Reproduction of GC vs MaxSAT"
# Loop through each subdirectory in the input directory (M2N100, M2N200, etc.)
for subfolder_path in "$INPUT_DIR"/*/; do
    # Extract the name of the subfolder (e.g., M2N100)
    subfolder=$(basename "$subfolder_path")
    
    # Create a corresponding output directory in raw_data
    current_out_dir="$RAWDATA_DIR/$subfolder"
    mkdir -p "$current_out_dir"
    
    echo "[*] Processing Dataset: $subfolder"

    # 1. Run Graph Coloring (sergcp) Experiments for this subfolder
    for method in "${GC_METHODS[@]}"; do
        echo "    - sergcp | Method: $method"
        # We point test_driver.py to the specific subfolder for input and output
        python3 test_driver.py sergcp "$method" "$subfolder_path" "$current_out_dir" "$TIMEOUT"
    done

    # 2. Run MaxSAT Experiments for this subfolder
    for method in "${MAXSAT_METHODS[@]}"; do
        echo "    - maxsat | Method: $method"
        python3 test_driver.py maxsat "$method" "$subfolder_path" "$current_out_dir" "$TIMEOUT"
    done
done

echo "Experiments Complete!"
echo "Layered results are stored in: $RAWDATA_DIR"

echo "[*] Experiments complete. Running analyzer..."
python3 rawdata_analyzer.py "$RAWDATA_DIR" "$OUTPUT_DIR" "$TIMEOUT" 86400
