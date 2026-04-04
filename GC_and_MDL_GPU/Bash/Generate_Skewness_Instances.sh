#!/bin/bash

# Configuration
OUT_DIR="Skewness"
PYTHON_SCRIPT="policy_generator.py"

K=1
M=20
N=200

# Create main and sub-directories
mkdir -p "$OUT_DIR/Partial"
mkdir -p "$OUT_DIR/Noise"

echo "[*] Batch Execution Started: Categorizing into Partial/ and Noise/"
echo "[*] Parameters: ps [0.1-0.9], pn [0.0-0.3], alpha [0.1-0.9]"

count=0

# Loop ps from 0.1 to 0.9
for ps_idx in $(seq 1 9); do
    ps=$(echo "scale=1; $ps_idx / 10" | bc -l)

    # Loop pn from 0.0 to 0.3
    for pn_idx in $(seq 0 3); do
        pn=$(echo "scale=1; $pn_idx / 10" | bc -l)

        # Constraint: ps + pn <= 1.0
        is_valid=$(echo "$ps + $pn <= 1.0" | bc -l)
        
        if [ "$is_valid" -eq 1 ]; then
            
            # Determine Output Sub-directory
            # If pn is 0.0, it's Partial. Otherwise, it's Noise.
            if [ "$(echo "$pn == 0" | bc -l)" -eq 1 ]; then
                SUB_DIR="$OUT_DIR/Partial"
                TYPE="Partial"
            else
                SUB_DIR="$OUT_DIR/Noise"
                TYPE="Noise"
            fi

            # Loop alpha from 0.1 to 0.9
            for alpha_idx in $(seq 1 9); do
                alpha=$(echo "scale=1; $alpha_idx / 10" | bc -l)
                
                count=$((count + 1))
                
                echo "----------------------------------------------------"
                echo "Exp #$count | Type: $TYPE | ps=$ps, pn=$pn, alpha=$alpha"
                
                # Execute Python Script with specific sub-directory
                python3 "$PYTHON_SCRIPT" "$K" "$M" "$N" "$ps" "$pn" --alpha="$alpha" --dir "$SUB_DIR"
                
                if [ $? -ne 0 ]; then
                    echo "[!] Failed at Exp #$count ($TYPE). Stopping."
                    exit 1
                fi
            done
        fi
    done
done

echo "----------------------------------------------------"
echo "[+] Done! Summary:"
echo "    - Total Experiments: $count"
echo "    - Partial Stored in: $OUT_DIR/Partial"
echo "    - Noise Stored in:   $OUT_DIR/Noise"
