"""
Policy Generator - Memory Efficient with Bit-Packing
Unified script to generate large synthetic policies (up to n=100,000).
Supports:
1. Mode A: Standard (int8) - 0:Deny, 1:Grant, 2:Wildcard
2. Mode B: Bit-Packed (uint8) - Counts of Grants/Denies for noisy data
"""

import argparse
import os
import time
import csv
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import DisjointSet

# =============================================================================
# Core Logic Functions
# =============================================================================

def get_counts(tensor):
    """
    Decodes the bit-packed uint8 tensor (Mode B) into separate arrays for Grant and Deny counts.
    High 4 bits: Grant counter. Low 4 bits: Deny counter.
    """
    grant_counts = (tensor >> 4).astype(np.uint8)
    deny_counts = (tensor & 0x0F).astype(np.uint8)
    return grant_counts, deny_counts

def get_decisions(tensor):
    """
    Decodes the bit-packed tensor into single decisions (0:Deny, 1:Grant, 2:Wildcard).
    Uses 'Grant > Deny' for Grant, else Deny (Deny-by-Default if G == D).
    Cells with G=0 and D=0 (0x00) are Wildcards (2).
    """
    g, d = get_counts(tensor)
    # Only Grant if Grant has strict majority
    decisions = np.where(g > d, 1, 0).astype(np.int8)
    # Preserve wildcards
    decisions[g + d == 0] = 2
    return decisions

def create_summary_policy(k, m):
    """Generates a small summary policy H with distinguishable domains."""
    print(f"[*] Creating summary policy for m={m}...")
    H = np.random.choice([0, 1], size=(k, m, m), p=[0.5, 0.5]).astype(np.int8)

    # Minimize H using DisjointSet to find truly distinguishable 'domains'
    ds = DisjointSet(range(m))
    for u, v in combinations(range(m), 2):
        if ds[u] != ds[v] and np.all((H[:, u, u] == H[:, u, v]) & 
                                     (H[:, v, u] == H[:, v, v]) & 
                                     (H[:, u, :] == H[:, v, :]) & 
                                     (H[:, :, u] == H[:, :, v])):
            ds.merge(u, v)
    
    subsets = ds.subsets()
    m_actual = len(subsets)
    print(f"    - Reduced m from {m} to {m_actual} distinguishable domains")

    # Map domains to H_actual
    H_actual = np.zeros((k, m_actual, m_actual), dtype=np.int8)
    for i, s in enumerate(subsets):
        rep_i = next(iter(s))
        for j, s2 in enumerate(subsets):
            rep_j = next(iter(s2))
            H_actual[:, i, j] = H[:, rep_i, rep_j]
            
    return H_actual, m_actual

def process_action_slice(a_idx, H_actual, mapping, n, ps, pn, mode_b):
    """
    Generates and processes one action slice of the policy.
    This is memory-efficient by avoiding full tensor operations on multiple actions at once.
    """
    print(f"[*] Processing Action {a_idx}...")
    
    # 1. Expand from H_actual. Start with uint8 or int8 based on mode.
    dtype = np.uint8 if mode_b else np.int8
    slice_data = H_actual[a_idx][np.ix_(mapping, mapping)].astype(dtype)
    
    # 2. Inject Wildcards
    # Using a mask for ps. To save memory, we can do this in smaller chunks if n^2 is huge.
    
    total_wildcards = 0
    chunk_size = 10000 
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        nrows = end - i
        mask = np.random.random((nrows, n)) < ps
        total_wildcards += np.sum(mask)
        if mode_b:
            slice_data[i:end][mask] = 0x00 
        else:
            slice_data[i:end][mask] = 2 

    # 3. Apply Noise/Replication (Mode B only)
    total_noise = 0
    total_observed = 0

    if mode_b:
        # G_orig: 0=Deny, 1=Grant, 2=Wildcard
        G_orig = slice_data.copy()
        
        if pn > 0:
            import math
            print(f"    - Applying replication and noise (pn={pn})...")
            slice_data.fill(0) 
            
            # PRE-CALCULATE PROBABILITY TABLES
            # There are only 12 possible outcomes for non-wildcards (6 if it's Grant, 6 if it's Deny)
            R_vals = [1, 3, 5]
            R_probs = [0.90, 0.09, 0.01]
            
            def get_outcome_table(is_grant):
                outcomes = []
                probs = []
                for r, pr in zip(R_vals, R_probs):
                    for x in range(r + 1):
                        px = math.comb(r, x) * (pn**x) * ((1-pn)**(r-x))
                        packed = ((r - x) << 4) | x if is_grant else (x << 4) | (r - x)
                        outcomes.append(packed)
                        probs.append(pr * px)
                res_p = np.array(probs)
                return np.array(outcomes, dtype=np.uint8), res_p / res_p.sum()

            out_grant, p_grant = get_outcome_table(True)
            out_deny, p_deny = get_outcome_table(False)

            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                # We reuse H_actual and mapping to avoid holding G_orig in RAM
                chunk_orig = H_actual[a_idx][np.ix_(mapping[i:end], mapping)].astype(np.int8)
                mask_wild = np.random.random(chunk_orig.shape) < ps
                chunk_orig[mask_wild] = 2
                
                target = np.zeros(chunk_orig.shape, dtype=np.uint8)
                mask1, mask0 = (chunk_orig == 1), (chunk_orig == 0)
                if np.any(mask1): target[mask1] = np.random.choice(out_grant, size=np.sum(mask1), p=p_grant)
                if np.any(mask0): target[mask0] = np.random.choice(out_deny, size=np.sum(mask0), p=p_deny)
                slice_data[i:end] = target
        else:
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunk_orig = H_actual[a_idx][np.ix_(mapping[i:end], mapping)].astype(np.int8)
                mask_wild = np.random.random(chunk_orig.shape) < ps
                chunk_orig[mask_wild] = 2
                target = np.zeros(chunk_orig.shape, dtype=np.uint8)
                target[chunk_orig == 1] = 0x10
                target[chunk_orig == 0] = 0x01
                slice_data[i:end] = target
    else:
        # Mode A: Observed logs = non-wildcard cells
        total_observed = (n * n) - total_wildcards
        total_noise = 0

    return slice_data, int(total_noise), int(total_observed), int(total_wildcards)

def main():
    parser = argparse.ArgumentParser(description='Policy Generator')
    parser.add_argument('k', type=int, help='Number of actions')
    parser.add_argument('m', type=int, help='Initial number of domains')
    parser.add_argument('n', type=int, help='Number of entities')
    parser.add_argument('ps', type=float, help='Wildcard probability (ps)')
    parser.add_argument('pn', type=float, help='Noise probability (pn)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Dirichlet alpha for domain skewness (default: 1.0)')
    parser.add_argument('--dir', default='Policy', help='Output directory (default: Policy)')
    
    args = parser.parse_args()
    start_time = time.time()
    os.makedirs(args.dir, exist_ok=True)
    
    mode_b = args.pn > 0
    file_tag = f'k{args.k}m{args.m}n{args.n}ps_{args.ps}'
    if mode_b:
        file_tag += f'pn_{args.pn}'
    if args.alpha != 1.0:
        file_tag += f'alpha_{args.alpha}'
    
    if mode_b:
        print(f"[*] MODE B: Bit-Packed (uint8) enabled due to pn={args.pn}")
    else:
        print(f"[*] MODE A: Standard (int8) enabled (pn=0)")

    # 1. Create Summary Policy and Mapping
    H_actual, m_actual = create_summary_policy(args.k, args.m)
    
    # Random entity-to-domain mapping with Dirichlet skewness
    group_sizes = np.ones(m_actual, dtype=int)
    remaining = args.n - m_actual
    if remaining > 0:
        # Generate probability weights using Dirichlet distribution
        pvals = np.random.dirichlet([args.alpha] * m_actual)
        group_sizes += np.random.multinomial(remaining, pvals)
    
    elements = np.arange(args.n)
    np.random.shuffle(elements)
    
    mapping = np.zeros(args.n, dtype=int)
    domains = []
    curr = 0
    for d_idx, size in enumerate(group_sizes):
        members = elements[curr:curr+size]
        mapping[members] = d_idx
        domains.append(sorted(members.tolist()))
        curr += size

    # Save mapping early
    csv_file = os.path.join(args.dir, f'vertex_to_domain_{file_tag}.csv')
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerows(domains)

    # 2. Allocate Output Files using MEMMAP (Disk-Backed)
    orig_file = os.path.join(args.dir, f'original_policy_{file_tag}.npy')
    print(f"[*] Streams saving Original/Final Policies (n={args.n}) to disk via memmap...")
    
    # Use 'w+' to create/overwrite. Memmap saves RAM by writing to disk.
    G_orig = np.lib.format.open_memmap(orig_file, mode='w+', dtype='int8', shape=(args.k, args.n, args.n))
    
    final_type = "noise" if mode_b else "partial"
    out_file = os.path.join(args.dir, f'{final_type}_policy_{file_tag}.npy')
    dtype_str = 'uint8' if mode_b else 'int8'
    G_final = np.lib.format.open_memmap(out_file, mode='w+', dtype=dtype_str, shape=(args.k, args.n, args.n))

    # 3. Stream Process Action-by-Action
    chunk_size = 5000 # Memory-safe chunk size for 100k
    
    for a in range(args.k):
        # 3.1 Fill Original Policy chunks
        for i in range(0, args.n, chunk_size):
            end = min(i + chunk_size, args.n)
            # Expand H_actual into the chunk
            G_orig[a, i:end] = H_actual[a][np.ix_(mapping[i:end], mapping)].astype(np.int8)
        
        # 3.2 Fill Processed (Noisy/Partial) Policy slices
        # process_action_slice already returns optimized chunked data
        slice_data, _, _, _ = process_action_slice(a, H_actual, mapping, args.n, args.ps, args.pn, mode_b)
        G_final[a] = slice_data
        
        # Flush to physical disk regularly
        G_orig.flush()
        G_final.flush()

    # 4. Final Cleanup (Implicitly closed by gc or script exit)
    print(f"[*] Done. Original: {orig_file}")
    print(f"[*] Done. Final:    {out_file}")
    
    # 5. Summary Output
    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"Total time: {elapsed:.2f}s")
    # Finished
    print("-" * 40)
    print(f"Total time (Optimized): {elapsed:.2f}s")
    print(f"Final shape: {G_final.shape}")
    print(f"Files stored in: {args.dir}")
    del G_orig, G_final

if __name__ == '__main__':
    main()
