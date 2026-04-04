"""
SEAndroid Policy Converter (Semi-Synthetic) - Memory Efficient with Bit-Packing
Converts real SEAndroid policy structures into synthetic benchmarks.
Follows the same encoding logic as policy_generator.py.
"""

import json
import numpy as np
import argparse
import os
import time
import csv

# =============================================================================
# Helper Functions
# =============================================================================

def get_counts(tensor):
    """Decodes bit-packed uint8 tensor (Mode B)."""
    grant_counts = (tensor >> 4).astype(np.uint8)
    deny_counts = (tensor & 0x0F).astype(np.uint8)
    return grant_counts, deny_counts

def load_dataset(path):
    print(f"[*] Loading dataset from {path}...")
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif path.endswith('.pkl'):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .pkl")

def build_seandroid_h(data):
    """
    Builds a summary policy H and calculates the probability distribution
    of entities based on the original SEAndroid assignments.
    """
    domains = data['domains']
    types = data['types']
    actions = data['actions']
    rules = data['allow_rules']
    
    m_count = len(domains) + len(types)
    k_count = len(actions)
    
    act_to_idx = {name: i for i, name in enumerate(actions)}
    dom_to_idx = {name: i for i, name in enumerate(domains)}
    type_to_idx = {name: i + len(domains) for i, name in enumerate(types)}
    
    # Calculate Frequency Distribution (P)
    counts = np.zeros(m_count, dtype=float)
    
    proc_assign = data.get('process_domain_assignment', {})
    for dom_name, procs in proc_assign.items():
        if dom_name in dom_to_idx:
            counts[dom_to_idx[dom_name]] = len(procs)
            
    res_assign = data.get('resource_type_assignment', {})
    for type_name, resources in res_assign.items():
        if type_name in type_to_idx:
            counts[type_to_idx[type_name]] = len(resources)
            
    # Normalize to get probability vector
    total_entities = np.sum(counts)
    p_dist = counts / total_entities if total_entities > 0 else np.ones(m_count) / m_count

    print(f"[*] Constructing H from SEAndroid: K={k_count}, M={m_count}")
    H = np.zeros((k_count, m_count, m_count), dtype=np.int8)
    for dom, act, typ in rules:
        if dom in dom_to_idx and act in act_to_idx and typ in type_to_idx:
            H[act_to_idx[act], dom_to_idx[dom], type_to_idx[typ]] = 1
            
    return H, m_count, k_count, p_dist

# =============================================================================
# Core Expansion & Processing Logic (Aligned with policy_generator.py)
# =============================================================================

def process_action_slice(a_idx, H_actual, mapping, n, ps, pn, mode_b):
    """Processes one action slice with wildcards and noise (Mode B)."""
    print(f"[*] Processing Action {a_idx}...")
    
    # 1. Expand from H. Start with uint8 for Mode B or int8 for Mode A.
    dtype = np.uint8 if mode_b else np.int8
    slice_data = H_actual[a_idx][np.ix_(mapping, mapping)].astype(dtype)
    
    # 2. Inject Wildcards (Action-wise to save RAM)
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

    # 3. Apply Noise/Replication (Mode B only)
    if mode_b and pn > 0:
        import math
        print(f"    - Applying replication and noise (pn={pn})...")
        slice_data.fill(0) 
        
        # PRE-CALCULATE PROBABILITY TABLES
        R_vals = [1, 3, 5]
        R_probs = [0.90, 0.09, 0.01]
        
        def get_outcome_table(is_grant):
            outcomes = []
            probs = []
            for r, pr in zip(R_vals, R_probs):
                for x in range(r + 1):
                    # Binomial PDF
                    px = math.comb(r, x) * (pn**x) * ((1-pn)**(r-x))
                    joint_p = pr * px
                    grants = (r - x) if is_grant else x
                    denies = x if is_grant else (r - x)
                    packed = (grants << 4) | denies
                    outcomes.append(packed)
                    probs.append(joint_p)
            res_p = np.array(probs)
            return np.array(outcomes, dtype=np.uint8), res_p / res_p.sum()

        out_grant, p_grant = get_outcome_table(True)
        out_deny, p_deny = get_outcome_table(False)

        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk_orig = H_actual[a_idx][np.ix_(mapping[i:end], mapping)].astype(np.int8)
            
            # Apply Wildcards
            mask_wild = np.random.random(chunk_orig.shape) < ps
            chunk_orig[mask_wild] = 2
            
            target = np.zeros(chunk_orig.shape, dtype=np.uint8)
            mask1, mask0 = (chunk_orig == 1), (chunk_orig == 0)
            if np.any(mask1): target[mask1] = np.random.choice(out_grant, size=np.sum(mask1), p=p_grant)
            if np.any(mask0): target[mask0] = np.random.choice(out_deny, size=np.sum(mask0), p=p_deny)
            slice_data[i:end] = target
    elif mode_b:
        # No noise, just pack 1-to-1 counts
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
        # Mode A implementation
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk_orig = H_actual[a_idx][np.ix_(mapping[i:end], mapping)].astype(np.int8)
            mask_wild = np.random.random(chunk_orig.shape) < ps
            chunk_orig[mask_wild] = 2
            slice_data[i:end] = chunk_orig

    return slice_data, 0, 0, 0 # Metrics omitted for speed

# =============================================================================
# Main Process
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SEAndroid Semi-Synthetic Policy Generator (Unified)')
    parser.add_argument('n', type=int, help='Target number of entities')
    parser.add_argument('ps', type=float, help='Wildcard probability (ps)')
    parser.add_argument('pn', type=float, help='Noise probability (pn)')
    parser.add_argument('input', help='Path to seandroid_dataset.json or .pkl')
    parser.add_argument('out_dir', nargs='?', default='Policy', help='Output directory')
    args = parser.parse_args()
    
    start_time = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Build H and get real-world probability distribution
    data = load_dataset(args.input)
    H, m_found, k_found, p_dist = build_seandroid_h(data)
    
    mode_b = args.pn > 0
    file_tag = f'k{k_found}m{m_found}n{args.n}ps_{args.ps}'
    if mode_b: file_tag += f'pn_{args.pn}'
    
    print(f"Params: n={args.n}, ps={args.ps}, pn={args.pn}, mode={'B (Bit-Packed)' if mode_b else 'A (Standard)'}")
    print("-" * 40)
    
    # Generate Mapping
    group_sizes = np.ones(m_found, dtype=int)
    remaining = args.n - m_found
    if remaining > 0:
        group_sizes += np.random.multinomial(remaining, p_dist)
        
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
        
    # Save Mapping
    csv_file = os.path.join(args.out_dir, f'vertex_to_domain_{file_tag}.csv')
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerows(domains)
    print(f"[*] Saved vertex-to-domain mapping to {csv_file}")

    # 2. Allocate Output Files using MEMMAP (Disk-Backed)
    orig_file = os.path.join(args.out_dir, f'original_policy_{file_tag}.npy')
    print(f"[*] Streams saving Original/Final Policies (n={args.n}) to disk via memmap...")
    
    # Use 'w+' to create/overwrite. Memmap saves RAM by writing to disk.
    G_orig = np.lib.format.open_memmap(orig_file, mode='w+', dtype='int8', shape=(k_found, args.n, args.n))
    
    final_type = "noise" if mode_b else "partial"
    out_file = os.path.join(args.out_dir, f'{final_type}_policy_{file_tag}.npy')
    dtype_str = 'uint8' if mode_b else 'int8'
    G_final = np.lib.format.open_memmap(out_file, mode='w+', dtype=dtype_str, shape=(k_found, args.n, args.n))

    # 3. Stream Process Action-by-Action
    chunk_size = 5000 
    
    for a in range(k_found):
        # 3.1 Fill Original Policy chunks
        for i in range(0, args.n, chunk_size):
            end = min(i + chunk_size, args.n)
            G_orig[a, i:end] = H[a][np.ix_(mapping[i:end], mapping)].astype(np.int8)
        
        # 3.2 Fill Processed (Noisy/Partial) Policy slices
        # slice_data is returned as a RAM-based array for the slice (approx 1GB at n=100k for uint8)
        # then assigned to the disk memmap
        slice_data, _, _, _ = process_action_slice(a, H, mapping, args.n, args.ps, args.pn, mode_b)
        G_final[a] = slice_data
        
        # Flush to physical disk regularly
        G_orig.flush()
        G_final.flush()

    # 4. Final Summary Output
    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"Total time (Optimized): {elapsed:.2f}s")
    print(f"Files stored in: {args.out_dir}")
    del G_orig, G_final

if __name__ == '__main__':
    main()
