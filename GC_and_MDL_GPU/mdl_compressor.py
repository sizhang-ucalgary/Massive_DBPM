import argparse
import math
import time
import numpy as np

# Optional GPU support
HAS_GPU = False
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        HAS_GPU = True
except Exception:
    HAS_GPU = False

# =============================================================================
# 1. HELPERS & MDL MATH
# =============================================================================

def get_xp(device):
    """Returns numpy or cupy based on device setting."""
    if device == 'gpu' and HAS_GPU:
        return cp
    return np

def log_star(n):
    """Universal code length for integers."""
    if n <= 0: return 0.0
    res, curr = 0, math.log2(max(n, 1))
    while curr > 1:
        res += curr
        curr = math.log2(curr)
    return res

def get_model_cost(n, k, m):
    return log_star(m) + (n * math.log2(m) if m > 1 else 0) + (k * m * m)

def get_entropy_wildcard(W_grant, W_specified, xp):
    """
    Calculate data cost (bits) based on Shannon entropy.
    
    Args:
        W_grant:     Count of 'Grant' (1) entries per domain-block.
        W_specified: Count of 'Specified' (0 or 1, non-wildcard) entries.
    Returns:
        Total entropy cost in bits (float).
    """
    safe_denom = xp.where(W_specified == 0, 1.0, W_specified.astype(xp.float64))
    grant_ratio = xp.clip(W_grant / safe_denom, 1e-12, 1.0 - 1e-12)
    entropy = -(grant_ratio * xp.log2(grant_ratio) + (1 - grant_ratio) * xp.log2(1 - grant_ratio))
    return float(xp.sum(entropy * W_specified))

def to_cpu(arr):
    if hasattr(arr, 'get'): return arr.get()
    return np.asarray(arr)

def evaluate_labels(D_is1, D_spec, labels, num_groups, xp):
    """Computes total cost (data + model) on GPU/CPU."""
    n, k = labels.shape[0], D_is1.shape[0]
    membership = xp.zeros((n, num_groups), dtype=xp.float32)
    membership[xp.arange(n), labels] = 1.0
    
    W_grant = xp.zeros((k, num_groups, num_groups), dtype=xp.float32)
    W_specified = xp.zeros((k, num_groups, num_groups), dtype=xp.float32)
    
    chunk_size = 2000
    for row_start in range(0, n, chunk_size):
        row_end = min(row_start + chunk_size, n)
        mc = membership[row_start:row_end]
        for a in range(k):
            g_sums = D_is1[a, row_start:row_end].astype(xp.float32) @ membership
            s_sums = D_spec[a, row_start:row_end].astype(xp.float32) @ membership
            W_grant[a] += mc.T @ g_sums
            W_specified[a] += mc.T @ s_sums
    
    cost = get_entropy_wildcard(W_grant, W_specified, xp)
    return cost + get_model_cost(n, k, num_groups), cost

# =============================================================================
# 2. IMPROVED BIT-PACKING WITH PER-ACTION SEPARATION
# =============================================================================

def pack_bits_per_action_gpu(D_is1, D_spec, xp):
    """
    Pack per-action policy entries into 64-bit bitmasks on GPU.
    
    Bit layout per entity: [outgoing bits 0..n-1 | incoming bits n..2n-1]
    packed into ceil(2n/64) 64-bit words.
    
    Args:
        D_is1:  (k, n, n) grant count tensor.
        D_spec: (k, n, n) specified count tensor (grants + denies).
        xp:     numpy or cupy.
    Returns:
        allow_masks: (k, n, n_words) uint64 bitmasks at grant positions.
        deny_masks:  (k, n, n_words) uint64 bitmasks at deny positions.
    """
    k, n, _ = D_is1.shape
    bits_per_entity = 2 * n  # n outgoing + n incoming
    n_words = (bits_per_entity + 63) // 64
    
    allow_masks = xp.zeros((k, n, n_words), dtype=xp.uint64)
    deny_masks  = xp.zeros((k, n, n_words), dtype=xp.uint64)
    
    pack_kernel_code = r'''
    extern "C" __global__
    void pack_action_bits(const int n, const int n_words,
                         const unsigned char* is1,
                         const unsigned char* spec,
                         unsigned long long* allow_out,
                         unsigned long long* deny_out) {
        long long u = blockIdx.x;
        long long v = (long long)blockIdx.y * blockDim.x + threadIdx.x;
        if (u >= n || v >= n) return;
        
        long long idx = u * n + v;
        unsigned char val_is1 = is1[idx];
        unsigned char val_spec = spec[idx];
        
        if (val_spec == 0) return;  // wildcard
        
        // Word indices
        long long out_word = v / 64;
        int out_bit = v % 64;
        
        long long in_word = (n + u) / 64;
        int in_bit = (n + u) % 64;
        
        if (val_is1) {
            atomicOr(&allow_out[u * n_words + out_word], (1ULL << out_bit));
            atomicOr(&allow_out[v * n_words + in_word], (1ULL << in_bit));
        } else {
            atomicOr(&deny_out[u * n_words + out_word], (1ULL << out_bit));
            atomicOr(&deny_out[v * n_words + in_word], (1ULL << in_bit));
        }
    }
    '''
    
    mod = cp.RawModule(code=pack_kernel_code)
    pack_kernel = mod.get_function('pack_action_bits')
    
    # Sync and clear pool to avoid fragmentation before packing
    xp.cuda.runtime.deviceSynchronize()
    if xp == cp: cp.get_default_memory_pool().free_all_blocks()
    
    threads = 256
    for a in range(k):
        grid_x = n
        grid_y = (n + threads - 1) // threads
        
        # Ensure contiguous data for kernel
        d_is1_a = xp.ascontiguousarray((D_is1[a] > 0).astype(xp.uint8))
        d_spec_a = xp.ascontiguousarray((D_spec[a] > 0).astype(xp.uint8))
        
        pack_kernel((grid_x, grid_y), (threads,), 
                   (n, n_words, d_is1_a, d_spec_a, allow_masks[a], deny_masks[a]))
        
        del d_is1_a, d_spec_a
    
    xp.cuda.runtime.deviceSynchronize()
    return allow_masks, deny_masks

# =============================================================================
# 3. CUDA KERNELS FOR DISTANCE COMPUTATION
# =============================================================================

HAMMING_KERNEL_CODE = r'''
extern "C" __global__
void compute_hamming_distance(const int n_items, const int k, const int n, const int n_words,
                              const unsigned long long* allow_masks,
                              const unsigned long long* deny_masks,
                              const int* indices,
                              int seed1_idx, int seed2_idx,
                              unsigned char* assignments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_items) return;
    
    int u = indices[idx];
    unsigned long long dist1 = 0, dist2 = 0;
    
    long long n_x_words = (long long)n * n_words;
    
    for (int a = 0; a < k; ++a) {
        long long a_off = (long long)a * n_x_words;
        for (int w = 0; w < n_words; ++w) {
            unsigned long long u_a = allow_masks[a_off + (long long)u * n_words + w];
            unsigned long long u_d = deny_masks[a_off + (long long)u * n_words + w];
            
            unsigned long long s1_a = allow_masks[a_off + (long long)seed1_idx * n_words + w];
            unsigned long long s1_d = deny_masks[a_off + (long long)seed1_idx * n_words + w];
            
            unsigned long long s2_a = allow_masks[a_off + (long long)seed2_idx * n_words + w];
            unsigned long long s2_d = deny_masks[a_off + (long long)seed2_idx * n_words + w];
            
            // Conflict Distance: Count positions where one is 1 and other is 0
            // Original logic: (p0 @ p1.T) + (p1 @ p0.T)
            dist1 += __popcll(u_a & s1_d) + __popcll(u_d & s1_a);
            dist2 += __popcll(u_a & s2_d) + __popcll(u_d & s2_a);
        }
    }
    assignments[idx] = (dist1 > dist2) ? 1 : 0;
}

extern "C" __global__
void compute_pairwise_hamming(const int n_samples, const int k, const int n, const int n_words,
                              const unsigned long long* allow_masks,
                              const unsigned long long* deny_masks,
                              const int* sample_indices,
                              float* dist_matrix) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples || j >= n_samples || j <= i) return;
    
    int u = sample_indices[i];
    int v = sample_indices[j];
    unsigned long long dist = 0;
    long long n_x_words = (long long)n * n_words;
    
    for (int a = 0; a < k; ++a) {
        long long a_off = (long long)a * n_x_words;
        for (int w = 0; w < n_words; ++w) {
            unsigned long long u_a = allow_masks[a_off + (long long)u * n_words + w];
            unsigned long long u_d = deny_masks[a_off + (long long)u * n_words + w];
            unsigned long long v_a = allow_masks[a_off + (long long)v * n_words + w];
            unsigned long long v_d = deny_masks[a_off + (long long)v * n_words + w];
            
            // Symmetric Conflict Distance
            dist += __popcll(u_a & v_d) + __popcll(u_d & v_a);
        }
    }
    dist_matrix[i * n_samples + j] = (float)dist;
}
'''

# =============================================================================
# 4. CORE COMPRESSOR LOGIC (MDL-Classic-Precise)
# =============================================================================

def _compute_data_cost(D_is1, D_spec, labels, num_groups, xp, chunk_size=2000):
    """Compute data cost for a given labeling."""
    n = len(labels)
    k = D_is1.shape[0]
    membership = xp.zeros((n, num_groups), dtype=xp.float32)
    membership[xp.arange(n), labels] = 1.0
    
    W_grant     = xp.zeros((k, num_groups, num_groups), dtype=xp.float32)
    W_specified = xp.zeros((k, num_groups, num_groups), dtype=xp.float32)
    
    for row_start in range(0, n, chunk_size):
        row_end = min(row_start + chunk_size, n)
        membership_chunk = membership[row_start:row_end]
        for a in range(k):
            grant_sums = D_is1[a][row_start:row_end].astype(xp.float32) @ membership
            spec_sums  = D_spec[a][row_start:row_end].astype(xp.float32) @ membership
            W_grant[a]     += membership_chunk.T @ grant_sums
            W_specified[a] += membership_chunk.T @ spec_sums
    
    cost = get_entropy_wildcard(W_grant, W_specified, xp)
    del membership, W_grant, W_specified
    return cost

def inner_loop_vectorized(D_is1, D_spec, group_labels, num_groups, xp):
    """
    Batch-refinement of domain assignments to minimize MDL coding cost.
    Uses MDL-Classic-Precise logic: re-evaluates cost from scratch after
    moves and rolls back if no objective improvement is achieved.
    
    Iteratively moves entities between groups to reduce the total
    Shannon entropy of the domain-block weight matrices.
    
    Args:
        D_is1:        (k, n, n) grant count tensor.
        D_spec:       (k, n, n) specified count tensor.
        group_labels: (n,) int32 current group assignment for each entity.
        num_groups:   Number of groups (m).
        xp:           numpy or cupy.
    Returns:
        (group_labels, data_cost, num_passes): Refined assignments, final entropy cost, and pass count.
    """
    k = D_is1.shape[0]
    n = len(group_labels)
    chunk_size = 2000  # Keep float32 slices within VRAM limits
    
    # Track the actual cost across passes (Precise mode)
    current_total_cost = _compute_data_cost(D_is1, D_spec, group_labels, num_groups, xp) + get_model_cost(n, k, num_groups)

    for pass_idx in range(10): 
        # One-hot membership matrix: membership[i, g] = 1 iff entity i is in group g
        membership = xp.zeros((n, num_groups), dtype=xp.float32)
        membership[xp.arange(n), group_labels] = 1.0
        group_sizes = xp.bincount(group_labels, minlength=num_groups).astype(xp.float32)
        
        # Snapshot for rollback (Precise mode)
        old_labels_snapshot = group_labels.copy()
        
        # PASS 1: Compute domain-pair weight matrices W_grant and W_specified
        #   W_grant[a, g1, g2]     = total grant counts for (group g1) -> (group g2)
        #   W_specified[a, g1, g2] = total specified counts for same
        W_grant     = xp.zeros((k, num_groups, num_groups), dtype=xp.float32)
        W_specified = xp.zeros((k, num_groups, num_groups), dtype=xp.float32)
        
        for row_start in range(0, n, chunk_size):
            row_end = min(row_start + chunk_size, n)
            membership_chunk = membership[row_start:row_end]
            for a in range(k):
                # grant_sums[i, g] = sum of grants from entity i to all entities in group g
                grant_sums = D_is1[a][row_start:row_end].astype(xp.float32) @ membership
                spec_sums  = D_spec[a][row_start:row_end].astype(xp.float32) @ membership
                # Accumulate into (num_groups x num_groups) weight matrix
                W_grant[a]     += membership_chunk.T @ grant_sums
                W_specified[a] += membership_chunk.T @ spec_sums
                del grant_sums, spec_sums
            if xp == cp: cp.get_default_memory_pool().free_all_blocks()

        # Compute per-cell coding costs from the grant density
        grant_ratio = W_grant / W_specified.clip(min=1)
        cost_grant = -xp.log2(xp.clip(grant_ratio, 1e-12, 1.0 - 1e-12))
        cost_deny  = -xp.log2(xp.clip(1 - grant_ratio, 1e-12, 1.0 - 1e-12))
        
        # PASS 2: Compute coding_cost[i, g] = total bits if entity i were in group g
        coding_cost = xp.zeros((n, num_groups), dtype=xp.float32)
        for row_start in range(0, n, chunk_size):
            row_end = min(row_start + chunk_size, n)
            for a in range(k):
                # Outgoing interactions
                grants_out = xp.ascontiguousarray(D_is1[a][row_start:row_end].astype(xp.float32)) @ membership
                specs_out  = xp.ascontiguousarray(D_spec[a][row_start:row_end].astype(xp.float32)) @ membership
                coding_cost[row_start:row_end] += grants_out @ cost_grant[a].T + (specs_out - grants_out) @ cost_deny[a].T
                del grants_out, specs_out
                
                # Incoming interactions
                grants_in = xp.ascontiguousarray(D_is1[a][:, row_start:row_end].T.astype(xp.float32)) @ membership
                specs_in  = xp.ascontiguousarray(D_spec[a][:, row_start:row_end].T.astype(xp.float32)) @ membership
                coding_cost[row_start:row_end] += grants_in @ cost_grant[a] + (specs_in - grants_in) @ cost_deny[a]
                del grants_in, specs_in
                
            if xp == cp: cp.get_default_memory_pool().free_all_blocks()
        
        # For each entity, compute savings from moving to the best alternative group
        curr_costs = xp.take_along_axis(coding_cost, group_labels[:, None], axis=1).squeeze(1)
        savings = curr_costs[:, None] - coding_cost
        
        best_target = xp.argmax(savings, axis=1).astype(xp.int32)
        best_savings = xp.max(savings, axis=1)
        
        del coding_cost, savings, curr_costs
        
        candidates = xp.where(best_savings > 1e-2)[0]
        if len(candidates) == 0: 
            del best_target, best_savings, membership
            break
        
        # Move top candidates (by savings) to their best target group
        move_limit = max(16, n // 10)
        top_indices = xp.argsort(best_savings[candidates])[-move_limit:]
        nodes_to_move = candidates[top_indices]
        
        moves_applied = 0
        nodes_to_move_cpu = to_cpu(nodes_to_move)
        best_target_cpu = to_cpu(best_target)
        for node_idx in nodes_to_move_cpu:
            target_g = int(best_target_cpu[node_idx])
            old_g = int(group_labels[node_idx])
            if group_sizes[old_g] > 1:  # Don't empty a group
                group_labels[node_idx] = target_g
                group_sizes[old_g] -= 1
                group_sizes[target_g] += 1
                moves_applied += 1
        
        del nodes_to_move, top_indices, candidates, best_target, best_savings, membership
        if xp == cp: cp.get_default_memory_pool().free_all_blocks()
        if moves_applied == 0: break
        
        # [Precise] Re-evaluate cost from scratch and rollback if no improvement
        actual_total_cost, _ = evaluate_labels(D_is1, D_spec, group_labels, num_groups, xp)
        if actual_total_cost >= current_total_cost:
            group_labels[:] = old_labels_snapshot
            break
        else:
            current_total_cost = actual_total_cost
        
    # Final data cost calculation
    cost = current_total_cost - get_model_cost(n, k, num_groups)
    return group_labels, cost, pass_idx + 1

def split_group_hamming_gpu(allow_masks, deny_masks, group_indices, xp):
    """
    GPU-accelerated splitting using Conflict Distance across ALL actions.
    Matches original mathematical behavior exactly.
    """
    if len(group_indices) < 2: 
        return xp.array([0], dtype=xp.uint8)
    
    k, n, n_words = allow_masks.shape
    n_group = len(group_indices)
    
    # Sample for seed selection (Using original logic exactly)
    n_samples = min(50, n_group)
    s_idx = xp.array(np.random.choice(n_group, n_samples, replace=False), dtype=xp.int32)
    sample_indices = group_indices[s_idx].astype(xp.int32)
    
    # Compute pairwise distances on GPU
    dist_mat = xp.zeros((n_samples, n_samples), dtype=xp.float32)
    
    if HAS_GPU and xp == cp:
        mod = cp.RawModule(code=HAMMING_KERNEL_CODE)
        pairwise_kernel = mod.get_function('compute_pairwise_hamming')
        
        threads = 32
        grid_x = (n_samples + threads - 1) // threads
        pairwise_kernel((grid_x, n_samples), (threads,), 
                       (n_samples, k, n, n_words, 
                        allow_masks, deny_masks,
                        sample_indices, dist_mat))
    else:
        # CPU fallback (Match Conflict Logic)
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                u, v = int(sample_indices[i]), int(sample_indices[j])
                dist = 0
                for a in range(k):
                    dist += xp.sum(xp.bitwise_and(allow_masks[a, u], deny_masks[a, v]).astype(xp.int32))
                    dist += xp.sum(xp.bitwise_and(deny_masks[a, u], allow_masks[a, v]).astype(xp.int32))
                dist_mat[i, j] = float(dist)
    
    # Find furthest pair
    flat_idx = xp.argmax(dist_mat)
    s1_local, s2_local = xp.unravel_index(flat_idx, dist_mat.shape)
    seed1_idx = int(sample_indices[s1_local])
    seed2_idx = int(sample_indices[s2_local])
    
    # Assign all nodes to nearest seed
    assignments = xp.zeros(n_group, dtype=xp.uint8)
    
    if HAS_GPU and xp == cp:
        mod = cp.RawModule(code=HAMMING_KERNEL_CODE)
        assign_kernel = mod.get_function('compute_hamming_distance')
        
        threads = 256
        blocks = (n_group + threads - 1) // threads
        assign_kernel((blocks,), (threads,), 
                     (n_group, k, n, n_words,
                      allow_masks, deny_masks,
                      group_indices.astype(xp.int32),
                      seed1_idx, seed2_idx,
                      assignments))
    else:
        # CPU fallback (Match Conflict Logic)
        for idx in range(n_group):
            u = int(group_indices[idx])
            d1, d2 = 0, 0
            for a in range(k):
                d1 += xp.sum(xp.bitwise_and(allow_masks[a, u], deny_masks[a, seed1_idx]).astype(xp.int32))
                d1 += xp.sum(xp.bitwise_and(deny_masks[a, u], allow_masks[a, seed1_idx]).astype(xp.int32))
                d2 += xp.sum(xp.bitwise_and(allow_masks[a, u], deny_masks[a, seed2_idx]).astype(xp.int32))
                d2 += xp.sum(xp.bitwise_and(deny_masks[a, u], allow_masks[a, seed2_idx]).astype(xp.int32))
            assignments[idx] = 1 if d1 > d2 else 0
    
    return assignments

def split_group_hamming(D_all, group_indices, xp):
    """Splits a group based on furthest-pair Hamming distance seeds."""
    if len(group_indices) < 2: return xp.array([0], dtype=xp.uint8)
    
    # 1. Select seeds using a subset of actions/entities for speed
    n_group = len(group_indices)
    n_samples = min(100, n_group)
    sample_indices = group_indices[xp.linspace(0, n_group-1, n_samples).astype(xp.int32)]
    
    # Flatten D for Hamming computation: (n_samples, k * n * 2)
    # Using row and column slices to capture full entity context
    def get_features(indices):
        f = []
        for a in range(D_all.shape[0]):
            f.append(D_all[a][indices].astype(xp.int8))   # Outgoing
            f.append(D_all[a][:, indices].T.astype(xp.int8)) # Incoming
        return xp.concatenate(f, axis=1)

    feats = get_features(sample_indices)
    
    # Seed 1: Pick one at random (first in sample)
    seed1_idx = 0
    
    # Seed 2: furthest from seed1
    d1 = xp.sum(xp.abs(feats - feats[seed1_idx]), axis=1)
    seed2_idx = int(xp.argmax(d1))
    
    # 2. Assign all in group
    full_feats = get_features(group_indices)
    s1 = feats[seed1_idx]
    s2 = feats[seed2_idx]
    
    dist1 = xp.sum(xp.abs(full_feats - s1), axis=1)
    dist2 = xp.sum(xp.abs(full_feats - s2), axis=1)
    
    return (dist1 > dist2).astype(xp.uint8)

def outer_loop_autopart(D_is1, D_spec, xp, 
                        allow_masks=None, deny_masks=None,
                        base_data=None, verbose=True):
    """AutoPart-Classic outer loop with separate split/refine commitment checks.
    
    At each iteration:
      1. Splits all groups.
      2. Checks if the raw split improves MDL cost (commits if yes).
      3. Runs refinement and checks if that further improves (commits if yes).
      4. Stops if neither split nor refinement improved.
    
    Uses MDL-Classic-Precise as the default algorithm.
    
    Args:
        D_is1:       (k, n, n) grant count tensor.
        D_spec:      (k, n, n) specified count tensor.
        xp:          numpy or cupy.
        allow_masks: (k, n, n_words) GPU bitmasks for grant positions (optional).
        deny_masks:  (k, n, n_words) GPU bitmasks for deny positions (optional).
        base_data:   Raw data tensor for CPU fallback splitting (optional).
        verbose:     Print progress.
    Returns:
        (group_labels, num_groups, num_inner_loops, num_outer_loops):
            Final assignments, domain count, total refinement passes, and outer iterations.
    """
    k, n, _ = D_is1.shape
    num_groups = 1
    group_labels = xp.zeros(n, dtype=xp.int32)
    
    # Calculate initial data cost (single-group baseline)
    total_grants    = xp.zeros(k, dtype=xp.float64)
    total_specified = xp.zeros(k, dtype=xp.float64)
    for a in range(k):
        total_grants[a]    = float(xp.sum(D_is1[a].astype(xp.float32)))
        total_specified[a] = float(xp.sum(D_spec[a].astype(xp.float32)))
    
    curr_data_cost = get_entropy_wildcard(total_grants, total_specified, xp)
    del total_grants, total_specified
    if xp == cp: cp.get_default_memory_pool().free_all_blocks()
    
    curr_total_cost = get_model_cost(n, k, 1) + curr_data_cost
    num_inner_loops = 0
    num_outer_loops = 0
    
    while num_groups < n:
        num_outer_loops += 1
        new_labels = group_labels.copy()
        current_groups = xp.unique(group_labels)
        next_id = num_groups
        splits_performed = 0
        
        for group_id in to_cpu(current_groups):
            indices = xp.where(group_labels == group_id)[0]
            if len(indices) < 2: continue
            
            # Choose splitting method (GPU bitmask or CPU fallback)
            if allow_masks is not None:
                split_labels = split_group_hamming_gpu(allow_masks, deny_masks, indices, xp)
            elif base_data is not None:
                split_labels = split_group_hamming(base_data, indices, xp)
            else:
                raise ValueError("Error: Neither bit-masks nor base data available for splitting.")
            
            if xp.any(split_labels == 1):
                new_labels[indices[split_labels == 1].astype(xp.int32)] = next_id
                next_id += 1
                splits_performed += 1
        
        if splits_performed == 0: break
        new_num_groups = next_id
        
        improved = False
        
        # [Classic] Split Check: commit split if it improves cost
        split_data_cost = _compute_data_cost(D_is1, D_spec, new_labels, new_num_groups, xp)
        split_total_cost = get_model_cost(n, k, new_num_groups) + split_data_cost
        
        if split_total_cost < curr_total_cost:
            group_labels = new_labels.copy()
            num_groups = new_num_groups
            curr_total_cost = split_total_cost
            improved = True
            if verbose: print(f"    - m={num_groups:3d} | Split | Cost: {curr_total_cost:12.2f}")
        
        # [Classic] Refinement Check: commit refinement if it further improves
        refined_labels, refined_data_cost, passes = inner_loop_vectorized(
            D_is1, D_spec, new_labels, new_num_groups, xp
        )
        num_inner_loops += passes
        refined_total_cost = get_model_cost(n, k, new_num_groups) + refined_data_cost
        
        if refined_total_cost < curr_total_cost:
            group_labels = xp.asarray(refined_labels, dtype=xp.int32)
            num_groups = new_num_groups
            curr_total_cost = refined_total_cost
            improved = True
            if verbose: print(f"    - m={num_groups:3d} | Refined | Cost: {curr_total_cost:12.2f}")
        
        if not improved: break
        if xp == cp: cp.get_default_memory_pool().free_all_blocks()
            
    return group_labels, num_groups, num_inner_loops, num_outer_loops

# =============================================================================
# 5. H CONSTRUCTION
# =============================================================================

def construct_H(D_is1, D_spec, group_labels, xp):
    """
    Construct summary policy H from grouping with 'deny-by-default' behavior.
    
    For each domain-block (i, j) in each action a:
      W_grant[a,i,j]     = total grant counts across entity pairs (u in Di, v in Dj)
      W_specified[a,i,j]  = total specified (non-wildcard) counts
    
    Decision rule (Deny-by-Default majority):
      H[a,i,j] = 1  iff  W_grant > W_specified * 0.5  (grants outnumber denies)
    
    Args:
        D_is1:        (k, n, n) grant count tensor.
        D_spec:       (k, n, n) specified count tensor.
        group_labels: (n,) int32 group assignment for each entity.
        xp:           numpy or cupy.
    Returns:
        H: (k, m, m) uint8 summary policy.
    """
    k = D_is1.shape[0]
    n = len(group_labels)
    m = int(xp.max(group_labels)) + 1
    
    W_grant     = xp.zeros((k, m, m), dtype=xp.float32)
    W_specified = xp.zeros((k, m, m), dtype=xp.float32)
    
    chunk_size = 10000
    for a in range(k):
        v1_a = D_is1[a]
        vs_a = D_spec[a]
        
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            v1_chunk = v1_a[i_start:i_end]
            vs_chunk = vs_a[i_start:i_end]
            labels_i = group_labels[i_start:i_end]
            
            idx_2d = (labels_i[:, None], group_labels[None, :])
            xp.add.at(W_grant[a], idx_2d, v1_chunk)
            xp.add.at(W_specified[a], idx_2d, vs_chunk)
            
            del v1_chunk, vs_chunk, idx_2d
    
    # Deny-by-default: H=1 iff grants > denies, i.e. W_grant > (W_specified - W_grant)
    # Rearranged: W_grant > W_specified * 0.5
    H = (W_grant > (W_specified.clip(min=1) * 0.5)).astype(xp.uint8)
    
    del W_grant, W_specified
    return H

# =============================================================================
# 6. METRICS
# =============================================================================

def calculate_metrics(H, pi_arr, G_orig):
    """
    Standardized Metrics Calculation.
    H: Summary policy (k, m, m)
    pi_arr: Mapping array (n,) where pi_arr[i] is domain index of entity i in H
    G_orig: Original policy (k, n, n).
    """
    if hasattr(H, 'get'): H = H.get()
    if hasattr(pi_arr, 'get'): pi_arr = pi_arr.get()
    
    k, n, _ = G_orig.shape
    tp, tn, fp, fn = 0, 0, 0, 0
    
    chunk_size = 5000 
    for a in range(k):
        H_a = H[a]
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            pi_i = pi_arr[i_start:i_end]
            for j_start in range(0, n, chunk_size):
                j_end = min(j_start + chunk_size, n)
                pi_j = pi_arr[j_start:j_end]
                
                G_chunk = G_orig[a, i_start:i_end, j_start:j_end]
                H_chunk = H_a[np.ix_(pi_i, pi_j)]
                                
                mask = (G_chunk != 2)
                if not np.any(mask): continue
                
                truth_vals = G_chunk[mask]
                pred_vals  = H_chunk[mask]
                
                tp += np.sum((truth_vals == 1) & (pred_vals == 1))
                tn += np.sum((truth_vals == 0) & (pred_vals == 0))
                fp += np.sum((truth_vals == 0) & (pred_vals == 1))
                fn += np.sum((truth_vals == 1) & (pred_vals == 0))
                
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

# =============================================================================
# 7. MAIN
# =============================================================================

####################################################################################
# Usage:
# python3 mdl_compressor.py noise_policy original_policy [--device {cpu,gpu}] [--verbose]
# 
# noise_policy: Path to the noise policy (.npy) - MUST be bit-packed (uint8)
# original_policy: Path to the original policy (.npy) for testing
# --device: Execution mode, either 'cpu' or 'gpu' (default: 'gpu')
# --verbose: Enable verbose progress output
####################################################################################
def main():
    parser = argparse.ArgumentParser(description='MDL-based Policy Compressor')
    parser.add_argument('noise_policy', help='Noise policy (.npy)')
    parser.add_argument('original_policy', help='Original policy (.npy)')
    
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Execution mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose progress output')
    
    args = parser.parse_args()
    
    # 0. Load Data (Memory efficient)
    data_raw = np.load(args.noise_policy, mmap_mode='r')
    if data_raw.ndim == 4: data_raw = data_raw.squeeze(-1)
    k, n, _ = data_raw.shape

    # Auto-downgrade logic
    if args.device == 'gpu' and not HAS_GPU:
        print("Warning: GPU requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    xp = get_xp(args.device)
    print(f"\nMDL: n={n}, k={k}, device={args.device.upper()}")
    
    # 1. Prepare Tensors
    print(f"[*] Preparing data tensors...")
    start_time = time.time()
    
    if data_raw.dtype != np.uint8:
        raise ValueError("Error: Standard 'noise_policy' (int8) detected. "
                         "mdl_compressor only handles bit-packed 'noise_policy' (uint8). "
                         "Please use gc_compressor.py for standard data.")
                         
    print("    - Detected Bit-Packed (Mode B) policy.")
    D_packed = xp.array(data_raw)
    del data_raw
    
    # Extract counts (Vectorized & In-place to save memory)
    # grants and denies are uint8 (Max 15)
    D_is1 = (D_packed >> 4).astype(xp.uint8)
    denies = (D_packed & 0x0F).astype(xp.uint8)
    
    # Storage optimized: Keep in uint8 for VRAM efficiency
    D_spec = D_is1 + denies # In-place uint8 add: counts of (0s + 1s)
    
    # Note: In bit-packed mode, we use GPU bitmasks for the splitting seeds,
    # so we don't need the O(n^2) decision tensor used by the CPU fallback path.
    
    # Explicitly clear intermediate variables
    del D_packed, denies
    if xp == cp: 
        cp.get_default_memory_pool().free_all_blocks()
        xp.cuda.runtime.deviceSynchronize()


    print(f"[*] Packing bits for GPU-accelerated splitting...")
    allow_masks, deny_masks = pack_bits_per_action_gpu(D_is1, D_spec, xp)
    print(f"    - Packed into shape: {allow_masks.shape}")
    if xp == cp: cp.get_default_memory_pool().free_all_blocks()
    
    # 2. Outer Loop (MDL-Classic-Precise)
    print(f"[*] Starting Hierarchical Divisive MDL loop (Classic-Precise)...")
    final_G, final_m, num_inner, num_outer = outer_loop_autopart(
        D_is1, D_spec, xp, 
        allow_masks, deny_masks,
        verbose=args.verbose)
    
    # 3. Final Summary Construction
    print(f"[*] Constructing final summary H...")
    H_res = construct_H(D_is1, D_spec, final_G, xp)
    
    elapsed = time.time() - start_time
    print(f"    - Done in {elapsed:.2f} s. Domains found: {final_m}")
    
    # 4. Evaluation
    print(f"[*] Evaluating against original policy...")
    G_orig = np.load(args.original_policy, mmap_mode='r')
    if G_orig.ndim == 4: G_orig = G_orig.squeeze(-1)
    
    acc, prec, rec, f1 = calculate_metrics(H_res, final_G, G_orig)
    
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Final Domains: {final_m}")
    print(f"Compression Time: {elapsed:.2f} s")
    print("-" * 40)

if __name__ == "__main__":
    main()
