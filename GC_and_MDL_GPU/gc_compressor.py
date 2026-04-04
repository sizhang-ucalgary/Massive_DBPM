import time
import argparse
import numpy as np
import numba as nb

# Optional GPU support
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        HAS_GPU = True
    else:
        HAS_GPU = False
except Exception:
    HAS_GPU = False

# =============================================================================
# 1. CONFLICT GRAPH CONSTRUCTION
# =============================================================================

@nb.njit
def _has_conflict(u_slice, v_slice):
    """Check if two entity slices have conflicting hard decisions.
    
    A conflict exists when one slice has a grant (1) and the other a deny (0)
    at the same position. Wildcards (2) never cause conflicts.
    """
    for i in range(len(u_slice)):
        if (u_slice[i] != 2) and (v_slice[i] != 2) and (u_slice[i] != v_slice[i]):
            return True
    return False

@nb.njit
def create_conflict_graph_numba(partial_policy):
    """
    CPU-based conflict graph construction via pairwise comparison.
    
    Two entities conflict if they have opposing hard decisions (grant vs deny)
    for ANY action, in either the outgoing (row) or incoming (column) direction.
    
    Args:
        partial_policy: (k, n, n) array where 0=deny, 1=grant, 2=wildcard.
    Returns:
        (u_edges, v_edges): int32 edge lists of upper-triangle conflicts.
    """
    k = partial_policy.shape[0]
    n = partial_policy.shape[1]
    adjacency = np.zeros((n, n), dtype=np.bool_)
    for a in range(k):
        for u in range(n):
            for v in range(u + 1, n):
                if adjacency[u, v]: continue
                # Check both outgoing (row) and incoming (column) directions
                if _has_conflict(partial_policy[a, u], partial_policy[a, v]) or \
                   _has_conflict(partial_policy[a, :, u], partial_policy[a, :, v]):
                    adjacency[u, v] = True
                    adjacency[v, u] = True
    u_edges, v_edges = np.where(np.triu(adjacency, k=1))
    return u_edges.astype(np.int32), v_edges.astype(np.int32)

def pack_bits_per_action_gpu(partial_policy):
    """
    Pack per-action policy entries into 64-bit bitmasks on GPU.
    
    Bit layout per entity: [outgoing bits 0..n-1 | incoming bits n..2n-1]
    packed into ceil(2n/64) 64-bit words.
    
    Args:
        partial_policy: (k, n, n) array where 0=deny, 1=grant, 2=wildcard.
    Returns:
        allow_masks: (k, n, n_words) uint64 bitmasks at grant positions.
        deny_masks:  (k, n, n_words) uint64 bitmasks at deny positions.
    """
    k, n, _ = partial_policy.shape
    bits_per_entity = 2 * n  # n outgoing + n incoming
    n_words = (bits_per_entity + 63) // 64
    
    allow_masks = cp.zeros((k, n, n_words), dtype=cp.uint64)
    deny_masks  = cp.zeros((k, n, n_words), dtype=cp.uint64)
    
    pack_kernel_code = r'''
    extern "C" __global__
    void pack_action_bits(const int n, const int n_words,
                         const unsigned char* is1,
                         const unsigned char* is0,
                         unsigned long long* allow_out,
                         unsigned long long* deny_out) {
        int u = blockIdx.x;
        int v = blockIdx.y * blockDim.x + threadIdx.x;
        if (u >= n || v >= n) return;
        
        long long idx = (long long)u * n + v;
        unsigned char val_is1 = is1[idx];
        unsigned char val_is0 = is0[idx];
        
        if (val_is1 == 0 && val_is0 == 0) return;  // wildcard
        
        // Outgoing edge u->v at bit position v
        int out_word = v / 64;
        int out_bit = v % 64;
        
        // Incoming edge v->u at bit position (n + u)
        int in_word = (n + u) / 64;
        int in_bit = (n + u) % 64;
        
        if (val_is1) {
            atomicOr(&allow_out[(long long)u * n_words + out_word], (1ULL << out_bit));
            atomicOr(&allow_out[(long long)v * n_words + in_word], (1ULL << in_bit));
        }
        if (val_is0) {
            atomicOr(&deny_out[(long long)u * n_words + out_word], (1ULL << out_bit));
            atomicOr(&deny_out[(long long)v * n_words + in_word], (1ULL << in_bit));
        }
    }
    '''
    
    mod = cp.RawModule(code=pack_kernel_code)
    pack_kernel = mod.get_function('pack_action_bits')
    
    threads = 256
    for a in range(k):
        grid_x = n
        grid_y = (n + threads - 1) // threads
        
        # Convert to CuPy array, then extract binary grant/deny masks
        action_slice = cp.asarray(partial_policy[a])
        is_grant = (action_slice == 1).astype(cp.uint8)
        is_deny  = (action_slice == 0).astype(cp.uint8)
        
        pack_kernel((grid_x, grid_y), (threads,), 
                   (n, n_words, is_grant, is_deny, allow_masks[a], deny_masks[a]))
        
        del action_slice, is_grant, is_deny
    
    return allow_masks, deny_masks

def create_conflict_graph_gpu(partial_policy):
    """
    Builds conflict graph on GPU using bit-parallel conflict detection.
    
    Two entities conflict if they have different hard decisions for ANY action.
    
    Args:
        partial_policy: (k, n, n) array where 0=deny, 1=grant, 2=wildcard.
    Returns:
        adj_matrix: (n, n) uint8 CuPy adjacency matrix of the conflict graph.
    """
    k, n, _ = partial_policy.shape
    
    print("    - Packing bits on GPU (per-action)...")
    allow_masks, deny_masks = pack_bits_per_action_gpu(partial_policy)
    n_words = allow_masks.shape[2]
    
    print("    - Building conflict graph...")
    
    adj_matrix = cp.zeros((n, n), dtype=cp.uint8)
    
    kernel_code = r'''
    extern "C" __global__
    void detect_conflicts_per_action(const int n, const int k, const int n_words,
                                     const unsigned long long* allow_masks,
                                     const unsigned long long* deny_masks,
                                     unsigned char* adj_matrix,
                                     const int row_start, const int row_end) {
        
        int u_local = blockIdx.y;
        int u = u_local + row_start;
        int v = blockIdx.x * blockDim.x + threadIdx.x;

        if (u >= row_end || v >= n || v <= u) return;

        bool conflict = false;
        
        // Check conflicts for EACH action separately
        for (int a = 0; a < k && !conflict; ++a) {
            long long offset_u = (long long)a * n * n_words + (long long)u * n_words;
            long long offset_v = (long long)a * n * n_words + (long long)v * n_words;
            
            for (int w = 0; w < n_words; ++w) {
                unsigned long long u_allow = allow_masks[offset_u + w];
                unsigned long long u_deny = deny_masks[offset_u + w];
                unsigned long long v_allow = allow_masks[offset_v + w];
                unsigned long long v_deny = deny_masks[offset_v + w];
                
                // Conflict if: (u_allow & v_deny) OR (v_allow & u_deny)
                if ((u_allow & v_deny) || (v_allow & u_deny)) {
                    conflict = true;
                    break;
                }
            }
        }

        if (conflict) {
            long long idx1 = (long long)u * n + v;
            long long idx2 = (long long)v * n + u;
            adj_matrix[idx1] = 1;
            adj_matrix[idx2] = 1;
        }
    }
    '''
    detect_kernel = cp.RawKernel(kernel_code, 'detect_conflicts_per_action')
    
    # Adaptive chunk size based on graph size
    CHUNK_SIZE = min(max(500, n // 20), 2000)
    threads = 256
    blocks_v = (n + threads - 1) // threads
    
    print(f"    - Launching CUDA kernels ({n} vertices, {k} actions, {n_words} words)...")
    for i in range(0, n, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, n)
        actual_chunk = end - i
        
        detect_kernel((blocks_v, actual_chunk), (threads, 1), 
                     (n, k, n_words, allow_masks, deny_masks, adj_matrix, i, end))
        
    del allow_masks, deny_masks
    cp.get_default_memory_pool().free_all_blocks()
    
    return adj_matrix

# =============================================================================
# 2. GRAPH COLORING
# =============================================================================

@nb.njit
def greedy_color(indptr, indices, nodes_order):
    """Greedy Graph Coloring on CPU (Welsh-Powell ordering).
    
    Args:
        indptr:      CSR row pointer array (n+1,).
        indices:     CSR column index array.
        nodes_order: Vertex processing order (n,), typically by decreasing degree.
    Returns:
        colors: (n,) int32 array of color assignments.
    """
    n = len(nodes_order)
    colors = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        u = nodes_order[i]
        neighbor_colors = set()
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            if colors[v] != -1:
                neighbor_colors.add(colors[v])
        c = 0
        while True:
            if c not in neighbor_colors:
                colors[u] = c
                break
            c += 1
    return colors

def color_gpu_gm(adj_matrix, verbose=True):
    """
    Iterative Parallel Graph Coloring (IPGC) using the Gebremedhin-Manne (GM) 
    speculative approach. Implements the Vertex-Based Bit-Parallel (VB-BIT) MEX 
    logic with Largest Log-Degree First (LLF) priority (Deveci et al. 2016).
    
    Args:
        adj_matrix: (n, n) CuPy uint8 adjacency matrix.
        verbose:    Print iteration progress.
    Returns:
        colors: (n,) CuPy int32 color assignments.
    """    
    n = adj_matrix.shape[0]
    cp.get_default_memory_pool().free_all_blocks()

    # Convert to CSR format
    indptr = cp.zeros(n + 1, dtype=cp.int64)
    degrees = cp.sum(adj_matrix, axis=1, dtype=cp.int64)
    cp.cumsum(degrees, out=indptr[1:])
    
    total_edges = int(indptr[n])
    indices = cp.empty(total_edges, dtype=cp.int32)
    
    # Adaptive chunk size
    CHUNK_ROWS = min(max(1000, n // 50), 5000)
    for i in range(0, n, CHUNK_ROWS):
        end_r = min(i + CHUNK_ROWS, n)
        _, col_idx_local = cp.where(adj_matrix[i:end_r] > 0)
        indices[indptr[i] : indptr[end_r]] = col_idx_local.astype(cp.int32)
        del col_idx_local
    
    del adj_matrix, degrees
    
    # Compute priority weights using Largest Log-Degree First (LLF) strategy
    degrees = indptr[1:] - indptr[:-1]
    log_degrees = cp.ceil(cp.log2(degrees + 1)).astype(cp.float32)
    priority_weights = (log_degrees * 10.0) + cp.random.random(n).astype(cp.float32)
    
    colors = cp.full(n, -1, dtype=cp.int32)
    conflict_mask = cp.zeros(n, dtype=cp.bool_)

    # Algorithm 2: ASSIGN COLORS (Vertex-Based Bit-Parallel MEX logic)
    tentative_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void tentative(const int n, const long long* indptr, const int* indices, 
                   int* colors) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        
        int base_color = 0;
        while (true) {
            unsigned long long forbidden_mask = 0; // VB-BIT optimization
            for (long long i = indptr[idx]; i < indptr[idx+1]; ++i) {
                int nb_c = colors[indices[i]];
                if (nb_c >= base_color && nb_c < base_color + 64) {
                    forbidden_mask |= (1ULL << (nb_c - base_color));
                }
            }
            
            // hardware-accelerated MEX finding using __ffsll
            unsigned long long available_mask = ~forbidden_mask;
            if (available_mask != 0) {
                colors[idx] = base_color + (__ffsll(available_mask) - 1);
                break;
            }
            
            base_color += 64; // Sliding window for more than 64 colors
            if (base_color > n) break;
        }
    }
    ''', 'tentative')

    # Algorithm 3: DETECT CONFLICTS (Section IV-D)
    conflict_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void conflict(const int n, const long long* indptr, const int* indices, 
                  const int* colors, const float* priority_weights, bool* conflict_mask) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        
        int my_c = colors[idx];
        float my_w = priority_weights[idx];
        
        for (long long i = indptr[idx]; i < indptr[idx+1]; ++i) {
            int nb = indices[i];
            if (colors[nb] == my_c && (priority_weights[nb] > my_w || (priority_weights[nb] == my_w && nb > idx))) {
                conflict_mask[idx] = true;
                return;
            }
        }
    }
    ''', 'conflict')

    # Main iteration loop
    threads = 256
    blocks_n = (n + threads - 1) // threads
    
    iter_count = 0
    total_conflicts_resolved = 0
    
    if verbose:
        print(f"\n    [GM Iterations]")
        print(f"      {'Iter':<6} {'Conflicts':<10} {'Resolved':<10} {'Progress':<10}")
        print(f"      {'-'*50}")
    
    while True:
        if iter_count > 0 and not cp.any(conflict_mask):
            break
        
        tentative_kernel((blocks_n,), (threads,), (n, indptr, indices, colors))
        
        conflict_mask[:] = False
        conflict_kernel((blocks_n,), (threads,), (n, indptr, indices, colors, priority_weights, conflict_mask))
        
        num_conflicts = int(cp.sum(conflict_mask))
        num_resolved = n - num_conflicts
        total_conflicts_resolved += num_resolved
        
        if verbose:
            progress_pct = (total_conflicts_resolved / n) * 100
            print(f"      {iter_count:<6} {num_conflicts:<10} {num_resolved:<10} {progress_pct:>6.2f}%")
        
        colors[conflict_mask] = -1
        
        iter_count += 1
        if iter_count > max(1000, n):
            if verbose:
                print(f"      WARNING: Max iterations ({iter_count}) reached!")
            break
    
    if verbose:
        print(f"      {'-'*50}")
        print(f"      Converged in {iter_count} iterations")
        print(f"      Total nodes colored: {total_conflicts_resolved}")
        num_colors = int(cp.max(colors) + 1)
        avg_degree = float(cp.mean(degrees))
        print(f"      Colors used: {num_colors} (avg degree: {avg_degree:.1f})")
        
    return colors

# =============================================================================
# 3. H CONSTRUCTION
# =============================================================================

def construct_H(partial_policy, pi_dict, use_gpu_for_h=False):
    """
    Construct summary policy H from coloring with 'deny-by-default' behavior.
    """
    n = partial_policy.shape[1]
    k = partial_policy.shape[0]
    
    unique_colors = sorted(list(set(pi_dict.values())))
    color_to_idx = {color: idx for idx, color in enumerate(unique_colors)}
    m = len(unique_colors)
    
    if use_gpu_for_h and HAS_GPU:
        # Build a mapping: entity → domain index
        colors = cp.zeros(n, dtype=cp.int32)
        for vertex, color in pi_dict.items():
            colors[vertex] = color_to_idx[color]
            
        partial_policy_cp = cp.asarray(partial_policy)
        H = cp.zeros((k, m, m), dtype=cp.int8)
        
        # grant_counts[i, j] = total grants among entity pairs (u in Di, v in Dj)
        
        for a in range(k):
            action_matrix = partial_policy_cp[a]
            grant_counts = cp.zeros((m, m), dtype=cp.float64)
            
            # Vectorized scatter-add: process chunks of rows
            chunk_size = 10000
            for row_start in range(0, n, chunk_size):
                row_end = min(row_start + chunk_size, n)
                chunk = action_matrix[row_start:row_end]
                row_domains = colors[row_start:row_end]
                is_grant = (chunk == 1).astype(cp.float64)
                # Scatter-add (chunk_size × n) contributions into (m × m)
                idx_2d = (row_domains[:, None], colors[None, :])
                cp.add.at(grant_counts, idx_2d, is_grant)
                del chunk, is_grant, idx_2d
            
            # Deny-by-default: H=1 iff grants outnumber denies
            H[a] = (grant_counts > 0).astype(cp.int8)
            del grant_counts
            
        return H.get(), pi_dict, color_to_idx
    else:
        H = np.zeros((k, m, m), dtype=np.int8)
        domain_groups = [[] for _ in range(m)]
        for vertex, color in pi_dict.items():
            domain_groups[color_to_idx[color]].append(vertex)
        domain_vertex_arrays = [np.array(nodes, dtype=np.int32) for nodes in domain_groups]
        
        for a in range(k):
            action_matrix = partial_policy[a]
            for d1 in range(m):
                nodes1 = domain_vertex_arrays[d1]
                if nodes1.size == 0: continue
                for d2 in range(m):
                    nodes2 = domain_vertex_arrays[d2]
                    if nodes2.size == 0: continue
                    
                    block = action_matrix[np.ix_(nodes1, nodes2)]
                    num_grants = np.sum(block == 1)
                    num_denies = np.sum(block == 0)
                    
                    # Deny-by-default: grant wins only if majority
                    if (num_grants + num_denies) > 0 and num_grants > num_denies:
                        H[a, d1, d2] = 1
        
        return H, pi_dict, color_to_idx

# =============================================================================
# 4. METRICS
# =============================================================================

def calculate_metrics(H, pi_arr, G_orig):
    """
    Standardized Metrics Calculation.
    H: Summary policy (k, m, m)
    pi_arr: Mapping array (n,) where pi_arr[i] is domain index of entity i in H
    G_orig: Original policy (k, n, n)
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
# 5. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GC-based Policy Compressor (Fixed)')
    parser.add_argument('partial_policy', help='Partial policy (.npy)')
    parser.add_argument('original_policy', help='Original policy (.npy)')

    # options
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Execution mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose iteration logging')
    
    args = parser.parse_args()
    
    # Load Data
    partial_policy = np.load(args.partial_policy, mmap_mode='r')
    if partial_policy.dtype == np.uint8:
        raise ValueError("Error: Bit-packed 'noise_policy' (uint8) detected. "
                         "gc_compressor only handles standard 'partial_policy' (int8). "
                         "Please use mdl_compressor.py for bit-packed data.")
    k, n, _ = partial_policy.shape

    # Detect hardware
    device_info = args.device.upper()
    if args.device == 'gpu' and HAS_GPU:
        cc = cp.cuda.Device().compute_capability
        sm_count = cp.cuda.Device().attributes['MultiProcessorCount']
        device_info += f" ({cc[0]}.{cc[1]} capability, {sm_count} SMs)"
    

    print(f"\nGC: n={n}, k={k}, device={device_info}")
    print(f"[*] Building conflict graph...")
    start_time = time.time()
    use_gpu_build = (args.device == 'gpu') and HAS_GPU
    if use_gpu_build:
        if args.verbose:
            print("    - Strategy: GPU (Bit-Parallel CUDA, Per-Action)")
        adj_matrix = create_conflict_graph_gpu(partial_policy)
        num_edges = int(cp.sum(cp.triu(adj_matrix, k=1)))
    else:
        if args.verbose:
            print("    - Strategy: CPU (Numba)")
        pp_np = np.array(partial_policy) 
        u, v = create_conflict_graph_numba(pp_np)
        del pp_np
        num_edges = len(u)
        adj_matrix = None
    cg_time = time.time() - start_time
    print(f"    - Done in {cg_time:.2f} s. Edges: {num_edges}")
    
    print(f"[*] Coloring graph...")
    start_time = time.time()
    if args.device == 'gpu' and HAS_GPU:
        print("    - Strategy: GPU (Speculative Greedy / Gebremedhin-Manne)")
        cp.get_default_memory_pool().free_all_blocks()
        colors_gpu = color_gpu_gm(
            adj_matrix,
            verbose=args.verbose
        )
        colors_arr = cp.asnumpy(colors_gpu)
        del adj_matrix, colors_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        print("    - Strategy: CPU (Greedy / Welsh-Powell)")
        # Build CSR from edge lists
        u_sym = np.concatenate([u, v])
        v_sym = np.concatenate([v, u])
        del u, v
        
        sort_idx = np.argsort(u_sym, kind='mergesort')
        u_sorted = u_sym[sort_idx]
        v_sorted = v_sym[sort_idx]
        del u_sym, v_sym, sort_idx
        
        indptr = np.zeros(n + 1, dtype=np.int64)
        degrees = np.bincount(u_sorted, minlength=n)
        np.cumsum(degrees, out=indptr[1:])
        
        indices = v_sorted.astype(np.int32)
        del u_sorted, v_sorted, degrees
        
        vertex_degrees = indptr[1:] - indptr[:-1]
        nodes_order = np.argsort(vertex_degrees)[::-1].astype(np.int32)
        
        colors_arr = greedy_color(indptr, indices, nodes_order)
        del indptr, indices
        
    coloring_result = {i: colors_arr[i] for i in range(n)}
    num_colors = len(np.unique(colors_arr))
    del colors_arr
    color_time = time.time() - start_time
    print(f"    - Done in {color_time:.2f} s. Colors: {num_colors}")
    
    print(f"[*] Constructing summary policy H...")
    start_time = time.time()
    use_gpu_h = (args.device == 'gpu') and HAS_GPU
    if use_gpu_h:
        pp_full = np.array(partial_policy)
        H, pi_dict, color_to_idx = construct_H(pp_full, coloring_result, use_gpu_for_h=True)
        del pp_full
        cp.get_default_memory_pool().free_all_blocks()
    else:
        pp_full = np.array(partial_policy)
        H, pi_dict, color_to_idx = construct_H(pp_full, coloring_result, use_gpu_for_h=False)
        del pp_full
    h_time = time.time() - start_time
    print(f"    - Done in {h_time:.2f} s. H shape: {H.shape}")

    print(f"[*] Calculating metrics...")
    original_policy = np.load(args.original_policy, mmap_mode='r')    
    pi_arr = np.zeros(n, dtype=np.int32)
    for ent, color in pi_dict.items():
        pi_arr[ent] = color_to_idx[color]
    acc, prec, rec, f1 = calculate_metrics(H, pi_arr, original_policy)
    
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Final Domains: {num_colors}")
    print(f"Compression Time: {cg_time + color_time + h_time:.2f} s")
    print("-" * 40)

if __name__ == '__main__':
    main()
