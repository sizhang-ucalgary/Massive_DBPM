from disjoint_set import DisjointSet
from itertools import combinations
from datetime import datetime
import numpy as np
import logging
import json
import sys
import os



def indistinguishable(G, u, v):
    _, rows, cols = G.shape
    if rows != cols:
        raise ValueError('indistinguishable : G is not a k x n x n matrix')

    return np.all(
        (G[:, u, u] == G[:, u, v]) & (G[:, v, u] == G[:, v, v]) & 
        (G[:, u, :] == G[:, v, :]) & (G[:, :, u] == G[:, :, v])
    )



def summarize(G):
    slices, rows, cols = G.shape
    if rows != cols:
        raise ValueError('summarize : G is not k x n x n matrix')

    ds = DisjointSet()
    for i in range(rows):
        ds.find(i)
    for (u, v) in combinations(range(rows), 2):
        if ds.find(u) != ds.find(v) and indistinguishable(G, u, v):
            ds.union(u, v)
    vertex_sets = list(ds.itersets())
    new_size = len(vertex_sets)
    
    # If no merging needed, return original matrix
    if new_size == rows:
        return G, rows
    else:
        new_G = np.full((slices, new_size, new_size), ' ', dtype=str)
        old_to_new = {vertex: new_idx 
                      for new_idx, vertex_set in enumerate(vertex_sets) 
                        for vertex in vertex_set}
        # Fill new matrix using representatives
        for k in range(slices):
            for i in range(rows):
                for j in range(cols):
                    if i in old_to_new and j in old_to_new:
                        new_i = old_to_new[i]
                        new_j = old_to_new[j]
                        new_G[k, new_i, new_j] = G[k, i, j]
        return new_G, new_size



def partial_matrix_from_summary_digraph(H, n, ps):    
    k, m, _ = H.shape
    if n < m:
        raise ValueError('partial_matrix_from_summary : n is less than m')

    G = np.full((k,n,n), ' ', dtype=str)
    
    '''
    # equally groups n elements into m classes
    partitions = [list(range(n)[i::m]) for i in range(m)]
    '''
    
    # randomly groups n elements into m classes
    group_sizes = np.random.multinomial(n, np.ones(m) / m)
    # generate elements and shuffle them
    elements = np.arange(n)
    np.random.shuffle(elements)
    # create partitions
    partitions = []
    start = 0
    for size in group_sizes:
        partitions.append(elements[start:start+size].tolist())
        start += size
    
    # update G based on the adjacency and non-adjacency of H
    for a in range(k):
        for u in range(m):
            for v in range(m):
                for i in partitions[u]:
                    for j in partitions[v]:
                        G[a, i, j] = H[a, u, v]
    # randomly change cells in G to wildcards
    total_cells = G.size
    num_wildcards = int(total_cells * ps)
    indices_to_change = np.random.choice(total_cells, num_wildcards, replace=False)
    G.flat[indices_to_change] = '*'
    return G



####################################################################################
# Usage:
# python3 dataset_generator.py config_file
# 
# config_file: Path to the JSON configuration file (e.g., config_dataset.json)
####################################################################################
def main(argv):
    if len(argv) != 1:
        print('Usage: python3.10 dataset_generator.py config_dataset.json')
        sys.exit(1)

    config_file = sys.argv[1]
    
    with open(config_file, 'r') as fp:
        config = json.load(fp)
        k, m, n, ps, cnt, path, prefix = (config['K#'], config['M#'], config['N#'], config['PS%'], config['Instances#'], config['Path'], config['Prefix'])

        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime('_%Y%m%d_%H%M%S%f_')
        pid = os.getpid()
        file_name = os.path.join(path, f'{prefix}{timestamp}{pid}')

        logging.basicConfig(level=logging.INFO, filename='dataset_generator.out', format='%(asctime)s - %(levelname)s - %(message)s')

        for i in range(1, cnt + 1):
            H = np.random.choice(['0', '1'], size=(k, m, m), p=[0.5, 0.5])
            irreducible_H, actual_m = summarize(H)
            G = partial_matrix_from_summary_digraph(irreducible_H, n, ps)

            data = {
                'num_rights': k,
                'vertex_size_in_H': actual_m,
                'vertex_size_in_G': n,
                'wildcard_ratio_in_G': ps,
                'summary_digraph': irreducible_H.tolist(),
                'partial_policy': G.tolist()
            }
            instance_name = f"{file_name}{i}.json"
            logging.info(f'Created an instance {instance_name}')
            with open(instance_name, 'w') as fp:
                json.dump(data, fp)

        logging.info(f'In total, created {cnt} instances in the "{path}" folder.\n')



if __name__ == '__main__':
    main(sys.argv[1:])
