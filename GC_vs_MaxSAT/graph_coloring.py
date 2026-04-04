import networkx as nx
import numpy as np
import itertools
import signal
import time
import json
import sys



def handle_timeout(sig, frame):
    raise TimeoutError('Coloring Program timed out.')



def distinguish_characters(str1, str2):
    return sum(c1 != '*' and c2 != '*' and c1 != c2 for c1, c2 in zip(str1, str2))



####################################################################################
# Usage:
# python3 graph_coloring.py data_file method timeout
# 
# data_file: Path to the JSON instance file
# method: RS, LF, SL, RSI, LFI, SLI, CSB, CSD, SLF, GIS
# timeout: Relaxation time in seconds (e.g., 300)
####################################################################################
def main(argv):
    if len(argv) != 3:
        print("Usage: python3.10 graph_coloring.py data_file method 300")
        sys.exit(1)

    data_file, method, seconds_str = argv
    seconds = int(seconds_str)

    with open(data_file, 'r') as fp:
        instance = json.load(fp)
        M = np.array(instance['partial_policy'])        
        k, n, _ = M.shape

        G = nx.Graph()
        for a in range(k):
            for (u, v) in itertools.combinations([i for i in range(n)], 2):
                row_diff = distinguish_characters(M[a, u], M[a, v])
                col_diff = distinguish_characters(M[a, :, u], M[a, :, v])
                if (row_diff + col_diff) > 0 :
                    G.add_edge(u, v)

        # RS, LF, SL are easy to implement but may not produce the best results.
        # CSB, CSD utilize graph structure but can have unpredictable performance.
        # SLF, GIS often provide better results buy can be computationally more expensive. 
        build_in = {
            'RS': ('random_sequential', False), # Vertices are colored in a random order.
            'LF': ('largest_first', False), # Vertices are colored in descending order of their degree.
            'SL': ('smallest_last', False), # Vertices are colored in ascending order of their degree.
            'RSI': ('random_sequential', True), # RS with interchange
            'LFI': ('largest_first', True), # LF with interchange
            'SLI': ('smallest_last', True), # SL with interchange
            'CSB': ('connected_sequential_bfs', False),
            'CSD': ('connected_sequential_dfs', False), # alias for connected_sequential
            'SLF': ('saturation_largest_first', False), # alias for DSATUR
            'GIS': ('independent_set', False) # Find a maximum independent set, Color all vertices in the independent set with the same color, Repeat above steps until all vertices are colored.
        }

        # Register the signal function handler
        signal.signal(signal.SIGALRM, handle_timeout)
        # Define a timeout for your function
        signal.alarm(seconds)

        result = {'solution': 'UNSATISFIABLE', 'runtime': f'{seconds:.2f}'}
        try:
            start_time = time.time()
            x, y = build_in.get(method, ('DSATUR', False))
            sol = nx.greedy_color(G, strategy=x, interchange=y)
            end_time = time.time()
            run_time = end_time - start_time
            '''
            # Group nodes by color
            nodes_by_color = {}
            for node, color in sol.items():
                if color not in nodes_by_color:
                    nodes_by_color[color] = []
                nodes_by_color[color].append(node)

            # Print nodes grouped by color
            for color, nodes in nodes_by_color.items():
                print(f"Color {color}: {sorted(nodes)}")
            '''
            chromatic_num = 1
            if sol and sol.values():
                chromatic_num += max(sol.values())
            
            result = {'solution': str(chromatic_num), 'runtime': f'{run_time:.4f}'}
        except TimeoutError:
            result = {'solution': 'TIMEOUT', 'runtime': f'{seconds:.2f}'}
        finally:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            print(result)



if __name__ == '__main__':
    main(sys.argv[1:])
