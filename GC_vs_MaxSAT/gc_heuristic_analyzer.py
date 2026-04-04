import json
import sys
import os
import numpy as np

def merge_json_files(input_dir):
    results = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            heuristic = filename.split("_")[1].split(".")[0]

            with open(filepath, "r") as f:
                for line in f:
                    data = json.loads(line)
                    problem = data["problem"]

                    if problem not in results:
                        results[problem] = {} # Dictionary for each problem
                    if heuristic not in results[problem]:
                        results[problem][heuristic] = ([], [])  # Initialize tuple of lists for each heuristic

                    solution = int(data["result"]["solution"])
                    runtime = float(data["result"]["runtime"])

                    results[problem][heuristic][0].append(solution)
                    results[problem][heuristic][1].append(runtime)

    return results



def calculate_stats(solutions):  # Same as before

    trials = np.array(solutions)
    lower_bound = np.min(trials)
    upper_bound = np.max(trials)
    median = np.median(trials)
    mean = np.mean(trials)

    return lower_bound, upper_bound, median, mean



####################################################################################
# Usage:
# python3.10 gc_heuristic_analyzer.py input_dir
# 
# input_dir: Directory containing JSON output files from graph coloring solvers
####################################################################################
def main(argv):
    if len(argv) != 1:
        print("Usage: python3.10 gc_heuristic_analyzer.py rawdata_ps")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    data = merge_json_files(input_dir)

    heuristic_order = ["RS", "RSI", 
                       "LF", "LFI", 
                       "SL", "SLI", 
                       "CSB", "CSD", 
                       "SLF", "GIS"]

    for problem, heuristics in data.items():
        print(f"Problem: {problem}")

        # Print heuristics in the specified order if they exist for the current problem
        for heuristic in heuristic_order:
            if heuristic in heuristics:
                solutions, runtimes = heuristics[heuristic]  # Retrieve data for the heuristic

                sol_lower, sol_upper, sol_median, _ = calculate_stats(solutions)
                _, _, _, rt_mean = calculate_stats(runtimes)
                
                ratio = sol_median/20.0
                time = rt_mean*1000
                #print(f"{heuristic}, L, U, A, T")
                print(f"  {sol_lower} & {sol_upper} & {ratio:.2f} & {time:.2f} &")
        print("-" * 20)



if __name__ == "__main__":
    main(sys.argv[1:])
