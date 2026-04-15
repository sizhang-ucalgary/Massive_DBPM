import os
import json
import csv
import sys

def analyze_raw_data(input_dir, output_dir, timeout, timelimit):
    """
    Traverses the input directory, merges results by method, and creates
    CSV files for cactus plot visualization in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # method -> [list of sub-threshold runtimes]
    method_runtimes = {}
    
    print(f"[*] Scanning {input_dir} for results...")
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.json'):
                method = filename.replace('.json', '')
                if method not in method_runtimes:
                    method_runtimes[method] = []
                
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            runtime = float(data['result']['runtime'])
                            if runtime < timeout:
                                method_runtimes[method].append(runtime)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue

    for method, runtimes in method_runtimes.items():
        runtimes.sort()
        csv_path = os.path.join(output_dir, f"{method}.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            
            cumulative_time = 0.0
            solved_count = 0
            for rt in runtimes:
                cumulative_time += rt
                if cumulative_time > timelimit:
                    break
                solved_count += 1
                writer.writerow([solved_count, f"{cumulative_time:.2f}"])
        
        print(f"    - Created {os.path.basename(csv_path)} ({solved_count} points)")

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 rawdata_analyzer.py <input_dir> <output_dir> <timeout> <timelimit>")
        print("Example: python3 rawdata_analyzer.py raw_data output 300 86400")
        sys.exit(1)
        
    input_root = sys.argv[1]
    output_target = sys.argv[2]
    try:
        timeout = float(sys.argv[3])
        timelimit = float(sys.argv[4])
    except ValueError:
        print("Error: timeout and timelimit must be numeric.")
        sys.exit(1)
        
    analyze_raw_data(input_root, output_target, timeout, timelimit)
    print(f"\n[*] All CSV files generated in '{output_target}/' folder.")
    
    # Automatically update PDF if template is present
    if os.path.exists('cactus.tex'):
        print("[*] Found cactus.tex. Updating PDF...")
        # Since cactus.tex hardcodes "table {output/xxx.csv}", ensure the output dir name matches
        os.system('pdflatex cactus.tex')
        for ext in ['.aux', '.log']:
            if os.path.exists(f'cactus{ext}'): os.remove(f'cactus{ext}')
        print("[*] Cactus plot PDF updated.")

if __name__ == "__main__":
    main()
