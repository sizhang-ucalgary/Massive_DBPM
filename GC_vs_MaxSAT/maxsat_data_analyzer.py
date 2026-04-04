import json
import csv
import sys
import os



def merge_json_files(input_dir):
    # Create the destination folder if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Dictionary to store merged contents by file name
    merged_contents = {}
    
    for dir_path, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                raw_file = os.path.join(dir_path, file_name)
                # open an file
                with open(raw_file, 'r') as fp:
                    content = fp.read()
                    # Check if the file name is already present in the dictionary
                    if file_name in merged_contents:
                        # Append the content to the existing entry
                        merged_contents[file_name] += content
                    else:
                        # Create a new entry for the file name
                        merged_contents[file_name] = content

    # Write the merged contents to individual files
    for file_name, content in merged_contents.items():
        # Path to the merged file
        merged_file_path = os.path.join('output', file_name)

        # Write the merged content to the file
        with open(merged_file_path, 'w') as fp:
            fp.write(content)



def analyze_data(timeout, timelimit):
    merged_dict = {}
    for dir_path, _, files in os.walk('output'):
        for file_name in files:
            if file_name.endswith('.json'):
                raw_file = os.path.join(dir_path, file_name)
                runtime_dict = {}
                # open an file
                with open(raw_file, 'r') as fp:                 
                    lines = fp.read().strip().split('\n')
                    for line in lines:
                        # Parse the JSON object
                        json_obj = json.loads(line)
                        runtime_dict[json_obj['problem']] = float(json_obj['result']['runtime'])
                    for key, value in runtime_dict.items():
                        merged_dict.setdefault(key, []).append(value)    
    
    i = 0
    for dir_path, _, files in os.walk('output'):
        for file_name in files:
            if file_name.endswith('.json'):
                csv_file = os.path.join('output', file_name.replace('.json', '.csv'))
#                os.system('rm ' + os.path.join(dir_path, file_name))
                with open(csv_file, 'w', newline='') as fp:
                    writer = csv.writer(fp)
                    # Write the header row
                    writer.writerow(['x', 'y'])
                    
                    # Sort the dictionary based on the i-th value in the list
                    sorted_dict = dict(sorted(merged_dict.items(), key=lambda x: x[1][i]))
                    sum_time = 0.0
                    m = 0                                        
                    for key, value in sorted_dict.items():
                        # Find ti such that ti < timeout
                        if value[i] < timeout:
                            sum_time += value[i]
                            # Find the maximum value of 'm' such that t1 + t2 + ... + tm <= timelimit
                            if sum_time <= timelimit:
                                m += 1
                                writer.writerow([str(m), f'{sum_time:.2f}'])
                    i += 1



#########################################################
# python3.10 maxsat_data_analyzer.py rawdata 300 86400 
#########################################################
def main(argv):
    if len(argv) != 3:
        print("Usage: python3.10 maxsat_data_analyzer.py rawdata 300 86400")
        sys.exit(1)

    input_dir = sys.argv[1]         # the directory to load inputs
    timeout = float(sys.argv[2])    # a timeout in seconds
    timelimit = float(sys.argv[3])  # a timelimit in seconds
    merge_json_files(input_dir)
    analyze_data(timeout, timelimit)    
    print('Created csv files in the "output" folder.')
    os.system('pdflatex cactus.tex')
    os.system('rm cactus.aux cactus.log')
    print('Drawn a cactus plot.')



if __name__ == '__main__':
    main(sys.argv[1:])
