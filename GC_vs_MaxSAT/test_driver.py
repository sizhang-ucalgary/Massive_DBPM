import subprocess
import logging
import json
import sys
import os



def traversal_all_files(input_dir, output_dir, solver_type, method, seconds):
    data_files = [os.path.join(dir_path, file_name) 
                  for dir_path, _, files in os.walk(input_dir) 
                    for file_name in files if file_name.endswith('.json')]
        
    # Create the destination folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, f"{solver_type}_{method}.json")
    
    for data_file in data_files:
        logging.info(f'Processing instance {data_file} with {solver_type} solver using method {method}')
        
        if solver_type == "maxsat":
            solver_cmd = ['python3.10', 'maxsat_solver.py', data_file, method, str(seconds)]
        elif solver_type == "sergcp":
            solver_cmd = ['python3.10', 'graph_coloring.py', data_file, method, str(seconds)]
        # elif solver_type == "pargcp":
        #     solver_cmd = ['mpirun', '-np', '8', './GCPSolver', '--inputFile=' + data_file, '--colorAlgorithm=' + method, '--timeout=' + str(seconds)]
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        # call the solver to find the solution of the problem
        program = subprocess.Popen(solver_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = program.communicate()
        
        if error:
            logging.error(f'Error occurred: {error}')
            continue
            
        try:
            result = json.loads(output.rstrip().replace("'", "\""))
            logging.info(f'Solution written to {raw_file}')
            raw_data = {'problem': data_file, 'result': result}
            with open(raw_file, 'a') as fp:
                json.dump(raw_data, fp)
                fp.write('\n')
        except json.JSONDecodeError as e:
            logging.error(f'Failed to parse solver output: {e}')
            logging.error(f'Raw output: {output}')



####################################################################################
# Usage:
# python3.10 test_driver.py solver_type method input_dir output_dir timeout
# 
# solver_type: maxsat, sergcp, or pargcp
# method: 
#   - For maxsat:  BE, BE_CC, BE_NF, BE_NF_LI, BE_NF_FM, BE_NF_MD, BE_NF_MD_LI
#   - For sergcp:  RS, LF, SL, RSI, LFI, SLI, CSB, CSD, SLF, GIS
#   - For pargcp:  D1, D1-2GL
####################################################################################
def main(argv):
    if len(argv) != 5:
        print("Usage: python3.10 test_driver.py solver_type method input_dir output_dir timeout")
        sys.exit(1)

    solver_type, method, input_dir, output_dir, timeout = argv
    log_filename = f'{solver_type}_{method}_{os.path.basename(input_dir)}.log'
    logging.basicConfig(level=logging.INFO, filename=log_filename, format='%(asctime)s - %(levelname)s - %(message)s')
    traversal_all_files(input_dir, output_dir, solver_type, method, timeout)



if __name__ == '__main__':
    main(sys.argv[1:])
