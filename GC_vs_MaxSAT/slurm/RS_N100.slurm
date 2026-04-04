#!/bin/bash
####### Reserve computing resources #############
#SBATCH --job-name=RS_N100
#SBATCH --output=RS_N100.log
#
#SBATCH --partition=bigmem
#SBATCH --exclude=mmg1,mmg2
#SBATCH --time=1-00:00:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-user=si.zhang2@ucalgary.ca
#SBATCH --mail-type=end,fail

####### Set environment variables ###############
module load python/3.10.4 

####### Run your script #########################
printf "\n The Runtime of dataset N100\n";
time python3.10 test_driver.py sergcp RS input/M2N100 rawdata/M2N100 300;
time python3.10 test_driver.py sergcp RS input/M4N100 rawdata/M4N100 300;
time python3.10 test_driver.py sergcp RS input/M6N100 rawdata/M6N100 300;
time python3.10 test_driver.py sergcp RS input/M8N100 rawdata/M8N100 300;
time python3.10 test_driver.py sergcp RS input/M10N100 rawdata/M10N100 300;
