#!/bin/bash

#SBATCH -J  sleep                       # Job name
#SBATCH -o  out.sleep.%j   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p  normal                           # queue or partiton name
#SBATCH -t  01:30:00                      # Max Run time (hh:mm:ss) - 1.5 hours
#SBATCH -N 1                              # Use one node
#SBATCH -n 1                              # Run a single task

module purge
module ohpc

date

# Loop over tau values from 0.0001 to 0.0005 with a step of 0.0001
for tau in $(seq 0.0001 0.0001 0.0005); do
    echo "Running script with tau=$tau"
    python main.py --tau $tau &
done

# Wait for all background processes to finish
wait

date
