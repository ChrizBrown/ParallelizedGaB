#!/bin/bash

### Set the job name
#SBATCH --job-name=e569_project_GaB

### Specify the group for this job
### List of PI groups available to each user can be found with "va" command

####ece569 students can also replace ece569 with another group id if sponsored by another faculty. 
#SBATCH --account=ece569

### Set the queue for this job as windfall or standard (adjust ### and #)
#SBATCH --partition=standard

#SBATCH -e run.error
#SBATCH -o run.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 01:30:00

### Load required modules/libraries if needed (openmpi example)
### Use "module avail" command to list all available modules
###module load openmpi3
###module load cuda11/11.0

### set directory for job execution, replace the following path with your own path
hw_path=/home/u31/brownca/ece569/ParallelizedGaB/build
cd $hw_path
mkdir -p results
###
STREAMS=($(seq 1 1 4))
BATCHES=($(seq 1 1 16))

#SERIAL
/usr/bin/time ./GaB ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res  > results/gab_serial_results.txt

#GLOBAL
#/usr/bin/time ./GaB ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 1 > results/gab_global_results.txt

#BATCHED (BATCH=16)
#/usr/bin/time ./GaB_batch ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 16 > results/gab_batched_results.txt

#BATCHED STREAM (BATCH=16 STREAM=4)
#/usr/bin/time ./GaB_batch_streaming ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 16 4 > results/gab_batched_streamed_results.txt

#CONSTANT MEMORY BATCHED STREAM (BATCH=16 STREAM=4)
/usr/bin/time ./GaB_constant_batch_streaming ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 16 4 > results/gab_constant_batched_streamed_results.txt


#SWEEPING EXPERIMENT
#for i in ${STREAMS[@]}
#do
#   for j in ${BATCHES[@]}}
#   do	   
#	file=slurm_gab_output_"$i"_.txt
#       /usr/bin/time ./GaB_batch_streaming ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res $j $i > results/gab_constant_batched_streamed_"$i"_"$j"_results.txt
#   done	
#done


#SWEEPING EXPERIMENT (OLD)
#/usr/bin/time ./GaB_sweeping_experiment ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 16 4 > results/gab_sweeping_results.txt
