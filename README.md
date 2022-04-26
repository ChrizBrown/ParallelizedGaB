# ParallelizedGaB
ECE 569 Project Spring 2022 - CUDA Accelerated Implementation of Gallager B Algorithm 
Modified by Chris Brown and Jared Causey
# The input parameters are the following
# Command to use for running the  decoder with any code (matrix).
# Input matrix file name.  
# SimulationResults.txt contains some results (base line) of 
# simulation for Tanner code, IRISC dv3 or IRISC dv4 etc...

./GaB MatrixFileName ResultFileName

# How to compile:
mkdir build
cd build/
cmake ..
make

# And here is an example:
# ./EXEC MATRIX_FILE RESULT_FILE PARALLEL_FLAG TOTAL_ITERATIONS BATCH_SIZE
./GaB       ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 1 1000
./GaB_batch ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 1 1000 10