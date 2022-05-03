# ParallelizedGaB
ECE 569 Project Spring 2022 - CUDA Accelerated Implementation of Gallager B Algorithm 
Modified by Chris Brown and Jared Causey

# Compiling
mkdir build
cd build/
module load cuda11/11.0
cmake3 ..
make

This will also compile the serial implementation. Details on running the serial implementation can be found below.

# Running the code
All executables follow the format of:
./executable-file <matrix file> <result file> <additional options (depends on the executable file)>

# Manual Execution
# Serial Code
./GaB ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res
# Naive Global
./GaB ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 1
# Constant Memory
./GaB_constant ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res 1
# Batching
./GaB_batch ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res <batch size>
# Streaming
./GaB_streaming ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res <stream size>
# Batching and Streaming
./GaB_batch_streaming ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res <batch size> <stream size>
# Batching Streaming, Constant Memory (Fully optimized)
./GaB_constant_batch_streaming ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res <batch size> <stream size>

# Slurm Script
A slurm script can also be run instead. First, line 34 needs to be updated to the path of your code directory.
The slurm script will default run through the following scenarios:
- Serial Code
- Naive global
- batching
- batching and streaming
- batching, streaming, and constant memory

# Functional Verification
The functional verification is done by observing the stdout output from each file. The BER for each run should decrease
exponentially as the alpha values get lower.
The outputs should be within the ballpark range of the values found in "Example Outputs" folder.