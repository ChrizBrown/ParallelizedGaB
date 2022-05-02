# ParallelizedGaB
ECE 569 Project Spring 2022 - CUDA Accelerated Implementation of Gallager B Algorithm 
Modified by Chris Brown and Jared Causey

# Compiling
mkdir build
cd build/
module load cuda11/11.0
cmake ..
make

# Running the code
All executables follow the format of:
./executable-file <matrix file> <result file> <additional options (depends on the executable file)>
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
