########################################################################################################################
# ParallelizedGaB
########################################################################################################################
Group 3 ECE 569 Project Spring 2022 - CUDA Accelerated Implementation of Gallager B Algorithm
Modified by Chris Brown and Jared Causey


########################################################################################################################
# Compiling
########################################################################################################################
cd <path-to-code-directory>
mkdir build
cd build/
module load cuda11/11.0
cmake3 ..
make


########################################################################################################################
# Run the Serial Code
########################################################################################################################
From within the build directory run:
./GaB ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res


########################################################################################################################
# Run the optimized GPU Code
########################################################################################################################
To run manually from within the build directory:
./GaB_constant_batch_streaming ../datasets/IRISC_dv4_R050_L54_N1296_Dform IRISC_dv4_R050_L54_N1296_Dform_Res <batch size> <stream size>

Optimal Configuration:
batch size = 16
stream size = 4

To run with the slurm script:
1. Change line 34 in 'run_gab.slurm' to point to your path to the source code
2. Type:
   sbatch run_gab.slurm

The slurm script will run the optimized GPU code and the serial code. It also contains the ability to run some of the
other executables. The details on these can be found at the bottom of this README in the 'Running All Executables' section.


########################################################################################################################
# Functional Verification
########################################################################################################################
The functional verification is done by observing the stdout output from each file. The BER for each run should decrease
exponentially as the alpha values get lower.
The outputs should be within the ballpark range of the values found in "Example Outputs" folder.


########################################################################################################################
# Running All Executables
########################################################################################################################
Below shows how to manually run all of the different executables the get generated. Each executable served as a way to
test individual optimization strategies before combining them into the combined optimized implementation.

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
