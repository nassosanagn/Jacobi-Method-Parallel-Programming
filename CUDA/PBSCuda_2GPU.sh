#!/bin/bash

#Which Queue to use
#PBS -q GPUq

#Max Wall time, Example 1 Minute #
#PBS -l walltime=00:00:20

#How many nodes and tasks per node, Example 1 node with 1 CPU and 1 GPU#
#PBS -l select=1:ncpus=1:ngpus=2

# Only this job uses the chosen nodes
#PBS -l place=excl

#JobName #
#PBS -N myGPUJob

#Change Working directory to SUBMIT directory#
cd $PBS_O_WORKDIR

# Run executable #
./Cuda_2GPU < input