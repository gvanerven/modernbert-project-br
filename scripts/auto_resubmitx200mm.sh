#!/bin/bash

# ============================================================
# This batch script is meant to be used to complete a workload
# that requires more time than allowed by a partition's wall
# time limit.
#
# There are 2 sections in this script that can be edited by
# users
#
#     * Both sections will start with
#       "BEGIN USER-EDITABLE SECTION" and end with
#       "END USER-EDITABLE SECTION"
#     * DO NOT EDIT ELSEWHERE
#
# 1. The SBATCH options and MAX_ITERATIONS
# 2. The job step and associated logic near the bottom
# 
# NOTE: Editing elsewhere can lead to infinite job submission 
#       loops that can adversely affect other users.
# ============================================================

# ------------------------------------------------------------
# BEGIN USER-EDITABLE SECTION
# ------------------------------------------------------------
#     * SBATCH options
#     * MAX_ITERATIONS
#           - Max number of iterations (i.e., jobs) that will
#             be submitted to avoid infinite loops
# ------------------------------------------------------------

#SBATCH -J resubmit-mblg    # Job name
#SBATCH -o %x-output.%j     # Name of stdout output file (%j expands to jobId)
#SBATCH -N 6                # Total number of nodes requested
#SBATCH -t 24:00:00         # Run time (hh:mm:ss)
#SBATCH -p mi2104x          # Desired partition      

# ---------------------------------------------------------
# BEGIN USER-EDITABLE SECTION
# Begin job step and associated logic 
# ---------------------------------------------------------
#     * This is where you run an iteration (i.e., job) of
#       your overall workload.
#     * You will need to add custom logic based on your
#       application and its checkpoint system. 
#
#     E.g., something like
#     latest_checkpoint=$(ls checkpoint* | tail -n1)
#     python3 <script-name>.py --chk=latest_checkpoint
# ---------------------------------------------------------

# Activate the virtual environment
module load rocm/7.1.0
#source $WORK/bertx300/bin/activate
source $WORK/bertx200/bin/activate 
#
rocm-smi
export ACCELERATE_LOG_LEVEL=debug
export NNODES=$SLURM_JOB_NUM_NODES
export WORLD_SIZE=$(($NNODES * 4))
export MASTER_PORT=24456

ifconfig eth0 | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p' > main.node
export MASTER_ADDR=$(<main.node)

srun accelerate launch --multi_gpu --num_processes=$WORLD_SIZE --num_machines=$NNODES --dynamo_backend="no" --main_process_ip="$MASTER_ADDR" --main_process_port=$MASTER_PORT --machine_rank=$SLURM_NODEID --mixed_precision="bf16" $WORK/modernbert-project-br/scripts/train_entrypoint.py 
#accelerate launch --num_processes=1 --num_machines=1 --dynamo_backend="no" --mixed_precision="bf16" $WORK/modernbert-project-br/scripts/train_entrypoint.py 

# ---------------------------------------------------------
# END USER-EDITABLE SECTION
# ---------------------------------------------------------
