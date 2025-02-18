#!/bin/bash
# Configuration
USER="s2203859"
SESSION_NAME="test"
WORK_DIR="/home/${USER}/minf-1"  # Working directory
HEAR_DS_DIR="/home/s2203859/HEAR-DS"
CHIME_DIR="/home/s2203859/CHiME3/dt05_bth"
SCRATCH_DIR="/disk/scratch/${USER}/minf-1"  # Added minf-1 to keep it organized

# (Optional) Log directory – uncomment if required
# LOG_DIR="${WORK_DIR}/logs"  # Directory for logs

# Set the number of experiments to run.
EXPERIMENT_COUNT=5

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# GPU_LOG="${LOG_DIR}/gpu_${TIMESTAMP}.log"
# CPU_LOG="${LOG_DIR}/cpu_${TIMESTAMP}.log"
# PYTHON_LOG="${LOG_DIR}/python_${TIMESTAMP}.log"

# Create a new tmux session if it doesn't exist
tmux new-session -d -s $SESSION_NAME 2>/dev/null || true

# Create log directory if it doesn't exist
# (If using LOG_DIR, uncomment the next line)
# tmux send-keys -t $SESSION_NAME "mkdir -p ${LOG_DIR}" C-m

# Change to the working directory first
tmux send-keys -t $SESSION_NAME "cd ${WORK_DIR}" C-m

# Get an interactive session with both GPU and CPU resources
# allow 80 gigs of memory
tmux send-keys -t $SESSION_NAME "srun -p PGR-Standard --gres=gpu:1 --cpus-per-task=6 --mem=80G --pty bash" C-m

# Wait a few seconds for the shell to be ready
sleep 2

# Start monitors in background terminals if needed (currently commented out)
# tmux send-keys -t $SESSION_NAME "nvidia-smi -l 1 > ${GPU_LOG} 2>&1 &" C-m
# tmux send-keys -t $SESSION_NAME "top -b -d 1 > ${CPU_LOG} 2>&1 &" C-m

# Split the window into panes if desired (currently commented out)
# tmux split-window -v -t $SESSION_NAME
# tmux split-window -h -t $SESSION_NAME:0.1

# Configure the main pane:
tmux select-pane -t $SESSION_NAME:0.0
# Ensure scratch directory exists, and copy data
tmux send-keys -t $SESSION_NAME:0.0 "mkdir -p ${SCRATCH_DIR}" C-m
tmux send-keys -t $SESSION_NAME:0.0 "cp -r ${HEAR_DS_DIR} ${SCRATCH_DIR}" C-m
tmux send-keys -t $SESSION_NAME:0.0 "cp -r ${CHIME_DIR} ${SCRATCH_DIR}" C-m

# Start the experiment by running the run_full_adam.sh script.
# Make sure run_full_adam.sh is in the current working directory or adjust the path accordingly.
tmux send-keys -t $SESSION_NAME:0.0 "bash run_full_adam.sh ${EXPERIMENT_COUNT}" C-m

echo "Session created. Attach with: tmux attach-session -t ${SESSION_NAME}"
echo "Layout:"
echo "┌─────────────────────┐"
echo "│     Python Main     │"
echo "└──────────┴──────────┘"
