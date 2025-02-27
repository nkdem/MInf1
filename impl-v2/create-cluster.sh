# Configuration
USER="s2203859"
SESSION_NAME="gpu_session"
WORK_DIR="/home/${USER}/minf-1"  # Working directory
DATA_DIR="/home/${USER}/minf-1/dataset/abc"  # Data directory
SCRATCH_DIR="/disk/scratch/${USER}/minf-1"  # Added minf-1 to keep it organized
# LOG_DIR="${WORK_DIR}/logs"  # Directory for logs

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# GPU_LOG="${LOG_DIR}/gpu_${TIMESTAMP}.log"
# CPU_LOG="${LOG_DIR}/cpu_${TIMESTAMP}.log"
# PYTHON_LOG="${LOG_DIR}/python_${TIMESTAMP}.log"

# Create a new tmux session if it doesn't exist
tmux new-session -d -s $SESSION_NAME 2>/dev/null || true

# Create log directory if it doesn't exist
tmux send-keys -t $SESSION_NAME "mkdir -p ${LOG_DIR}" C-m

# Change to the working directory first
tmux send-keys -t $SESSION_NAME "cd ${WORK_DIR}" C-m

# Get an interactive session with both GPU and CPU resources
# allow 80 gigs of memory
tmux send-keys -t $SESSION_NAME "srun -p PGR-Standard --gres=gpu:1 --mem 80G --pty bash" C-m

# Wait a few seconds for the shell to be ready
sleep 2

# Start monitors in background terminals
# tmux send-keys -t $SESSION_NAME "nvidia-smi -l 1 > ${GPU_LOG} 2>&1 &" C-m
# tmux send-keys -t $SESSION_NAME "top -b -d 1 > ${CPU_LOG} 2>&1 &" C-m

# Split the window into panes
# tmux split-window -v -t $SESSION_NAME
# tmux split-window -h -t $SESSION_NAME:0.1

# Configure each pane
# Main pane (top): Python script
tmux select-pane -t $SESSION_NAME:0.0
tmux send-keys -t $SESSION_NAME:0.0 "python main.py" C-m

# Bottom left pane: GPU log
# tmux select-pane -t $SESSION_NAME:0.1
# tmux send-keys -t $SESSION_NAME:0.1 "tail -f ${GPU_LOG}" C-m

# # Bottom right pane: CPU log
# tmux select-pane -t $SESSION_NAME:0.2
# tmux send-keys -t $SESSION_NAME:0.2 "tail -f ${CPU_LOG} | grep ${USER}" C-m

# Return to the main pane
# tmux select-pane -t $SESSION_NAME:0.0

echo "Session created. Attach with: tmux attach-session -t ${SESSION_NAME}"
echo "Layout:"
echo "┌─────────────────────┐"
echo "│     Python Main     │"
echo "└──────────┴──────────┘"
