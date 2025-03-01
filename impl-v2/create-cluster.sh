echo "Enter the name for the session:"
read SESSION_NAME
# Configuration
USER="s2203859"
WORK_DIR="/home/${USER}/minf-1"
HEAR_DS_DIR="/home/s2203859/HEAR-DS"
CHIME_DIR="/home/s2203859/CHiME3/dt05_bth"
SCRATCH_DIR="/disk/scratch/${USER}/minf-1"  

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create a new tmux session if it doesn't exist
tmux new-session -d -s $SESSION_NAME 2>/dev/null || true

# Change to the working directory first
tmux send-keys -t $SESSION_NAME "cd ${WORK_DIR}" C-m

# Get an interactive session with both GPU and CPU resources
# allow 80 gigs of memory
tmux send-keys -t $SESSION_NAME "srun -p PGR-Standard --gres=gpu:1 --mem 80G --pty bash" C-m

# Wait a few seconds for the shell to be ready
sleep 2

tmux select-pane -t $SESSION_NAME:0.0
# copy over the data
tmux send-keys -t $SESSION_NAME:0.0 "mkdir -p ${SCRATCH_DIR}" C-m
tmux send-keys -t $SESSION_NAME:0.0 "cp -r ${HEAR_DS_DIR} ${SCRATCH_DIR}" C-m
tmux send-keys -t $SESSION_NAME:0.0 "cp -r ${CHIME_DIR} ${SCRATCH_DIR}" C-m

echo "Session created. Attach with: tmux attach-session -t ${SESSION_NAME}"
echo "Layout:"
echo "┌─────────────────────┐"
echo "│     Python Main     │"
echo "└──────────┴──────────┘"
