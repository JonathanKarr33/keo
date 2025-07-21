#!/bin/bash
# filepath: /home/kuangshiai/Desktop/AAAI2026/Code/keo/keo/sensemaking_QA/run_exp_0717.sh

# Status file to track current experiment
STATUS_FILE="/tmp/current_experiment_status.txt"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to update status file
update_status() {
    local status=$1
    echo "$status" > "$STATUS_FILE"
    log_message "Status updated: $status"
}

# Function to run experiment with environment variable
run_experiment() {
    local script_name=$1
    local exp_name=$2
    
    if [ ! -f "$script_name" ]; then
        log_message "ERROR: Script $script_name not found!"
        update_status "ERROR: Script $script_name not found!"
        return 1
    fi
    
    log_message "Starting experiment: $exp_name"
    log_message "Setting environment variable CURRENT_EXPERIMENT=$exp_name"
    
    # Set environment variable and update status file
    export CURRENT_EXPERIMENT="$exp_name"
    update_status "RUNNING: $exp_name (Started: $(date '+%Y-%m-%d %H:%M:%S'))"
    
    # Make script executable
    chmod +x "$script_name"
    
    # Run the script
    log_message "Executing: $script_name"
    bash "$script_name"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "SUCCESS: $exp_name completed successfully"
        update_status "COMPLETED: $exp_name (Finished: $(date '+%Y-%m-%d %H:%M:%S'))"
    else
        log_message "ERROR: $exp_name failed with exit code $exit_code"
        update_status "FAILED: $exp_name (Failed: $(date '+%Y-%m-%d %H:%M:%S')) - Exit code: $exit_code"
        return $exit_code
    fi
    
    log_message "Finished experiment: $exp_name"
    echo "----------------------------------------"
}

# Initialize status file
update_status "STARTING: Sequential experiments ($(date '+%Y-%m-%d %H:%M:%S'))"

# Main execution
log_message "Starting sequential execution of experiments"
log_message "You can check the current running experiment with: cat $STATUS_FILE"

# Run experiments sequentially
run_experiment "exp_0717.sh" "EXP_0717"
if [ $? -ne 0 ]; then
    log_message "ABORT: exp_0717.sh failed, stopping execution"
    update_status "ABORTED: After EXP_0717 failure ($(date '+%Y-%m-%d %H:%M:%S'))"
    exit 1
fi

run_experiment "exp_0717_2.sh" "EXP_0717_2"
if [ $? -ne 0 ]; then
    log_message "ABORT: exp_0717_2.sh failed, stopping execution"
    update_status "ABORTED: After EXP_0717_2 failure ($(date '+%Y-%m-%d %H:%M:%S'))"
    exit 1
fi

run_experiment "exp_0717_3.sh" "EXP_0717_3"
if [ $? -ne 0 ]; then
    log_message "ABORT: exp_0717_3.sh failed, stopping execution"
    update_status "ABORTED: After EXP_0717_3 failure ($(date '+%Y-%m-%d %H:%M:%S'))"
    exit 1
fi

# Clear environment variable after completion
unset CURRENT_EXPERIMENT
update_status "ALL COMPLETED: Sequential experiments finished ($(date '+%Y-%m-%d %H:%M:%S'))"

log_message "All experiments completed successfully!"
log_message "Sequential execution finished"

# Clean up status file after a delay (optional)
sleep 2
rm -f "$STATUS_FILE"