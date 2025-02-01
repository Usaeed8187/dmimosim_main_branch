#!/bin/bash

# Define variables
ns3_project_dir="$HOME/dMIMO/ns3-system-simulation/scratch/dMIMO_channel_extraction"
main_py_path="$ns3_project_dir/main.py"  # Path to main.py inside the NS-3 project
num_drops=10  # Number of iterations
initial_seed=3007  # Starting seed value

# Check if main.py exists
if [ ! -f "$main_py_path" ]; then
    echo "Error: File '$main_py_path' not found!"
    exit 1
fi

# Change to the NS-3 project directory
cd "$ns3_project_dir" || { echo "Failed to cd into $ns3_project_dir"; exit 1; }

for ((i=0; i<num_drops; i++)); do
    seed=$((initial_seed + i))  # Increment seed each iteration

    echo "Running iteration $((i+1)) with seed $seed"

    python "$main_py_path" --seed $seed --scenario V2V-Urban --small_scale_fading --num_subframes 50 --squad1_speed_km_h=10.0 --squad2_speed_km_h=10.0 --intra_sq1_rw_speed_km_h=1.0 --intra_sq2_rw_speed_km_h=1.0 --buildings_file 1narrow.txt --squad1_direction=0 --squad2_direction=0.785398 --format=both

    python "$main_py_path" --seed $seed --scenario V2V-Urban --small_scale_fading --num_subframes 50 --squad1_speed_km_h=0.1 --squad2_speed_km_h=0.1 --intra_sq1_rw_speed_km_h=0.01 --intra_sq2_rw_speed_km_h=0.01 --buildings_file 1narrow.txt --squad1_direction=0 --squad2_direction=0.785398 --format=both

    python "$main_py_path" --seed $seed --scenario V2V-Urban --small_scale_fading --num_subframes 50 --squad1_speed_km_h=3.0 --squad2_speed_km_h=3.0 --intra_sq1_rw_speed_km_h=0.3 --intra_sq2_rw_speed_km_h=0.3 --buildings_file 1narrow.txt --squad1_direction=0 --squad2_direction=0.785398 --format=both

    echo "Completed iteration $((i+1)) with seed $seed"
done