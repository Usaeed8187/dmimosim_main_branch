#!/bin/bash

# Define variables
ns3_project_dir="$HOME/dMIMO/ns3-system-simulation/scratch/dMIMO_channel_extraction"
main_py_path="$ns3_project_dir/main.py"  # Path to main.py inside the NS-3 project
ns3_channel_dir_start="V2V-Urban_1narrow_NoExternalNodes_"
ns3_channel_dir_high_mobility="_14_50_70_0_421_10_100.0_10.0_0.00_10_100.0_10.0_0.79_1000.0_1.0_1.0_2.0_1.5_0_0/"
ns3_channel_dir_end_medium_mobility="_14_50_70_0_421_10_100.0_3.0_0.00_10_100.0_3.0_0.79_1000.0_0.3_0.3_2.0_1.5_0_0/"
ns3_channel_dir_end_low_mobility="_14_50_70_0_421_10_100.0_3.0_0.00_10_100.0_3.0_0.79_1000.0_0.3_0.3_2.0_1.5_0_0/"
dMIMO_project_dir="$HOME/dMIMO/dmimosim_main/"
convert_ns3_channels_path="$dMIMO_project_dir/ns3/convert_ns3_channels.py"
num_drops=10  # Number of iterations
num_subframes=50
initial_seed=3007  # Starting seed value

channel_saving_dir="/home/data/ns3_channels_2way/"

# Check if main.py exists
if [ ! -f "$main_py_path" ]; then
    echo "Error: File '$main_py_path' not found!"
    exit 1
fi

# Change to the NS-3 project directory
cd "$ns3_project_dir" || { echo "Failed to cd into $ns3_project_dir"; exit 1; }

curr_drop_idx=1

for ((i=0; i<num_drops; i++)); do
    seed=$((initial_seed + i))  # Increment seed each iteration

    echo "Running iteration $((i+1)) with seed $seed"

    echo "Full path: $ns3_project_dir/dmimo_ch/$ns3_channel_dir_start$seed$ns3_channel_dir_high_mobility"

    # High mobility channels
    # python "$main_py_path" --seed $seed --scenario V2V-Urban --small_scale_fading --num_subframes $num_subframes --squad1_speed_km_h=10.0 --squad2_speed_km_h=10.0 --intra_sq1_rw_speed_km_h=1.0 --intra_sq2_rw_speed_km_h=1.0 --buildings_file 1narrow.txt --squad1_direction=0 --squad2_direction=0.785398 --format=both
    python "$convert_ns3_channels_path" "$ns3_project_dir/dmimo_ch/$ns3_channel_dir_start$seed$ns3_channel_dir_high_mobility" "$channel_saving_dir/channels_high_mobility_$curr_drop_idx/"    

    # Medium mobility channels
    # python "$main_py_path" --seed $seed --scenario V2V-Urban --small_scale_fading --num_subframes $num_subframes --squad1_speed_km_h=3.0 --squad2_speed_km_h=3.0 --intra_sq1_rw_speed_km_h=0.3 --intra_sq2_rw_speed_km_h=0.3 --buildings_file 1narrow.txt --squad1_direction=0 --squad2_direction=0.785398 --format=both
    python "$convert_ns3_channels_path" "$ns3_project_dir/dmimo_ch/$ns3_channel_dir_start$seed$ns3_channel_dir_high_mobility" "$channel_saving_dir/channels_high_mobility_$curr_drop_idx/"    

    # Low mobility channels
    # python "$main_py_path" --seed $seed --scenario V2V-Urban --small_scale_fading --num_subframes $num_subframes --squad1_speed_km_h=0.1 --squad2_speed_km_h=0.1 --intra_sq1_rw_speed_km_h=0.01 --intra_sq2_rw_speed_km_h=0.01 --buildings_file 1narrow.txt --squad1_direction=0 --squad2_direction=0.785398 --format=both
    python "$convert_ns3_channels_path" "$ns3_project_dir/dmimo_ch/$ns3_channel_dir_start$seed$ns3_channel_dir_high_mobility" "$channel_saving_dir/channels_high_mobility_$curr_drop_idx/"    

    curr_drop_idx=$((curr_drop_idx+1))

    echo "Completed iteration $((i+1)) with seed $seed"
done