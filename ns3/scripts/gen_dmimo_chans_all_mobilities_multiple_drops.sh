#!/bin/bash

curr_drop_idx=1
num_drops=20
num_subframes=100
initial_seed=3007
txsquad_ues=10
rxsquad_ues=10

squad1_direction=0.0
squad2_direction=0.0

squad1_speed_km_h=40.0
squad2_speed_km_h=40.0

# float math (10% of squad speed)
intra_sq1_rw_speed_km_h=$(awk "BEGIN{print ${squad1_speed_km_h}/10}")
intra_sq2_rw_speed_km_h=$(awk "BEGIN{print ${squad2_speed_km_h}/10}")

# Format helpers to mirror main.py’s run_postfix (dirs use 2 decimals for directions)
fmt_s1_dir=$(printf "%.2f" "$squad1_direction")
fmt_s2_dir=$(printf "%.2f" "$squad2_direction")
fmt_s1_speed=$(printf "%.1f" "$squad1_speed_km_h")
fmt_s2_speed=$(printf "%.1f" "$squad2_speed_km_h")
fmt_intra1=$(printf "%.1f" "$intra_sq1_rw_speed_km_h")
fmt_intra2=$(printf "%.1f" "$intra_sq2_rw_speed_km_h")

# Project paths
ns3_project_dir="$HOME/dMIMO/ns3-system-simulation/scratch/dMIMO_channel_extraction"
main_py_path="$ns3_project_dir/main.py"
ns3_channel_dir_start="V2V-Urban_1narrow_NoExternalNodes_"
dMIMO_project_dir="$HOME/dMIMO/dmimosim_main/"
convert_ns3_channels_path="$dMIMO_project_dir/ns3/convert_ns3_channels.py"
channel_saving_dir="$HOME/dMIMO/minor_revision_channels/"

# Build the directory suffix dynamically (must match main.py’s run_postfix order/format)
# Layout matches:
# _14_{num_subframes}_70_0_421_{txsquad_ues}_100.0_{s1_speed}_{s1_dir}_{rxsquad_ues}_100.0_{s2_speed}_{s2_dir}_1000.0_{intra1}_{intra2}_2.0_1.5_0_0/
ns3_dir_suffix=$(printf "_14_%d_70_0_421_%d_100.0_%s_%s_%d_100.0_%s_%s_1000.0_%s_%s_2.0_1.5_0_0/" \
  "$num_subframes" "$txsquad_ues" "$fmt_s1_speed" "$fmt_s1_dir" "$rxsquad_ues" "$fmt_s2_speed" "$fmt_s2_dir" "$fmt_intra1" "$fmt_intra2")

# Sanity check main.py presence
if [ ! -f "$main_py_path" ]; then
    echo "Error: File '$main_py_path' not found!"
    exit 1
fi

cd "$ns3_project_dir" || { echo "Failed to cd into $ns3_project_dir"; exit 1; }

for ((i=0; i<num_drops; i++)); do
    seed=$((initial_seed + i))
    echo "Running iteration $((i+1)) with seed $seed"

    # High mobility run (your custom speeds/directions)
    python "$main_py_path" \
        --seed "$seed" \
        --scenario V2V-Urban \
        --small_scale_fading \
        --num_subframes "$num_subframes" \
        --squad1_speed_km_h="$squad1_speed_km_h" \
        --squad2_speed_km_h="$squad2_speed_km_h" \
        --intra_sq1_rw_speed_km_h="$intra_sq1_rw_speed_km_h" \
        --intra_sq2_rw_speed_km_h="$intra_sq2_rw_speed_km_h" \
        --buildings_file 1narrow.txt \
        --squad1_direction="$squad1_direction" \
        --squad2_direction="$squad2_direction" \
        --format=both \
        --nUEs_sq1="$txsquad_ues" \
        --nUEs_sq2="$rxsquad_ues"

    raw_ns3_path="$ns3_project_dir/dmimo_ch/${ns3_channel_dir_start}${seed}${ns3_dir_suffix}"
    python "$convert_ns3_channels_path" "$raw_ns3_path" "$channel_saving_dir/channels_high_mobility_$curr_drop_idx/"
    echo "$raw_ns3_path"
    rm -rf -- "$raw_ns3_path"

    curr_drop_idx=$((curr_drop_idx+1))
    echo "Completed iteration $((i+1)) with seed $seed"
done
