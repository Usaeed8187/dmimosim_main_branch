#!/bin/bash
set -euo pipefail

# ======== Config ========
curr_drop_idx=40
num_drops=61
num_subframes=100
initial_seed=3047
txsquad_ues=10
rxsquad_ues=10

squad1_direction=0.0
squad2_direction=0.0

squad1_speed_km_h=10.0
squad2_speed_km_h=10.0

# Radii (meters) and inter-squad distance (meters)
squad1_radius=100.0
squad2_radius=100.0
inter_squad_distance=1000.0

# Float math (10% of squad speed)
intra_sq1_rw_speed_km_h=$(awk "BEGIN{print ${squad1_speed_km_h}/10}")
intra_sq2_rw_speed_km_h=$(awk "BEGIN{print ${squad2_speed_km_h}/10}")

# ======== Paths ========
ns3_project_dir="$HOME/dMIMO/ns3-system-simulation/scratch/dMIMO_channel_extraction"
main_py_path="$ns3_project_dir/main.py"
ns3_channel_dir_start="V2V-Urban_1narrow_NoExternalNodes_"
dMIMO_project_dir="$HOME/dMIMO/dmimosim_main"
convert_ns3_channels_path="$dMIMO_project_dir/ns3/convert_ns3_channels.py"
# channel_saving_dir="$HOME/dMIMO/chan_pred_channels"
channel_saving_dir="/run/media/wireless/Extreme SSD/chan_pred_channels"

# Find GNU parallel (user-local install ok)
PARALLEL_BIN="${PARALLEL_BIN:-$(command -v parallel || true)}"
if [[ -z "${PARALLEL_BIN}" ]]; then
  PARALLEL_BIN="$HOME/.local/bin/parallel"
fi
if [[ ! -x "$PARALLEL_BIN" ]]; then
  echo "Error: GNU parallel not found. Put it in PATH or set PARALLEL_BIN." >&2
  exit 1
fi

# Concurrency: default to all cores (override with JOBS=N)
JOBS="${JOBS:-$(nproc)}"

# ======== Sanity checks ========
[[ -f "$main_py_path" ]] || { echo "Error: $main_py_path not found"; exit 1; }
[[ -f "$convert_ns3_channels_path" ]] || { echo "Error: $convert_ns3_channels_path not found"; exit 1; }
cd "$ns3_project_dir" || { echo "Failed to cd into $ns3_project_dir"; exit 1; }

# ======== Build suffix to mirror main.py's run_postfix ========
# match formatting used in main.py’s run_postfix
fmt_s1_dir=$(printf "%.2f" "$squad1_direction")
fmt_s2_dir=$(printf "%.2f" "$squad2_direction")
fmt_s1_speed=$(printf "%.1f" "$squad1_speed_km_h")
fmt_s2_speed=$(printf "%.1f" "$squad2_speed_km_h")
fmt_intra1=$(printf "%.1f" "$intra_sq1_rw_speed_km_h")
fmt_intra2=$(printf "%.1f" "$intra_sq2_rw_speed_km_h")
fmt_s1_rad=$(printf "%.1f" "$squad1_radius")
fmt_s2_rad=$(printf "%.1f" "$squad2_radius")
fmt_dist=$(printf "%.1f" "$inter_squad_distance")

# _14_{num_subframes}_70_0_421_{txsquad_ues}_{s1_rad}_{s1_speed}_{s1_dir}_{rxsquad_ues}_{s2_rad}_{s2_speed}_{s2_dir}_{distance}_{intra1}_{intra2}_2.0_1.5_0_0/
ns3_dir_suffix=$(printf "_14_%d_70_0_421_%d_%s_%s_%s_%d_%s_%s_%s_%s_%s_%s_2.0_1.5_0_0/" \
  "$num_subframes" \
  "$txsquad_ues" "$fmt_s1_rad" "$fmt_s1_speed" "$fmt_s1_dir" \
  "$rxsquad_ues" "$fmt_s2_rad" "$fmt_s2_speed" "$fmt_s2_dir" \
  "$fmt_dist" "$fmt_intra1" "$fmt_intra2")

# ======== Seed + drop pairs ========
pairs=()
for ((i=0; i<num_drops; i++)); do
  seed=$((initial_seed + i))
  drop_idx=$((curr_drop_idx + i))
  pairs+=("$seed $drop_idx")
done

# Export for the job environment
export ns3_project_dir main_py_path ns3_channel_dir_start ns3_dir_suffix \
       convert_ns3_channels_path channel_saving_dir \
       num_subframes squad1_speed_km_h squad2_speed_km_h \
       intra_sq1_rw_speed_km_h intra_sq2_rw_speed_km_h \
       squad1_direction squad2_direction txsquad_ues rxsquad_ues \
       squad1_radius squad2_radius inter_squad_distance

# Exported env vars are already set above; tell parallel to pass them through
ENVVARS="ns3_project_dir,main_py_path,ns3_channel_dir_start,ns3_dir_suffix,convert_ns3_channels_path,channel_saving_dir,num_subframes,squad1_speed_km_h,squad2_speed_km_h,intra_sq1_rw_speed_km_h,intra_sq2_rw_speed_km_h,squad1_direction,squad2_direction,txsquad_ues,rxsquad_ues,squad1_radius,squad2_radius,inter_squad_distance"

printf "%s\n" "${pairs[@]}" \
| "$PARALLEL_BIN" -j "$JOBS" --colsep ' ' --env "$ENVVARS" '
    seed={1}
    drop_idx={2}

    echo "[seed=$seed drop=$drop_idx] Starting…"
    env LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
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
      --squad1_radius="$squad1_radius" \
      --squad2_radius="$squad2_radius" \
      --distance="$inter_squad_distance" \
      --format=both \
      --nUEs_sq1="$txsquad_ues" \
      --nUEs_sq2="$rxsquad_ues"

    raw_ns3_path="$ns3_project_dir/dmimo_ch/${ns3_channel_dir_start}${seed}${ns3_dir_suffix}"
    echo "[seed=$seed drop=$drop_idx] Raw path: $raw_ns3_path"

    out_folder="$channel_saving_dir/channels_${squad1_speed_km_h}_${squad2_speed_km_h}_r1_${squad1_radius}_r2_${squad2_radius}_d_${inter_squad_distance}_seed_${seed}_drop_${drop_idx}"
    if [[ -d "$out_folder" ]]; then
      echo "[seed=$seed drop=$drop_idx] Removing pre-existing out folder: $out_folder"
      rm -rf -- "$out_folder"
    fi

    python "$convert_ns3_channels_path" "$raw_ns3_path" "$out_folder"

    if [[ -f "$out_folder/00_config.npz" ]]; then
      rm -rf -- "$raw_ns3_path"
      echo "[seed=$seed drop=$drop_idx] Saved to $out_folder"
    else
      echo "[seed=$seed drop=$drop_idx] Conversion did not produce files; keeping raw at $raw_ns3_path"
    fi
'

echo "All jobs finished."
