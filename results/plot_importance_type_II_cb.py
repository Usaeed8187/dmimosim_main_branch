"""
Plot average throughput versus number of simulated drops for Type-II CSI matrices.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def _parse_bool(value):
    return str(value).lower() in ("true", "1", "yes")


def _load_throughput(file_path):
    data = np.load(file_path, allow_pickle=True)
    throughput = data["throughput"]
    return float(np.squeeze(throughput))


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python sims/plot_mu_mimo_matrix_importance.py "
            "<mobility> <drop_indices_csv> <rx_ue> <num_txue_sel> "
            "[mod_order] [code_rate] [link_adapt] [csi_quantization_on]"
        )
        sys.exit(1)

    mobility = sys.argv[1]
    drop_indices = [idx.strip() for idx in sys.argv[2].split(",") if idx.strip()]
    drop_numbers = [int(idx) for idx in drop_indices]
    rx_ue = int(sys.argv[3])
    num_txue_sel = int(sys.argv[4])

    modulation_order = int(sys.argv[5]) if len(sys.argv) >= 6 else 4
    code_rate = float(sys.argv[6]) if len(sys.argv) >= 7 else 1 / 2
    link_adapt = _parse_bool(sys.argv[7]) if len(sys.argv) >= 8 else True
    csi_quantization_on = _parse_bool(sys.argv[8]) if len(sys.argv) >= 9 else True

    if link_adapt:
        mcs_string = "link_adapt"
    else:
        mcs_string = f"mod_order_{modulation_order}_code_rate_{code_rate}"

    scenarios = [
        {
            "name": "perfect_w1",
            "label": "Perfect W_1",
            "file_suffix": "perfect_W_1_matrix",
        },
        {
            "name": "perfect_wc1",
            "label": "Perfect W_C1",
            "file_suffix": "perfect_W_C1_matrix",
        },
        {
            "name": "perfect_wc2_t",
            "label": "Perfect W_C2_t",
            "file_suffix": "perfect_W_C2_t_matrix",
        },
        {
            "name": "perfect_everything",
            "label": "Perfect CSI",
            "file_suffix": "perfect_CSI_True",
        },
        {
            "name": "outdated_everything",
            "label": "Outdated CSI",
            "file_suffix": "perfect_CSI_False",
        },
    ]

    plt.figure(figsize=(8, 5))

    for scenario in scenarios:
        throughputs = []
        for drop_idx in drop_indices:
            folder_name = f"channels_{mobility}_{drop_idx}"
            file_name = (
                "mu_mimo_results_{}_rx_UE_{}_tx_UE_{}_{}_pmi_quantization_{}.npz"
            ).format(
                mcs_string,
                rx_ue,
                num_txue_sel,
                scenario["file_suffix"],
                csi_quantization_on,
            )
            file_path = os.path.join(
                "results",
                "channels_multiple_mu_mimo",
                folder_name,
                file_name,
            )
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing results file: {file_path}")

            throughputs.append(_load_throughput(file_path))

        plt.plot(
            drop_numbers,
            throughputs,
            marker="o",
            label=scenario["label"],
        )

    plt.grid(True)
    plt.xlabel("Drop index")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Throughput across simulated drops")
    plt.legend()

    output_dir = os.path.join("results", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"mu_mimo_matrix_importance_{mobility}.png",
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()