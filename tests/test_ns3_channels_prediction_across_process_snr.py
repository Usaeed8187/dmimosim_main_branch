"""P2P channel-prediction NMSE vs extra process-noise power Q.

Reuses the production-aligned evaluation in
test_ns3_channels_prediction_across_snr.py. Observation SNR (R) is held
fixed while temporally white process noise is added to the true CSI:

    x_t <- x_t + w_t,   Q = E[|x|^2] / 10^(process_snr_db/10)
    y_t  = x_t + v_t    with fixed observation SNR
"""

import argparse
from pathlib import Path
import sys
import os
import types

import matplotlib.pyplot as plt
import numpy as np

dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if "dmimo" not in sys.modules:
    dmimo_package = types.ModuleType("dmimo")
    dmimo_package.__path__ = [os.path.join(dmimo_root, "dmimo")]
    sys.modules["dmimo"] = dmimo_package
if "dmimo.channel" not in sys.modules:
    channel_package = types.ModuleType("dmimo.channel")
    channel_package.__path__ = [os.path.join(dmimo_root, "dmimo", "channel")]
    sys.modules["dmimo.channel"] = channel_package

from test_ns3_channels_prediction_across_snr import (
    evaluate_nmse_over_chunks,
    load_clean_p2p_channels,
)


def main():
    parser = argparse.ArgumentParser(
        description="P2P Kalman/WESN channel prediction NMSE vs process SNR (Q)"
    )
    parser.add_argument("--ns3-root", type=str, default="ns3")
    parser.add_argument("--mobility", type=str, default="higher_mobility")
    parser.add_argument("--drop-idx", type=int, default=1)
    parser.add_argument("--start-slot", type=int, default=1)
    parser.add_argument("--end-slot", type=int, default=100)
    parser.add_argument("--feedback-delay", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--ar-order", type=int, default=2)
    parser.add_argument("--esn-m", "--fb-m", dest="esn_m", type=int, default=2)
    parser.add_argument("--esn-k", "--fb-k", dest="esn_k", type=int, default=4)
    parser.add_argument(
        "--esn-num-freqs", "--fb-num-freqs", dest="esn_num_freqs", type=int, default=64
    )
    parser.add_argument(
        "--esn-activation",
        "--fb-activation",
        dest="esn_activation",
        type=str,
        default="tanh",
        choices=["identity", "tanh", "relu"],
    )
    parser.add_argument(
        "--esn-ls-reg", "--fb-ls-reg", dest="esn_ls_reg", type=float, default=1e-4
    )
    parser.add_argument("--wesn-lite-energy", type=float, default=0.80)
    parser.add_argument("--wesn-lite-readout-reg", type=float, default=1e-4)
    parser.add_argument(
        "--wesn-lite-readout-mode",
        choices=["matched-ridge", "centered-ridge"],
        default="centered-ridge",
    )
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--window-length", type=int, default=2)
    parser.add_argument("--subcarriers-per-rb", type=int, default=12)
    parser.add_argument("--rx-ant", type=int, default=4)
    parser.add_argument("--tx-ant", type=int, default=4)
    parser.add_argument(
        "--snr-db",
        type=float,
        default=15.0,
        help="Fixed observation SNR in dB (R). Default 15.",
    )
    parser.add_argument(
        "--process-snr-start",
        type=int,
        default=0,
        help="Smallest process SNR in dB: 10 log10(E[|x|^2]/Q). Low = large Q.",
    )
    parser.add_argument(
        "--process-snr-stop",
        type=int,
        default=26,
        help="Largest process SNR in dB (inclusive).",
    )
    parser.add_argument("--process-snr-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--offline-ratio", type=float, default=0.5)
    parser.add_argument(
        "--plot-path",
        type=str,
        default="results/kalman_p2p_nmse_vs_process_snr.png",
    )
    parser.add_argument("--esn-diagnostics", action="store_true", default=False)
    args = parser.parse_args()

    save_path = (
        f"results/kalman_p2p_nmse_vs_process_snr_data_activation_"
        f"{args.esn_activation}_{args.mobility}.npz"
    )

    h_clean_dec, selected_slots = load_clean_p2p_channels(
        ns3_folder=Path(args.ns3_root),
        drop_idx=args.drop_idx,
        mobility=args.mobility,
        start_slot=args.start_slot,
        end_slot=args.end_slot,
        feedback_delay=args.feedback_delay,
        rx_ant=args.rx_ant,
        tx_ant=args.tx_ant,
    )

    print(
        f"Loaded decimated slots ({len(selected_slots)}): "
        f"{selected_slots[:10]}{' ...' if len(selected_slots) > 10 else ''}"
    )
    print(f"Clean decimated channel tensor shape: {h_clean_dec.shape}")
    process_snr_vals = np.arange(
        args.process_snr_start, args.process_snr_stop + 1, args.process_snr_step
    )
    print(
        f"Observation SNR (R)={args.snr_db:g} dB; "
        f"process SNR (E[|x|^2]/Q) values={process_snr_vals.tolist()} dB; "
        f"history_len={args.history_len}."
    )

    nmse_ss_vals = []
    nmse_full_vals = []
    nmse_cfg_vals = []
    nmse_lite_vals = []
    nmse_rand_vals = []
    wesn_lite_rank_mean_vals = []
    wesn_lite_rank_mode_vals = []
    wesn_lite_state_dim_vals = []

    for process_snr_db in process_snr_vals:
        (
            nmse_ss,
            nmse_full,
            nmse_cfg,
            nmse_lite,
            nmse_rand,
            lite_rank,
        ) = evaluate_nmse_over_chunks(
            h_clean_dec=h_clean_dec,
            snr_db=float(args.snr_db),
            history_len=int(args.history_len),
            ar_order=args.ar_order,
            num_basis=args.esn_m,
            rp_degree=args.esn_k,
            num_freqs=args.esn_num_freqs,
            activation=args.esn_activation,
            ls_reg=float(args.esn_ls_reg),
            seed=args.seed,
            offline_ratio=float(args.offline_ratio),
            run_diagnostics=bool(args.esn_diagnostics),
            wesn_lite_energy=float(args.wesn_lite_energy),
            wesn_lite_readout_reg=float(args.wesn_lite_readout_reg),
            wesn_lite_readout_mode=str(args.wesn_lite_readout_mode),
            input_scale=float(args.input_scale),
            window_length=int(args.window_length),
            subcarriers_per_rb=int(args.subcarriers_per_rb),
            process_snr_db=float(process_snr_db),
        )
        nmse_ss_vals.append(nmse_ss)
        nmse_full_vals.append(nmse_full)
        nmse_cfg_vals.append(nmse_cfg)
        nmse_lite_vals.append(nmse_lite)
        nmse_rand_vals.append(nmse_rand)
        wesn_lite_rank_mean_vals.append(lite_rank["mean"])
        wesn_lite_rank_mode_vals.append(lite_rank["mode"])
        wesn_lite_state_dim_vals.append(lite_rank["state_dimension"])
        print(
            f"process SNR={int(process_snr_db):>2d} dB | NMSE steady_state_kf={nmse_ss:.4e}, "
            f"full={nmse_full:.4e}, "
            f"configured_wesn_balanced={nmse_cfg:.4e}, "
            f"configured_wesn_balanced_lite={nmse_lite:.4e}, "
            f"random_esn={nmse_rand:.4e} | Balanced WESN-Lite "
            f"Hankel order mean={lite_rank['mean']:.2f}, mode={lite_rank['mode']}, "
            f"R={lite_rank['state_dimension']}"
        )

    nmse_ss_vals = np.asarray(nmse_ss_vals)
    nmse_full_vals = np.asarray(nmse_full_vals)
    nmse_cfg_vals = np.asarray(nmse_cfg_vals)
    nmse_lite_vals = np.asarray(nmse_lite_vals)
    nmse_rand_vals = np.asarray(nmse_rand_vals)

    out_path = Path(args.plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_name(
        out_path.stem
        + f"_activation_{args.esn_activation}_{args.mobility}"
        + out_path.suffix
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(
        process_snr_vals,
        nmse_full_vals,
        marker="s",
        color="tab:orange",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Kalman Filter",
    )
    ax.plot(
        process_snr_vals,
        nmse_ss_vals,
        marker="P",
        color="tab:purple",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Steady-State KF",
    )
    ax.plot(
        process_snr_vals,
        nmse_cfg_vals,
        marker="^",
        color="tab:green",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Balanced Configured WESN",
    )
    ax.plot(
        process_snr_vals,
        nmse_lite_vals,
        marker="v",
        color="tab:brown",
        linestyle="--",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Balanced Configured WESN-Lite",
    )
    ax.plot(
        process_snr_vals,
        nmse_rand_vals,
        marker="d",
        color="0.45",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Random WESN",
    )

    ax.set_xlabel(r"Process SNR $10\log_{10}(\mathbb{E}|x|^{2}/Q)$ (dB)")
    ax.set_ylabel("Channel prediction NMSE")
    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    fig.tight_layout(pad=0.2)

    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

    np.savez(
        save_path,
        process_snr_vals=process_snr_vals,
        snr_db=np.asarray(args.snr_db),
        nmse_ss_vals=nmse_ss_vals,
        nmse_full_vals=nmse_full_vals,
        nmse_cfg_vals=nmse_cfg_vals,
        nmse_lite_vals=nmse_lite_vals,
        nmse_rand_vals=nmse_rand_vals,
        wesn_lite_energy=np.asarray(args.wesn_lite_energy),
        wesn_lite_readout_reg=np.asarray(args.wesn_lite_readout_reg),
        wesn_lite_readout_mode=np.asarray(args.wesn_lite_readout_mode),
        wesn_lite_rank_mean_vals=np.asarray(wesn_lite_rank_mean_vals),
        wesn_lite_rank_mode_vals=np.asarray(wesn_lite_rank_mode_vals),
        wesn_lite_state_dim_vals=np.asarray(wesn_lite_state_dim_vals),
    )
    print(f"Saved raw NMSE data to: {save_path}")


if __name__ == "__main__":
    main()
