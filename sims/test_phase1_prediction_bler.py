"""Independent Phase-1 current-vs-outdated full-CSI BLER experiment.

This intentionally does not alter the production Phase-1/MU-MIMO path.  It
compares a delayed full channel with current-slot (oracle) CSI while using the
existing Phase1v receiver and grad_ascent precoder. Receive SNR is clipped
before AWGN is added. No channel predictor is used.
"""

import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-prediction")

import numpy as np
import tensorflow as tf
from sionna.channel import AWGN, ApplyOFDMChannel
from sionna.ofdm import ResourceGrid
from sionna.utils import BinarySource

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The production channel package exposes an optional PyTorch DDPG predictor at
# import time. This standalone TensorFlow-only experiment does not use it, and
# the Sionna environment intentionally has no PyTorch installation.
ddpg_stub = types.ModuleType("dmimo.channel.ddpg_predictor")
ddpg_stub.default_ddpg_predictor = None
ddpg_stub.DDPGChannelPredictor = None
sys.modules.setdefault("dmimo.channel.ddpg_predictor", ddpg_stub)

from dmimo.channel import dMIMOChannels
from dmimo.config import Ns3Config, SimConfig
from dmimo.phase_1 import Phase1v


def corrected_grad_ascent_precoder(
    x, h, rx_snr_db, num_iterations=3, eps=1e-6, alpha=0.2,
    ptx=1.0, return_precoding_matrix=False,
):
    """Gradient ascent with a consistent channel-row convention."""
    x = np.asarray(x)
    h = np.asarray(h)
    rx_snr_db = np.asarray(rx_snr_db)
    if h.ndim != 4:
        raise ValueError("This phase-1 diagnostic expects subband CSI")
    num_users, num_sc, num_streams, num_tx = h.shape

    snr_lin = 10.0 ** (rx_snr_db / 10.0)
    channel_rows = h.astype(np.complex64)
    channel_rows /= np.linalg.norm(channel_rows, axis=-1, keepdims=True) + 1e-12

    p = np.zeros((num_sc, num_tx, num_streams), dtype=np.complex64)
    for stream in range(num_streams):
        p[:, :, stream] = np.sum(
            np.sqrt(snr_lin)[:, None, None]
            * np.conj(channel_rows[:, :, stream, :]),
            axis=0,
        )
    p /= np.linalg.norm(p, axis=(1, 2), keepdims=True) + 1e-12

    for _ in range(num_iterations):
        for sc in range(num_sc):
            p_sc = p[sc]
            rows = channel_rows[:, sc]
            # For channel rows h_s and beams p_s, the received stream
            # projections are h_s p_s. The objective uses their aggregate.
            projection = np.einsum("kst,ts->ks", rows, p_sc)
            gamma = snr_lin * np.sum(np.abs(projection) ** 2, axis=-1)
            weights = snr_lin / (gamma + eps)
            gradient = np.einsum(
                "k,kst,ks->ts", weights, np.conj(rows), projection
            )
            p_tilde = p_sc + alpha * gradient.astype(np.complex64)
            p[sc] = np.sqrt(ptx) * p_tilde / (np.linalg.norm(p_tilde) + 1e-12)

    x_precoded = np.matmul(
        x[..., :, None, :], np.transpose(p, (0, 2, 1))
    ).squeeze(-2)
    return x_precoded, p


class CappedSnrChannel:
    """TxSquad channel facade that clips ns-3 SNR before adding AWGN."""

    def __init__(self, base_channel, max_snr_db):
        self.base_channel = base_channel
        self.max_snr_db = float(max_snr_db)
        self.apply_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.complex64)
        self.awgn = AWGN(dtype=tf.complex64)

    def load_channel(self, slot_idx, batch_size=1, **kwargs):
        h, snr_db, rx_power = self.base_channel.load_channel(
            slot_idx=slot_idx, batch_size=batch_size, **kwargs
        )
        return h, np.minimum(np.asarray(snr_db), self.max_snr_db), rx_power

    def __call__(self, inputs):
        x, slot_idx = inputs[:2]
        h, snr_db, _ = self.load_channel(slot_idx, batch_size=int(x.shape[0]))
        y = self.apply_channel([x, h])
        noise_variance = np.power(10.0, -snr_db / 10.0).astype(np.float32)[..., None]
        return self.awgn([y, noise_variance]), None


class Phase1Metrics(Phase1v):
    """Phase1v variant that also returns pre-LDPC hard-decision bit counts."""

    def call(self, dmimo_chans, info_bits, precoding_matrices, precoding_method):
        info_bits = tf.reshape(
            info_bits,
            [self.batch_size, 1, self.rg.num_streams_per_tx,
             self.num_codewords, self.encoder.k],
        )
        coded = self.encoder(info_bits)
        coded = tf.reshape(
            coded,
            [self.batch_size, 1, self.rg.num_streams_per_tx,
             self.num_codewords * self.encoder.n],
        )
        interleaved = self.intlvr(coded)
        symbols = self.mapper(interleaved)
        resource_grid = self.rg_mapper(symbols)

        _, rx_snr_db_full, _ = dmimo_chans.load_channel(
            slot_idx=self.cfg.first_slot_idx, batch_size=self.batch_size
        )
        rx_per_ant = np.squeeze(np.mean(np.asarray(rx_snr_db_full), axis=(0, -1)))
        rx_snr_db = np.array([
            np.mean(rx_per_ant[..., ue * 2:ue * 2 + 2])
            for ue in range(self.cfg.num_scheduled_tx_ue)
        ])

        x_precoded, _, _, _, _ = self.p1_demo_precoder(
            [resource_grid, precoding_matrices, rx_snr_db, precoding_method]
        )
        received, _ = dmimo_chans([x_precoded, self.cfg.first_slot_idx])
        noise_variance = tf.cast(
            np.mean(np.abs(received) ** 2) / np.power(10.0, rx_snr_db / 10.0),
            tf.float32,
        )

        decoded_users = []
        uncoded_counts = []
        reference = np.asarray(interleaved)
        for rx_node in range(self.cfg.num_scheduled_tx_ue):
            curr_y = tf.gather(received, tf.range(rx_node * 2, rx_node * 2 + 2), axis=2)
            curr_h, err_var = self.ls_estimator([curr_y, noise_variance[rx_node]])
            curr_y = tf.gather(curr_y, self.rg.effective_subcarrier_ind, axis=-1)
            x_hat, no_eff = self.lmmse_equ(
                [curr_y, curr_h, err_var, noise_variance[rx_node]]
            )
            llr = self.demapper([x_hat, no_eff])

            hard_coded = np.asarray(llr > 0)
            uncoded_counts.append((
                int(np.count_nonzero(reference != hard_coded)),
                int(reference.size),
            ))

            llr = self.dintlvr(llr)
            llr = tf.reshape(
                llr,
                [self.batch_size, 1, self.rg.num_streams_per_tx,
                 self.num_codewords, self.encoder.n],
            )
            decoded = self.decoder(llr)
            decoded = tf.reshape(
                decoded,
                [self.batch_size,
                 self.rg.num_streams_per_tx * self.num_codewords * self.encoder.k],
            )
            decoded_users.append(decoded.numpy())

        return decoded_users, uncoded_counts


def phase1_csi_grid(cfg, num_bs_ant):
    effective = (cfg.fft_size // num_bs_ant) * num_bs_ant
    left = (cfg.fft_size - effective) // 2
    right = cfg.fft_size - effective - left
    return ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=cfg.fft_size,
        subcarrier_spacing=cfg.subcarrier_spacing,
        num_tx=1,
        num_streams_per_tx=num_bs_ant,
        cyclic_prefix_length=cfg.cyclic_prefix_len,
        num_guard_carriers=[left, right],
        dc_null=False,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
    )


def full_subband_csi(h, num_users, num_streams):
    """Return full-resolution dominant-mode CSI for rank-one or rank-two precoding."""
    h = np.asarray(h)[:, 0, : 2 * num_users, 0, :, :, :]
    batch, _, num_tx, num_sym, num_sc = h.shape
    h = h.reshape(batch, num_users, 2, num_tx, num_sym, num_sc)
    h = h.mean(axis=(0, 4))  # [user, rx_ant, tx_ant, subcarrier]
    h = np.transpose(h, (0, 3, 1, 2))  # [user, subcarrier, rx_ant, tx_ant]
    if num_streams > h.shape[-2]:
        raise ValueError(
            f"Requested {num_streams} streams but each UE has only "
            f"{h.shape[-2]} receive antennas"
        )
    # Use one right-singular channel row per transmitted stream. With two RX
    # antennas this supports rank one or rank two without codebook quantization.
    _, _, vh = np.linalg.svd(h, full_matrices=False)
    modes = vh[..., :num_streams, :]  # [user, subcarrier, stream, tx_ant]
    return np.transpose(modes, (0, 1, 3, 2))[:, None].astype(np.complex64)


def load_csi_modes(
    channel, first_slot, delay, batch_size, num_users, num_streams
):
    stale, _, _ = channel.load_channel(first_slot - delay, batch_size=batch_size)
    current, _, _ = channel.load_channel(first_slot, batch_size=batch_size)
    stale = full_subband_csi(stale, num_users, num_streams)
    current = full_subband_csi(current, num_users, num_streams)
    return {
        "outdated_full_csi": stale,
        "oracle_current_csi": current,
    }


def block_counts(reference, decoded, model):
    shape = [model.batch_size, 1, model.rg.num_streams_per_tx,
             model.num_codewords, model.encoder.k]
    reference = np.asarray(tf.reshape(reference, shape))
    decoded = np.asarray(tf.reshape(decoded, shape))
    errors = np.any(reference != decoded, axis=-1)
    return int(errors.sum()), int(errors.size)


def block_error_mask(reference, decoded, model):
    shape = [model.batch_size, model.rg.num_streams_per_tx, model.num_codewords, model.encoder.k]
    reference = np.asarray(tf.reshape(reference, shape))
    decoded = np.asarray(tf.reshape(decoded, shape))
    return np.any(reference != decoded, axis=-1)


def summarize_condition(
    aggregate_blocks,
    aggregate_uncoded_bits,
    per_user_blocks,
    per_user_uncoded_bits,
    joint_blocks,
):
    """Report worst-UE BER/BLER as the primary Phase-1 metrics."""
    per_user_bler = [
        {
            "user": user_idx,
            "block_errors": value[0],
            "blocks": value[1],
            "bler": value[0] / value[1],
        }
        for user_idx, value in enumerate(per_user_blocks)
    ]
    per_user_uncoded_ber = [
        {
            "user": user_idx,
            "uncoded_bit_errors": value[0],
            "uncoded_bits": value[1],
            "uncoded_ber": value[0] / value[1],
        }
        for user_idx, value in enumerate(per_user_uncoded_bits)
    ]
    worst_bler = max(per_user_bler, key=lambda value: value["bler"])
    worst_ber = max(
        per_user_uncoded_ber, key=lambda value: value["uncoded_ber"]
    )

    return {
        "evaluation_metric": "worst_user",
        "block_errors": worst_bler["block_errors"],
        "blocks": worst_bler["blocks"],
        "bler": worst_bler["bler"],
        "worst_bler_user": worst_bler["user"],
        "uncoded_bit_errors": worst_ber["uncoded_bit_errors"],
        "uncoded_bits": worst_ber["uncoded_bits"],
        "uncoded_ber": worst_ber["uncoded_ber"],
        "worst_uncoded_ber_user": worst_ber["user"],
        "aggregate_block_errors": aggregate_blocks[0],
        "aggregate_blocks": aggregate_blocks[1],
        "aggregate_bler": aggregate_blocks[0] / aggregate_blocks[1],
        "aggregate_uncoded_bit_errors": aggregate_uncoded_bits[0],
        "aggregate_uncoded_bits": aggregate_uncoded_bits[1],
        "aggregate_uncoded_ber": (
            aggregate_uncoded_bits[0] / aggregate_uncoded_bits[1]
        ),
        "per_user_bler": per_user_bler,
        "per_user_uncoded_ber": per_user_uncoded_ber,
        "joint_multicast_block_errors": joint_blocks[0],
        "joint_multicast_blocks": joint_blocks[1],
        "joint_multicast_bler": joint_blocks[0] / joint_blocks[1],
    }


def run(args):
    if args.correct_grad_ascent_conjugation:
        p1_precoder_module = importlib.import_module("dmimo.mimo.p1_demo_precoder")
        p1_precoder_module.grad_ascent_precoder = corrected_grad_ascent_precoder
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    totals = {
        mode: {str(snr): [0, 0] for snr in args.max_snr_db}
        for mode in ("outdated_full_csi", "oracle_current_csi")
    }
    uncoded_totals = {
        mode: {str(snr): [0, 0] for snr in args.max_snr_db}
        for mode in ("outdated_full_csi", "oracle_current_csi")
    }
    per_user_totals = {
        mode: {str(snr): [[0, 0] for _ in range(args.num_tx_ues)] for snr in args.max_snr_db}
        for mode in ("outdated_full_csi", "oracle_current_csi")
    }
    per_user_uncoded_totals = {
        mode: {str(snr): [[0, 0] for _ in range(args.num_tx_ues)] for snr in args.max_snr_db}
        for mode in ("outdated_full_csi", "oracle_current_csi")
    }
    joint_totals = {
        mode: {str(snr): [0, 0] for snr in args.max_snr_db}
        for mode in ("outdated_full_csi", "oracle_current_csi")
    }
    raw_snr = []
    selected_mcs = None

    for mobility in args.mobility:
        for drop in args.drop:
            ns3 = Ns3Config(
                data_folder=str(REPO_ROOT / "ns3" / f"channels_{mobility}_{drop}"),
                total_slots=99,
            )
            tx_mask = np.zeros(ns3.num_txue, dtype=bool)
            tx_mask[: args.num_tx_ues] = True
            rx_mask = np.zeros(ns3.num_rxue, dtype=bool)
            rx_mask[: args.num_rx_ues] = True
            ns3.update_ue_selection(tx_mask, rx_mask)
            base_channel = dMIMOChannels(ns3, "TxSquad", forward=True, add_noise=False)

            cfg = SimConfig()
            cfg.num_slots_p1 = 2
            cfg.csi_delay = args.csi_delay
            cfg.num_scheduled_tx_ue = args.num_tx_ues
            cfg.num_tx_streams = args.num_streams
            cfg.modulation_order = args.phase1_modulation_order
            cfg.code_rate = args.phase1_code_rate
            rg_csi = phase1_csi_grid(cfg, ns3.num_bs_ant)
            model = Phase1Metrics(cfg, rg_csi)
            selected_mcs = (cfg.modulation_order, cfg.code_rate)
            binary_source = BinarySource(seed=args.seed + int(drop))

            for first_slot in args.slot:
                cfg.first_slot_idx = first_slot
                raw = base_channel.load_channel(first_slot, batch_size=cfg.num_slots_p1)[1]
                raw_snr.extend(np.asarray(raw).ravel().tolist())
                csi_modes = load_csi_modes(
                    base_channel, first_slot, cfg.csi_delay,
                    cfg.num_slots_p1, args.num_tx_ues, args.num_streams,
                )
                info_bits = binary_source([cfg.num_slots_p1, model.num_bits_per_frame])

                for mode, csi in csi_modes.items():
                    for snr in args.max_snr_db:
                        # Use the same AWGN realization for every CSI mode in a
                        # given frame/SNR condition (paired comparison).
                        noise_seed = (
                            args.seed + int(drop) * 100_000 + first_slot * 100
                            + int(round((snr + 100.0) * 10.0))
                        )
                        tf.random.set_seed(noise_seed)
                        capped_channel = CappedSnrChannel(base_channel, snr)
                        decoded_users, uncoded_counts = model(
                            capped_channel, info_bits, csi,
                            precoding_method="grad_ascent",
                        )
                        error_masks = []
                        for user_idx, decoded in enumerate(decoded_users):
                            errors, blocks = block_counts(info_bits, decoded, model)
                            totals[mode][str(snr)][0] += errors
                            totals[mode][str(snr)][1] += blocks
                            per_user_totals[mode][str(snr)][user_idx][0] += errors
                            per_user_totals[mode][str(snr)][user_idx][1] += blocks
                            error_masks.append(block_error_mask(info_bits, decoded, model))
                        joint_errors = np.any(np.stack(error_masks), axis=0)
                        joint_totals[mode][str(snr)][0] += int(joint_errors.sum())
                        joint_totals[mode][str(snr)][1] += int(joint_errors.size)
                        for user_idx, (errors, bits) in enumerate(uncoded_counts):
                            uncoded_totals[mode][str(snr)][0] += errors
                            uncoded_totals[mode][str(snr)][1] += bits
                            per_user_uncoded_totals[mode][str(snr)][user_idx][0] += errors
                            per_user_uncoded_totals[mode][str(snr)][user_idx][1] += bits

                print(f"completed {mobility} drop={drop} slot={first_slot}", flush=True)

    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    results = {
        "settings": settings,
        "phase1_mcs": {
            "modulation_order": selected_mcs[0],
            "code_rate": selected_mcs[1],
            "num_streams": model.rg.num_streams_per_tx,
            "phase1_bits_per_cycle": model.num_bits_per_frame * model.batch_size,
            "precoder": "grad_ascent",
            "correct_grad_ascent_conjugation": args.correct_grad_ascent_conjugation,
            "feedback": "full-precision unquantized dominant-mode subband CSI",
            "prediction": "none",
        },
        "raw_phase1_snr_db": {
            "min": float(np.min(raw_snr)),
            "mean": float(np.mean(raw_snr)),
            "max": float(np.max(raw_snr)),
        },
        "results": {
            mode: {
                snr: summarize_condition(
                    aggregate_blocks=val,
                    aggregate_uncoded_bits=uncoded_totals[mode][snr],
                    per_user_blocks=per_user_totals[mode][snr],
                    per_user_uncoded_bits=per_user_uncoded_totals[mode][snr],
                    joint_blocks=joint_totals[mode][snr],
                )
                for snr, val in by_snr.items()
            }
            for mode, by_snr in totals.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobility", nargs="+", default=["high_mobility"])
    parser.add_argument("--drop", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--slot", nargs="+", type=int, default=[33, 49, 65, 81])
    parser.add_argument("--max-snr-db", nargs="+", type=float, default=[0, 5, 10, 15, 20])
    parser.add_argument("--num-tx-ues", type=int, default=4)
    parser.add_argument("--num-rx-ues", type=int, default=4)
    parser.add_argument("--num-streams", type=int, choices=[1, 2], default=1)
    parser.add_argument("--csi-delay", type=int, default=4)
    parser.add_argument("--phase1-modulation-order", type=int, default=6)
    parser.add_argument("--phase1-code-rate", type=float, default=0.5)
    parser.add_argument("--correct-grad-ascent-conjugation", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "results" / "phase1_prediction_bler.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
