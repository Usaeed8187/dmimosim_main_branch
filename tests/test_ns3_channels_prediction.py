import argparse
from pathlib import Path
import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)


from dmimo.channel.kalman_filter_pred import kalman_filter_pred


def discover_slot_files(folder: Path) -> list[tuple[int, Path]]:
    pat = re.compile(r"dmimochans_(\d+)\.npz$")
    out: list[tuple[int, Path]] = []
    for p in folder.glob("dmimochans_*.npz"):
        m = pat.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_clean_p2p_channels(
    ns3_folder: Path,
    drop_idx: int,
    mobility: str,
    start_slot: int,
    end_slot: int,
    feedback_delay: int,
    rx_ant: int,
    tx_ant: int,
) -> tuple[np.ndarray, np.ndarray]:
    folder = ns3_folder / f"channels_{mobility}_{drop_idx}"
    if not folder.exists():
        raise FileNotFoundError(f"Channel folder not found: {folder}")

    files = discover_slot_files(folder)
    if not files:
        raise FileNotFoundError(f"No channels_*.npz files found under: {folder}")

    by_slot = {slot: path for slot, path in files}
    selected_slots = [s for s in range(start_slot, end_slot + 1, feedback_delay) if s in by_slot]
    if len(selected_slots) < 3:
        raise RuntimeError(
            f"Too few decimated slots found. Requested {start_slot}:{end_slot}:{feedback_delay}, "
            f"found {len(selected_slots)} in {folder}."
        )

    h_list = []
    for s in selected_slots:
        with np.load(by_slot[s]) as data:
            hdm = data["Hdm"]  # [all_rx_ant, all_tx_ant, num_syms, num_sc]
        h_p2p = hdm[:rx_ant, :tx_ant, :, :].astype(np.complex128)
        h_list.append(h_p2p)

    # [T_decimated, rx_ant, tx_ant, num_syms, num_sc]
    h_seq = np.stack(h_list, axis=0)
    return h_seq, np.asarray(selected_slots, dtype=int)


def add_complex_awgn(h: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    # h shape: [T, rx, tx, syms, sc]
    signal_power = float(np.mean(np.abs(h) ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = signal_power / max(snr_lin, 1e-15)
    noise = (
        rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)
    ) * np.sqrt(noise_var / 2.0)
    return h + noise, noise_var


def channels_to_tiles(h_seq: np.ndarray) -> np.ndarray:
    # h_seq: [T, rx, tx, syms, sc]
    # tiles: [T, Ntiles, D] where Ntiles=syms*sc, D=rx*tx
    t_len, rx, tx, syms, sc = h_seq.shape
    return h_seq.transpose(0, 3, 4, 1, 2).reshape(t_len, syms * sc, rx * tx)


def build_augmented_obs_matrix(d: int, p: int) -> np.ndarray:
    h_mat = np.zeros((d, p * d), dtype=np.complex128)
    h_mat[:, :d] = np.eye(d, dtype=np.complex128)
    return h_mat


def solve_riccati_steady_state_complex(
    f_aug: np.ndarray,
    q_aug: np.ndarray,
    h_mat: np.ndarray,
    r_mat: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    pd = f_aug.shape[0]
    p_minus = np.eye(pd, dtype=np.complex128)

    for _ in range(max_iter):
        s = h_mat @ p_minus @ h_mat.conj().T + r_mat
        k = p_minus @ h_mat.conj().T @ np.linalg.pinv(s)
        p_plus = p_minus - k @ h_mat @ p_minus
        p_next = f_aug @ p_plus @ f_aug.conj().T + q_aug
        p_next = 0.5 * (p_next + p_next.conj().T)

        if np.linalg.norm(p_next - p_minus, ord="fro") < tol:
            p_minus = p_next
            break
        p_minus = p_next

    s = h_mat @ p_minus @ h_mat.conj().T + r_mat
    k_ss = p_minus @ h_mat.conj().T @ np.linalg.pinv(s)
    return p_minus, k_ss


def full_kalman_predict_next(y_hist: np.ndarray, f_aug: np.ndarray, q_aug: np.ndarray, r_diag: np.ndarray) -> np.ndarray:
    t_len, d = y_hist.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist[-1].astype(np.complex128)

    h_mat = build_augmented_obs_matrix(d, p)
    r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))

    # initialize state stack from earliest available history
    state_stack = [y_hist[p - 1 - k].astype(np.complex128) for k in range(p)]
    z_hat = np.concatenate(state_stack, axis=0)
    p_hat = np.eye(p * d, dtype=np.complex128) * (float(np.mean(r_diag)) + 1e-6)
    eye_pd = np.eye(p * d, dtype=np.complex128)

    for t in range(p, t_len):
        z_pred = f_aug @ z_hat
        p_pred = f_aug @ p_hat @ f_aug.conj().T + q_aug

        innov = y_hist[t].astype(np.complex128) - h_mat @ z_pred
        s = h_mat @ p_pred @ h_mat.conj().T + r_mat
        k = p_pred @ h_mat.conj().T @ np.linalg.pinv(s)

        z_hat = z_pred + k @ innov
        p_hat = (eye_pd - k @ h_mat) @ p_pred

    z_next = f_aug @ z_hat
    return z_next[:d]


def steady_kalman_predict_next(
    y_hist: np.ndarray,
    f_aug: np.ndarray,
    q_aug: np.ndarray,
    r_diag: np.ndarray,
) -> np.ndarray:
    t_len, d = y_hist.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist[-1].astype(np.complex128)

    h_mat = build_augmented_obs_matrix(d, p)
    r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
    _, k_ss = solve_riccati_steady_state_complex(f_aug, q_aug, h_mat, r_mat)

    state_stack = [y_hist[p - 1 - k].astype(np.complex128) for k in range(p)]
    z_hat = np.concatenate(state_stack, axis=0)

    for t in range(p, t_len):
        z_pred = f_aug @ z_hat
        innov = y_hist[t].astype(np.complex128) - h_mat @ z_pred
        z_hat = z_pred + k_ss @ innov

    z_next = f_aug @ z_hat
    return z_next[:d]

def _init_augmented_state_batch(y_hist_batch: np.ndarray, p: int) -> np.ndarray:
    # y_hist_batch: [T, Ntiles, D] -> state: [Ntiles, p*D]
    state_stack = y_hist_batch[p - 1 :: -1][:p].astype(np.complex128)
    ntiles, d = state_stack.shape[1], state_stack.shape[2]
    return state_stack.transpose(1, 0, 2).reshape(ntiles, p * d)


def full_kalman_predict_next_batch(
    y_hist_batch: np.ndarray,
    f_aug: np.ndarray,
    q_aug: np.ndarray,
    r_diag: np.ndarray,
) -> np.ndarray:
    t_len, ntiles, d = y_hist_batch.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist_batch[-1].astype(np.complex128)

    pd = p * d
    r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
    z_hat = _init_augmented_state_batch(y_hist_batch, p)
    p_hat = np.broadcast_to(
        np.eye(pd, dtype=np.complex128) * (float(np.mean(r_diag)) + 1e-6),
        (ntiles, pd, pd),
    ).copy()

    for t in range(p, t_len):
        z_pred = z_hat @ f_aug.T
        p_pred = f_aug[None, :, :] @ p_hat @ f_aug.conj().T[None, :, :] + q_aug[None, :, :]

        innov = y_hist_batch[t].astype(np.complex128) - z_pred[:, :d]
        s = p_pred[:, :d, :d] + r_mat[None, :, :]
        k = p_pred[:, :, :d] @ np.linalg.pinv(s)

        z_hat = z_pred + (k @ innov[:, :, None]).squeeze(-1)
        p_hat = p_pred - (k @ p_pred[:, :d, :])

    z_next = z_hat @ f_aug.T
    return z_next[:, :d]


def steady_kalman_predict_next_batch(
    y_hist_batch: np.ndarray,
    f_aug: np.ndarray,
    k_ss: np.ndarray,
) -> np.ndarray:
    t_len, _, d = y_hist_batch.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist_batch[-1].astype(np.complex128)

    z_hat = _init_augmented_state_batch(y_hist_batch, p)

    for t in range(p, t_len):
        z_pred = z_hat @ f_aug.T
        innov = y_hist_batch[t].astype(np.complex128) - z_pred[:, :d]
        z_hat = z_pred + (k_ss[None, :, :] @ innov[:, :, None]).squeeze(-1)

    z_next = z_hat @ f_aug.T
    return z_next[:, :d]

def evaluate_nmse_over_chunks(
    h_clean_dec: np.ndarray,
    snr_db: float,
    history_len: int,
    ar_order: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)

    h_noisy_dec, noise_var = add_complex_awgn(h_clean_dec, snr_db, rng)
    tiles_clean = channels_to_tiles(h_clean_dec)  # [T, Ntiles, D]
    tiles_noisy = channels_to_tiles(h_noisy_dec)

    t_dec, _, d = tiles_clean.shape
    if t_dec <= history_len:
        raise ValueError("Not enough decimated samples for at least one chunk.")

    kf_helper = kalman_filter_pred(ar_order=ar_order)
    p_eff = min(ar_order, history_len - 1)

    # shared R across all tiles/subcarriers
    r_diag = noise_var * np.ones((d,), dtype=np.float64)

    num_full = 0.0
    den_full = 0.0
    num_ss = 0.0
    den_ss = 0.0

    for s in range(0, t_dec - history_len):
        y_hist_chunk = tiles_noisy[s : s + history_len]       # [history_len, Ntiles, D]
        x_true_next = tiles_clean[s + history_len]            # [Ntiles, D]

        a_blocks, q_proc = kf_helper._estimate_ar_p_q_joint(y_hist_chunk, p_eff)
        a_blocks = [a.conj() for a in a_blocks]
        f_aug, q_aug = kf_helper._build_augmented_system(a_blocks, q_proc)

        h_mat = build_augmented_obs_matrix(d, p_eff)
        r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
        _, k_ss = solve_riccati_steady_state_complex(f_aug, q_aug, h_mat, r_mat)

        x_pred_full = full_kalman_predict_next_batch(y_hist_chunk, f_aug, q_aug, r_diag)
        x_pred_ss = steady_kalman_predict_next_batch(y_hist_chunk, f_aug, k_ss)

        num_full += float(np.sum(np.abs(x_pred_full - x_true_next) ** 2))
        den_full += float(np.sum(np.abs(x_true_next) ** 2))

        num_ss += float(np.sum(np.abs(x_pred_ss - x_true_next) ** 2))
        den_ss += float(np.sum(np.abs(x_true_next) ** 2))

    nmse_full = num_full / max(den_full, 1e-15)
    nmse_ss = num_ss / max(den_ss, 1e-15)
    return nmse_ss, nmse_full


def main():
    parser = argparse.ArgumentParser(description="P2P Kalman channel prediction NMSE vs SNR on ns3 saved channels")
    parser.add_argument("--ns3-root", type=str, default="ns3", help="Root folder containing channels_<mobility>_<drop>")
    parser.add_argument("--mobility", type=str, default="high_mobility", help="Mobility folder key")
    parser.add_argument("--drop-idx", type=int, default=1)
    parser.add_argument("--start-slot", type=int, default=1)
    parser.add_argument("--end-slot", type=int, default=100)
    parser.add_argument("--feedback-delay", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--ar-order", type=int, default=4)
    parser.add_argument("--rx-ant", type=int, default=4)
    parser.add_argument("--tx-ant", type=int, default=4)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=15)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--plot-path", type=str, default="results/kalman_p2p_nmse_vs_snr.png")
    args = parser.parse_args()

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

    print(f"Loaded decimated slots ({len(selected_slots)}): {selected_slots[:10]}{' ...' if len(selected_slots) > 10 else ''}")
    print(f"Clean decimated channel tensor shape: {h_clean_dec.shape}")

    snr_vals = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)
    nmse_ss_vals = []
    nmse_full_vals = []

    for snr_db in snr_vals:
        nmse_ss, nmse_full = evaluate_nmse_over_chunks(
            h_clean_dec=h_clean_dec,
            snr_db=float(snr_db),
            history_len=args.history_len,
            ar_order=args.ar_order,
            seed=args.seed + int(snr_db),
        )
        nmse_ss_vals.append(nmse_ss)
        nmse_full_vals.append(nmse_full)
        print(f"SNR={snr_db:>2d} dB | NMSE steady={nmse_ss:.4e}, full={nmse_full:.4e}")

    nmse_ss_vals = np.asarray(nmse_ss_vals)
    nmse_full_vals = np.asarray(nmse_full_vals)

    out_path = Path(args.plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(snr_vals, nmse_ss_vals, marker="o", label="Steady-state Kalman")
    plt.plot(snr_vals, nmse_full_vals, marker="s", label="Full Kalman")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Channel prediction NMSE")
    plt.title(
        "P2P channel prediction NMSE vs SNR\n"
        f"({args.mobility}, drop={args.drop_idx}, delay={args.feedback_delay}, history={args.history_len}, AR={args.ar_order})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()