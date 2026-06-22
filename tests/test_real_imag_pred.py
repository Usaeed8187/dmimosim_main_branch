import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from time import perf_counter
from types import MethodType

from test_twomode_wesn_pred_real import twomode_wesn_pred
from scipy.ndimage import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-rbs",
        type=int,
        default=32,
        help="Number of contiguous RB averages to form from the subcarrier axis.",
    )
    parser.add_argument(
        "--disable-rb-averaging",
        action="store_true",
        help="Use the raw subcarrier axis instead of averaging subcarriers into RBs.",
    )
    parser.add_argument(
        "--readout-solver",
        choices=("cholesky", "pinv"),
        default="pinv",
        help="Linear solver to use for W_out ridge readout training.",
    )
    return parser.parse_args()


def average_subcarriers_to_rbs(h_real_imag, num_rbs):
    if num_rbs <= 0:
        raise ValueError("num_rbs must be positive")

    num_subcarriers = h_real_imag.shape[-1]
    if num_rbs > num_subcarriers:
        raise ValueError(
            f"num_rbs={num_rbs} cannot exceed num_subcarriers={num_subcarriers}"
        )

    h_complex = h_real_imag[..., 0, :] + 1j * h_real_imag[..., 1, :]
    rb_chunks = np.array_split(h_complex, num_rbs, axis=-1)
    h_complex_rb = np.stack(
        [np.mean(chunk, axis=-1) for chunk in rb_chunks],
        axis=-1,
    )

    h_real_imag_rb = np.stack(
        [np.real(h_complex_rb), np.imag(h_complex_rb)],
        axis=-2,
    )

    rb_sizes = [chunk.shape[-1] for chunk in rb_chunks]
    return h_real_imag_rb.astype(h_real_imag.dtype, copy=False), rb_sizes


def reg_p_inv_pinv(self, X):
    feature_dim = X.shape[0]
    gram = X @ X.T + self.reg * np.eye(feature_dim, dtype=self.dtype)
    return X.T @ np.linalg.pinv(gram)


def reg_p_inv_cholesky(self, X):
    feature_dim = X.shape[0]
    gram = X @ X.T + self.reg * np.eye(feature_dim, dtype=self.dtype)

    try:
        chol = np.linalg.cholesky(gram)
        temp = np.linalg.solve(chol, X)
        reg_inv = np.linalg.solve(chol.T, temp)
    except np.linalg.LinAlgError:
        reg_inv = np.linalg.solve(gram, X)

    return reg_inv.T


def set_readout_solver(twomode_predictor, readout_solver):
    if readout_solver == "pinv":
        twomode_predictor.reg_p_inv = MethodType(reg_p_inv_pinv, twomode_predictor)
    elif readout_solver == "cholesky":
        twomode_predictor.reg_p_inv = MethodType(reg_p_inv_cholesky, twomode_predictor)
    else:
        raise ValueError(f"Unsupported readout solver: {readout_solver}")


def predict_esn(h_freq_csi_history, readout_solver="pinv"):
    h_freq_csi_history = np.asarray(h_freq_csi_history)
    twomode_predictor = twomode_wesn_pred(
        history_length=h_freq_csi_history.shape[0],
        num_rx_ant=h_freq_csi_history.shape[1],
        num_tx_ant=h_freq_csi_history.shape[2],
        type=h_freq_csi_history.dtype,
    )
    set_readout_solver(twomode_predictor, readout_solver)
    return np.asarray(twomode_predictor.predict(h_freq_csi_history))


def cal_nmse(prediction, target):
    prediction = tf.cast(prediction, dtype=target.dtype)
    return float(
        tf.reduce_mean((prediction - target) ** 2)
        / tf.reduce_mean(target ** 2)
    )


def cal_prediction_nmse(h_freq_csi_history, readout_solver="pinv"):
    h_freq_csi_up_to_date = h_freq_csi_history[-1, ...]
    h_freq_csi_history = h_freq_csi_history[:-1, ...]

    h_freq_csi_history_real = h_freq_csi_history[:,:,:,0,:]
    h_freq_csi_history_imag = h_freq_csi_history[:,:,:,1,:]

    real_start_time = perf_counter()
    h_freq_csi_real = predict_esn(h_freq_csi_history_real, readout_solver)
    real_prediction_time = perf_counter() - real_start_time

    h_freq_csi_up_to_date_real = h_freq_csi_up_to_date[:,:,0,:]
    nmse_outdated_real = cal_nmse(h_freq_csi_history_real[-1, ...], h_freq_csi_up_to_date_real)
    nmse_pred_real = cal_nmse(h_freq_csi_real, h_freq_csi_up_to_date_real)
    print(f"NMSE of outdated CSI (Real): {nmse_outdated_real:.4f}, NMSE of predicted CSI (Real): {nmse_pred_real:.4f}")

    imag_start_time = perf_counter()
    h_freq_csi_imag = predict_esn(h_freq_csi_history_imag, readout_solver)
    imag_prediction_time = perf_counter() - imag_start_time
    separated_prediction_time = real_prediction_time + imag_prediction_time

    h_freq_csi_up_to_date_imag = h_freq_csi_up_to_date[:,:,1,:]
    nmse_outdated_imag = cal_nmse(h_freq_csi_history_imag[-1, ...], h_freq_csi_up_to_date_imag)
    nmse_pred_imag = cal_nmse(h_freq_csi_imag, h_freq_csi_up_to_date_imag)
    print(f"NMSE of outdated CSI (Imag): {nmse_outdated_imag:.4f}, NMSE of predicted CSI (Imag): {nmse_pred_imag:.4f}")

    h_freq_csi_history_stacked = np.concatenate(
        [h_freq_csi_history_real, h_freq_csi_history_imag],
        axis=-1,
    )
    stacked_start_time = perf_counter()
    h_freq_csi_stacked = predict_esn(h_freq_csi_history_stacked, readout_solver)
    stacked_prediction_time = perf_counter() - stacked_start_time

    num_freq_re = h_freq_csi_up_to_date_real.shape[-1]
    h_freq_csi_stacked_real = h_freq_csi_stacked[..., :num_freq_re]
    h_freq_csi_stacked_imag = h_freq_csi_stacked[..., num_freq_re:]

    h_freq_csi_separated = np.concatenate([h_freq_csi_real, h_freq_csi_imag], axis=-1)
    h_freq_csi_joint = np.concatenate([h_freq_csi_stacked_real, h_freq_csi_stacked_imag], axis=-1)
    h_freq_csi_target = np.concatenate([h_freq_csi_up_to_date_real, h_freq_csi_up_to_date_imag], axis=-1)

    nmse_pred_separated = cal_nmse(h_freq_csi_separated, h_freq_csi_target)
    nmse_pred_joint = cal_nmse(h_freq_csi_joint, h_freq_csi_target)
    print(
        f"Combined NMSE separated prediction: {nmse_pred_separated:.4f}, "
        f"Combined NMSE stacked prediction: {nmse_pred_joint:.4f}"
    )
    print(
        f"Prediction time separated real+imag: {separated_prediction_time:.4f}s, "
        f"stacked: {stacked_prediction_time:.4f}s"
    )

    return (
        nmse_outdated_real,
        nmse_pred_real,
        nmse_outdated_imag,
        nmse_pred_imag,
        nmse_pred_separated,
        nmse_pred_joint,
        separated_prediction_time,
        stacked_prediction_time,
    )

if __name__ == "__main__":

    args = parse_args()
    print(f"Using W_out readout solver: {args.readout_solver}")

    total_cycles = 0
    chan_pred_nmse_real = []
    chan_pred_nmse_imag = []
    chan_outdated_nmse_real = []
    chan_outdated_nmse_imag = []
    chan_pred_nmse_separated = []
    chan_pred_nmse_joint = []
    separated_prediction_times = []
    stacked_prediction_times = []

    data = np.load("tests/H_real_imag_1kmph.npz")
    H_real_imag = data["H_real_imag"]
    H_real_imag = H_real_imag[-300:,...]
    # H_real_imag = H_real_imag[..., :32]

    if args.disable_rb_averaging:
        print(f"Using raw subcarriers: {H_real_imag.shape[-1]}")
    else:
        original_num_subcarriers = H_real_imag.shape[-1]
        H_real_imag, rb_sizes = average_subcarriers_to_rbs(
            H_real_imag,
            args.num_rbs,
        )
        print(
            f"Averaged {original_num_subcarriers} subcarriers into "
            f"{H_real_imag.shape[-1]} RBs; RB sizes: {rb_sizes}"
        )

    sigma = (2, 0, 0, 0, 0)  # adjust 2 and 1 as needed
    H_real_imag = gaussian_filter(H_real_imag, sigma=sigma)

    plt.figure()
    plt.plot(H_real_imag[:, 0, 0, 0, 0])
    plt.savefig('across_time.png')

    plt.figure()
    plt.plot(H_real_imag[0, 0, 0, 0, :])
    plt.savefig('across_freq.png')

    history_length = 8

    for i in range(H_real_imag.shape[0] - (history_length+1)):
        h_freq_csi_history = H_real_imag[i:i+history_length+1, ...]
        (
            nmse_outdated_real,
            nmse_pred_real,
            nmse_outdated_imag,
            nmse_pred_imag,
            nmse_pred_separated,
            nmse_pred_joint,
            separated_prediction_time,
            stacked_prediction_time,
        ) = cal_prediction_nmse(h_freq_csi_history, args.readout_solver)
        chan_outdated_nmse_real.append(nmse_outdated_real)
        chan_pred_nmse_real.append(nmse_pred_real)
        chan_outdated_nmse_imag.append(nmse_outdated_imag)
        chan_pred_nmse_imag.append(nmse_pred_imag)
        chan_pred_nmse_separated.append(nmse_pred_separated)
        chan_pred_nmse_joint.append(nmse_pred_joint)
        separated_prediction_times.append(separated_prediction_time)
        stacked_prediction_times.append(stacked_prediction_time)

    chan_outdated_nmse_real = np.array(chan_outdated_nmse_real)
    chan_pred_nmse_real = np.array(chan_pred_nmse_real)
    chan_outdated_nmse_imag = np.array(chan_outdated_nmse_imag)
    chan_pred_nmse_imag = np.array(chan_pred_nmse_imag)
    chan_pred_nmse_separated = np.array(chan_pred_nmse_separated)
    chan_pred_nmse_joint = np.array(chan_pred_nmse_joint)
    separated_prediction_times = np.array(separated_prediction_times)
    stacked_prediction_times = np.array(stacked_prediction_times)

    plt.figure()
    plt.plot(chan_outdated_nmse_real, label="Outdated CSI NMSE (Real)")
    plt.plot(chan_pred_nmse_real, label="Predicted CSI NMSE (Real)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("NMSE")
    plt.legend()
    plt.grid()
    plt.ylim(0, 1)
    plt.savefig('nmse_comparison_real.png')

    plt.figure()
    plt.plot(chan_outdated_nmse_imag, label="Outdated CSI NMSE (Imag)")
    plt.plot(chan_pred_nmse_imag, label="Predicted CSI NMSE (Imag)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("NMSE")
    plt.legend()
    plt.grid()
    plt.ylim(0, 1)
    plt.savefig('nmse_comparison_imag.png')

    print("Average NMSE of outdated CSI (Real): {:.4f}".format(np.mean(chan_outdated_nmse_real)))
    print("Average NMSE of predicted CSI (Real): {:.4f}".format(np.mean(chan_pred_nmse_real)))
    print("Average NMSE of outdated CSI (Imag): {:.4f}".format(np.mean(chan_outdated_nmse_imag)))
    print("Average NMSE of predicted CSI (Imag): {:.4f}".format(np.mean(chan_pred_nmse_imag)))
    print("Average combined NMSE of separated real+imag prediction: {:.4f}".format(np.mean(chan_pred_nmse_separated)))
    print("Average combined NMSE of stacked real+imag prediction: {:.4f}".format(np.mean(chan_pred_nmse_joint)))
    print("Average separated real+imag prediction time: {:.4f}s".format(np.mean(separated_prediction_times)))
    print("Average stacked real+imag prediction time: {:.4f}s".format(np.mean(stacked_prediction_times)))
    print(
        "Separated/stacked prediction time ratio: {:.4f}".format(
            np.mean(separated_prediction_times) / np.mean(stacked_prediction_times)
        )
    )

    hold = 1
