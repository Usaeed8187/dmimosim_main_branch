import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from test_twomode_wesn_pred_real import predict_real
from scipy.ndimage import gaussian_filter


def cal_prediction_nmse(h_freq_csi_history):
       

    h_freq_csi_up_to_date = h_freq_csi_history[-1, ...]
    
    h_freq_csi_history = h_freq_csi_history[:-1, ...]

    h_freq_csi_history_real = h_freq_csi_history[:,:,:,0,:]
    h_freq_csi_history_imag = h_freq_csi_history[:,:,:,1,:]

    h_freq_csi_real = predict_real(h_freq_csi_history_real)

    h_freq_csi_up_to_date_real = h_freq_csi_up_to_date[:,:,0,:]
    nmse_outdated_real = tf.reduce_mean((h_freq_csi_history_real[-1, ...] - h_freq_csi_up_to_date_real) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_real) ** 2)
    nmse_pred_real = tf.reduce_mean((h_freq_csi_real - h_freq_csi_up_to_date_real) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_real) ** 2)
    print(f"NMSE of outdated CSI (Real): {nmse_outdated_real:.4f}, NMSE of predicted CSI (Real): {nmse_pred_real:.4f}")


    h_freq_csi_imag = predict_real(h_freq_csi_history_imag)

    h_freq_csi_up_to_date_imag = h_freq_csi_up_to_date[:,:,1,:]
    nmse_outdated_imag = tf.reduce_mean((h_freq_csi_history_imag[-1, ...] - h_freq_csi_up_to_date_imag) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_imag) ** 2)
    nmse_pred_imag = tf.reduce_mean((h_freq_csi_imag - h_freq_csi_up_to_date_imag) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_imag) ** 2)
    print(f"NMSE of outdated CSI (Imag): {nmse_outdated_imag:.4f}, NMSE of predicted CSI (Imag): {nmse_pred_imag:.4f}")
    

    return nmse_outdated_real, nmse_pred_real, nmse_outdated_imag, nmse_pred_imag

if __name__ == "__main__":

    total_cycles = 0
    chan_pred_nmse_real = []
    chan_pred_nmse_imag = []
    chan_outdated_nmse_real = []
    chan_outdated_nmse_imag = []

    data = np.load("tests/H_real_imag_1kmph.npz")
    H_real_imag = data["H_real_imag"]
    H_real_imag = H_real_imag[-300:,...]
    # H_real_imag = H_real_imag[..., :32]

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
        nmse_outdated_real, nmse_pred_real, nmse_outdated_imag, nmse_pred_imag = cal_prediction_nmse(h_freq_csi_history)
        chan_outdated_nmse_real.append(nmse_outdated_real)
        chan_pred_nmse_real.append(nmse_pred_real)
        chan_outdated_nmse_imag.append(nmse_outdated_imag)
        chan_pred_nmse_imag.append(nmse_pred_imag)

    chan_outdated_nmse_real = np.array(chan_outdated_nmse_real)
    chan_pred_nmse_real = np.array(chan_pred_nmse_real)
    chan_outdated_nmse_imag = np.array(chan_outdated_nmse_imag)
    chan_pred_nmse_imag = np.array(chan_pred_nmse_imag)

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

    hold = 1