# Zero-Forcing (ZF) Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv


def weighted_mean_precoder(x, h, rx_snr_db, num_iterations=3, return_precoding_matrix=False):
    """
    MU-MIMO zero-forcing precoding supporting rank adaptation.

    :param x: data stream symbols
    :param h: channel coefficients
    :param rx_snr_db: receiver snr for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [num_rx_ues, num_rx_ant_per_ue, num_tx_ant_per_ue]
    num_streams_per_tx = x.shape[-1]
    total_tx_ant = h.shape[-1]
    num_user = h.shape[0]
    num_user_streams = h.shape[-2]
    assert num_user_streams == num_streams_per_tx, "total number of streams must match"

    rx_snr_linear = 10**(rx_snr_db / 10)
    n_var = 1 / rx_snr_linear

    w_k = np.ones(num_user)
    best_w_k = w_k.copy()
    best_SINR = None
    starting_SINR = None
    sigma2 = 1 / (rx_snr_linear * tf.reduce_mean(tf.abs(tf.linalg.norm(h, axis=(1, 2)))))  # Adjusted noise

    SINR = np.zeros(num_user)

    for iter in range(num_iterations):

        w_k_reshaped = tf.reshape(w_k, (-1, 1, 1))
        w_k_reshaped = tf.cast(w_k_reshaped, h.dtype)

        gram = tf.matmul(h * w_k_reshaped, h, adjoint_a=True)
        sum_gram = tf.reduce_sum(gram, axis=0)
        _, v = tf.linalg.eig(sum_gram)
        v = v[:, :num_user_streams]

        prev_SINR = SINR.copy()

        for k in range(num_user):

            h_ue = tf.gather(h, indices=k, axis=0)

            H_k_v = tf.matmul(h_ue, v)

            if num_user_streams > 1:
                SINR_0 = tf.abs(tf.linalg.norm(H_k_v[:, 0]))**2 / (tf.abs(tf.linalg.norm(H_k_v[:, 1]))**2 + n_var)
                SINR_1 = tf.abs(tf.linalg.norm(H_k_v[:, 1]))**2 / (tf.abs(tf.linalg.norm(H_k_v[:, 0]))**2 + n_var)
                SINR[k] = tf.reduce_min([SINR_0, SINR_1])
            else:
                SINR[k] = tf.abs(tf.linalg.norm(H_k_v))**2 / (n_var)

        if iter == 0:
            starting_SINR = SINR.copy()

        print("\n min SINR at step {} of precoding: ".format(iter), np.min(SINR))
        print("max SINR at step {} of precoding: ".format(iter), np.max(SINR))
        print("mean SINR at step {} of precoding: ".format(iter), np.mean(SINR))
        print("all SINRs at step {} of precoding: ".format(iter), SINR)

        if np.min(SINR) > np.min(prev_SINR):
            best_w_k = w_k.copy()
            best_SINR = SINR.copy()

        w_k = 1 / np.sqrt(SINR)
        w_k = w_k / np.sum(w_k)

    if num_iterations > 0:
        print("\n min SINR after precoding: ", np.min(best_SINR))
        print("max SINR after precoding: ", np.max(best_SINR))
        print("mean SINR after precoding: ", np.mean(best_SINR))
        print("all SINRs after precoding: ", best_SINR)

    w_k_reshaped = tf.reshape(best_w_k, (-1, 1, 1))
    w_k_reshaped = tf.cast(w_k_reshaped, h.dtype)

    gram = tf.matmul(h * w_k_reshaped, h, adjoint_a=True)
    sum_gram = tf.reduce_sum(gram, axis=0)
    _, v = tf.linalg.eig(sum_gram)
    v = v[:, :num_user_streams]

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(v)**2, axis=-2, keepdims=True))
    g = v/tf.cast(norm, v.dtype)
    g = tf.cast(g, x.dtype)

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g, starting_SINR, best_SINR
    else:
        return x_precoded, None, starting_SINR, best_SINR