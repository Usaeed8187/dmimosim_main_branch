# Zero-Forcing (ZF) Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv


def baseline_fiveG_precoder(x, precoding_matrices):
    """
    SU-MIMO precoding using 5G method, using the precoder returned by the user.

    :param x: data stream symbols
    :param precoding_matrices: precoding matrices for each RB
    :return: precoded data symbols
    """

    # Input dimensions:
    # x has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]

    # Precode
    x_precoded = tf.squeeze(tf.matmul(precoding_matrices, x), -1)

    return x_precoded

def dMIMO_p1_fiveG_max_min_precoder(x, h):
    """
    Phase 1 optimization based precoding

    :param x: data stream symbols
    :param h: channel coefficients
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]


    num_streams_per_tx = x.shape[-2]
    
    
    if return_precoding_matrix:
        return x_precoded, g
    else:
        return x_precoded


def dMIMO_p1_fiveG_max_min_demo_precoder(x, V, s, codebook, n_var, return_precoding_matrix=False):
    """
    Phase 1 optimization based precoding

    :param x: data stream symbols
    :param h: channel coefficients
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

    codebook = tf.cast(codebook, V.dtype)
    n_var = tf.cast(n_var, V.dtype)

    V_H = tf.transpose(tf.math.conj(V), perm=[0, 1, 2, 3, 4, 6, 5])
    s_2 = s ** 2
    H_gram = (V * s_2) @ V_H
    norm_H_gram = tf.norm(H_gram)
    H_gram = H_gram / norm_H_gram

    num_codebook_elements = np.product(codebook.shape[:-2])
    num_rx_ues = int(V.shape[1])

    curr_min_rate = np.inf
    max_min_rate = 0

    for codebook_idx in range(num_codebook_elements):
        
        for rx_ue_idx in range(num_rx_ues):
            
            P = codebook[codebook_idx, ...]
            curr_H_gram = H_gram[:, rx_ue_idx, ...]
            
            effective_SNR = (1 / n_var) * tf.matmul(tf.matmul(tf.linalg.adjoint(P), curr_H_gram), P)

            identity = tf.eye(2, batch_shape=effective_SNR.shape[:-2], dtype=effective_SNR.dtype)


            curr_capacity = tf.math.log(tf.linalg.det(identity + effective_SNR)) / tf.math.log(tf.constant(2.0, dtype=effective_SNR.dtype))
            curr_capacity = tf.math.real(curr_capacity)
            curr_sum_capacity = np.sum(curr_capacity)

            if curr_sum_capacity < curr_min_rate:
                curr_min_rate = curr_sum_capacity
        
        if curr_min_rate > max_min_rate:
            max_min_rate = curr_min_rate
            max_min_rate_idx = codebook_idx


    # Precode
    x_precoded = tf.squeeze(tf.matmul(codebook[max_min_rate_idx,...], x), -1)
    
    if return_precoding_matrix:
        return x_precoded, codebook[max_min_rate_idx,...]
    else:
        return x_precoded
