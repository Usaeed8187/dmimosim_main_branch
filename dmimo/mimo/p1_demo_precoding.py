# Zero-Forcing (ZF) Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv


def weighted_mean_precoder(x, h, return_precoding_matrix=False):
    """
    MU-MIMO zero-forcing precoding supporting rank adaptation.

    :param x: data stream symbols
    :param h: channel coefficients
    :param ue_indices: receiver antenna indices for all users
    :param ue_ranks: number of streams (ranks) for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
    num_streams_per_tx = x.shape[-1]
    total_tx_ant = h.shape[-1]
    num_user = h.shape[0]
    num_user_streams = h.shape[-2]
    assert num_user_streams == num_streams_per_tx, "total number of streams must match"

    gram = tf.matmul(h, h, adjoint_a=True)
    sum_gram = tf.reduce_sum(gram, axis=0)
    _, v = tf.linalg.eig(sum_gram)
    v = v[:, :num_user_streams]

    h_all = []
    for k in range(num_user):
        # Update effective channels for all users
        h_ue = tf.gather(h, indices=k, axis=0)
        if ue_ranks[k] == num_rx_ant:
            # full rank
            h_all.append(h_ue)
        else:  # assuming rank==1
            # support only one stream adaptation
            assert(ue_ranks[k] == 1)
            # Calculate MRC weights
            g = tf.math.conj(tf.math.reduce_sum(h_ue, axis=-1, keepdims=True))
            # g = tf.matmul(g, tf.cast(1.0, tf.complex64)/tf.matmul(g, g, adjoint_a=True))
            h_eff = tf.matmul(g, h_ue, adjoint_a=True)
            h_all.append(h_eff)
        # Combine h_eff for all users
        h_zf = tf.concat(h_all, axis=-2)  # [..., num_tx_ant, num_streams_per_tx]

        # Compute pseudo inverse for precoding
        g = tf.matmul(h_zf, h_zf, adjoint_b=True)
        g = tf.matmul(h_zf, matrix_inv(g), adjoint_a=True)

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    g = g/tf.cast(norm, g.dtype)

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g
    else:
        return x_precoded
