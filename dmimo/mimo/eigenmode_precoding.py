import tensorflow as tf


def eigenmode_precoder(x, h, return_precoding_matrix=False):
    """
    MU-MIMO precoding using SVD
    :param x: Transmit symbols [batch_size, num_ues, num_tx_ant, num_ofdm_sym, fft_size]
    :param h: Channel matrix [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
    :return: Precoded symbols [batch_size, num_ues, 1, num_ofdm_sym, fft_size]
    """

    
    batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym, fft_size = h.shape
    h_reshape = tf.transpose(h, perm=[0,5,6,1,2,3,4]) # batch_size, num_ofdm_symbols, fft_size, num_rx, num_rx_ant, num_tx, num_tx_ant
    h_reshape = tf.reshape(h_reshape, [batch_size, num_ofdm_sym, fft_size, num_rx, num_rx_ant, 10, 2])  # (batch, num_ofdm_sym, fft_size, num_rx, num_rx_ant, 10, 2)
    
    x_reshape = tf.transpose(x, perm=[0,3,4,1,2]) # batch_size, num_ofdm_sym, fft_size, num_ues, num_tx_ant


    num_tx_ue = x_reshape.shape[-2]
    num_streams = x_reshape.shape[-1]
    num_tx_ant = h_reshape.shape[-1]

    assert (num_streams <= num_tx_ant) and (num_streams <= num_rx_ant), \
        "Number of stream should not exceed number of antennas"


    # Extract per-UE channels and apply SVD for each UE
    x_precoded = []
    precoding_matrices = []

    for ue in range(num_tx_ue):
        h_ue = h_reshape[..., ue, :]  # Extract UE-specific channel matrix [batch, num_rx_ue, num_rx_ant, num_tx_ant, num_ofdm_sym, fft_size]
        h_ue = tf.reshape(h_ue, [-1, num_rx_ant, num_tx_ant])

        # Compute SVD: H_ue = U * Σ * V^H
        s, u, v = tf.linalg.svd(h_ue, compute_uv=True)

        v = tf.sign(v) * v

        # Select dominant eigenmode (first column of V)
        v1 = tf.expand_dims(v[..., 0], axis=-1)  # Shape: [batch, num_ofdm_sym, fft_size, num_tx_ant, 1]

        # Apply precoding: Multiply X_ue with V1
        x_ue = tf.reshape(x_reshape[..., ue, :], [-1, 1, 1])  

        # x_ue = tf.expand_dims(x_reshape[..., ue, :], axis=-1) 
        x_ue_precoded = tf.squeeze(tf.matmul(v1, x_ue), axis=-1)  # Project onto dominant eigenmode

        x_precoded.append(tf.expand_dims(x_ue_precoded, axis=1))  # Append in new UE dimension
        precoding_matrices.append(tf.expand_dims(v1, axis=1))

    # Stack along UE axis
    x_precoded = tf.concat(x_precoded, axis=-2)  # Shape: [batch, num_ofdm_sym, fft_size, num_ues, 1]
    x_precoded = tf.reshape(x_precoded, [batch_size, num_ofdm_sym, fft_size, x_precoded.shape[-2], x_precoded.shape[-1]])
    x_precoded = tf.transpose(x_precoded, perm=[0,3,4,1,2]) # batch_size, num_ues, num_tx_ant, num_ofdm_sym, fft_size
    precoding_matrices = tf.concat(precoding_matrices, axis=-2)  # Store precoding matrices for debugging

    if return_precoding_matrix:
        return x_precoded, precoding_matrices
    else:
        return x_precoded
    
