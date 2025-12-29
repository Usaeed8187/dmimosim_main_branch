# SLNR precoding for dMIMO channels
import numpy as np
import tensorflow as tf


def mumimo_slnr_precoder(x, h, no, ue_indices, return_precoding_matrix=False):
    """
    MU-MIMO precoding based on signal-to-leakage-noise-ratio (SLNR) criterion

    :param x: data stream symbols
    :param h: channel coefficients
    :param no: noise variance
    :param ue_indices: receiver antenna indices for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rxs_ant, num_txs_ant]

    num_streams_per_tx = x.shape[-1]
    total_rx_ant, num_tx_ant = h.shape[-2:]
    assert total_rx_ant >= num_streams_per_tx, "inputs with incorrect dimensions"
    num_user = len(ue_indices)

    F_all = []
    for k in range(num_user):
        # number of antennas for user k
        num_rx_ant = len(ue_indices[k])
        # antenna indices for users other than k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), ue_indices[k], axis=0)
        # effective channel for user k
        H_k = tf.gather(h, indices=ue_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
        # complement channels to user k
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        # compute inputs to the SLNR algorithm
        A_k = tf.matmul(tf.linalg.adjoint(H_k), H_k)  # [..., num_tx_ant, num_tx_ant]
        scaled_sigma = tf.linalg.diag((total_rx_ant/num_rx_ant * no) * tf.ones((num_tx_ant), dtype=tf.complex64))
        scaled_sigma = tf.reshape(scaled_sigma, (1, 1, 1, 1, *A_k.shape[-2:]))
        C_k = A_k + scaled_sigma + tf.matmul(tf.linalg.adjoint(H_t), H_t)  # [..., num_tx_ant, num_tx_ant]
        # step 1: compute Cholesky decomposition on C_k and obtain Q_k
        G_k = tf.linalg.cholesky(C_k)  # [..., num_tx_ant, num_tx_ant]
        Q_k = tf.linalg.adjoint(tf.linalg.inv(G_k))
        # step 2: compute eigen-decomposition on A_p and obtain U_k
        A_p = tf.matmul(tf.matmul(tf.linalg.adjoint(Q_k), A_k), Q_k)  # [..., num_tx_ant, num_tx_ant]
        s_k, u_k, v_k = tf.linalg.svd(A_p)
        # make the signs of eigen vectors consistent
        v_sign = tf.linalg.diag(tf.sign(v_k[..., 0, :]))
        u_k = tf.matmul(u_k, v_sign)
        # Step 3: compute P_k
        P_k = tf.matmul(Q_k, u_k)  # [..., num_tx_ant, num_tx_ant]
        F_k = P_k[..., :, :num_rx_ant]  # [..., num_tx_ant, num_rx_ant]
        # normalization to unit power
        norm = tf.sqrt(tf.reduce_sum(tf.abs(F_k)**2, axis=-2, keepdims=True))
        F_k = F_k/tf.cast(num_rx_ant*norm, F_k.dtype)
        # save for current user
        F_all.append(F_k)

    # combine precoding vectors for all users
    F_all = tf.concat(F_all, axis=-1)  # [..., num_tx_ant, num_streams_per_tx]

    # Precoding
    x_precoded = tf.expand_dims(x, -1)  # expand last dim of `x` for precoding
    x_precoded = tf.squeeze(tf.matmul(F_all, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, F_all
    else:
        return x_precoded

def mumimo_slnr_precoder_quantized(x, w, rx_snr_db, ue_indices, ue_ranks, return_precoding_matrix=False):

    """
    MU-MIMO precoding based on signal-to-leakage-noise-ratio (SLNR) criterion

    :param x: data stream symbols of shape [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    :param w: quantized channel coefficients of shape [batch_size, num_streams, num_tx_ants, num_ofdm_symbols, num_subcarriers]
    :param rx_snr_db: noise power [num_rx_ues]
    :param ue_indices: receiver antenna indices for all users of shape [num_users, list of antenna indices]
    :param ue_ranks: number of streams (ranks) for all users of shape [num_users]
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_streams, num_tx_ants, num_ofdm_symbols, num_subcarriers]
    num_streams_per_tx = x.shape[-1]
    num_user = len(ue_indices)
    num_user_streams = np.sum(ue_ranks)
    num_user_ant = np.sum(len(val) for val in ue_indices)
    assert num_user_streams == num_streams_per_tx, "total number of streams must match"

    W = tf.transpose(w, perm=[0,3,4,2,1])  # [B, num_ofdm_symbols, num_subcarriers, num_tx_ants, num_streams]

    B = tf.shape(W)[0]
    Nsym = tf.shape(W)[1]
    Nsc = tf.shape(W)[2]
    Nt = tf.shape(W)[3]
    Ns = tf.shape(W)[4]

    # --- Build per-stream noise from per-UE rx_snr_db ---
    rx_snr_db = tf.convert_to_tensor(rx_snr_db)
    rx_snr_db = tf.cast(rx_snr_db, tf.float32)  # keep in real for pow()

    # no_ue[k] = 10^(-SNR_dB/10)
    no_ue = tf.pow(10.0, -rx_snr_db / 10.0)  # [K], float32

    # Expand noise per stream according to ue_ranks
    ue_ranks_tf = tf.constant(np.array(ue_ranks, dtype=np.int32))
    no_stream = tf.repeat(no_ue, repeats=ue_ranks_tf)  # [Ns], float32
    # Cast to complex dtype of W for later math
    no_stream_c = tf.cast(no_stream, W.dtype)          # [Ns], complex

    # Identity
    I_Nt = tf.eye(Nt, batch_shape=[B, Nsym, Nsc], dtype=W.dtype)  # [B,Nsym,Nsc,Nt,Nt]

    # Signal covariance
    Rsig = tf.matmul(W, W, adjoint_b=True)  # [B,Nsym,Nsc,Nt,Nt]

    # Optional (recommended): scale noise relative to W power so rx_snr_db acts like SNR
    avg_sig = tf.math.real(tf.linalg.trace(Rsig)) / tf.cast(Nt, tf.float32)  # [B,Nsym,Nsc]
    avg_sig_c = tf.cast(avg_sig, W.dtype)                                     # [B,Nsym,Nsc] complex

    # Build SLNR precoder stream-by-stream.
    # (Nt is small in most NR codebook configs, so this is usually fine.)
    g_cols = []
    for i in range(num_streams_per_tx):
        w_i = W[..., i:i+1]  # [B,Nsym,Nsc,Nt,1]

        # Per-stream noise for this stream's UE
        # shape-broadcast to [B,Nsym,Nsc,1,1]
        no_i = no_stream_c[i] * avg_sig_c
        no_i = no_i[..., tf.newaxis, tf.newaxis]

        # R_i = (sum_{j≠i} w_j w_j^H) + no_i I
        # Use Rsig - w_i w_i^H to remove self-leakage term
        R_i = (Rsig - tf.matmul(w_i, w_i, adjoint_b=True)) + no_i * I_Nt  # [B,Nsym,Nsc,Nt,Nt]

        # Solve R_i * g_i = w_i  (more stable than explicit inverse)
        g_i = tf.linalg.solve(R_i, w_i)  # [B,Nsym,Nsc,Nt,1]

        # Normalize each beam to unit norm (per tone, per OFDM symbol)
        g_i_norm = tf.sqrt(tf.reduce_sum(tf.abs(g_i) ** 2, axis=-2, keepdims=True))  # [...,1,1]
        g_i = g_i / tf.cast(tf.maximum(g_i_norm, tf.cast(1e-12, g_i_norm.dtype)), g_i.dtype)

        g_cols.append(g_i)

    # Stack beams into G: [B,Nsym,Nsc,Nt,Ns]
    G = tf.concat(g_cols, axis=-1)
    G = G[tf.newaxis, ...]

    # Expand last dim of `x` for precoding: x_vec [B,num_tx,Nsym,Nsc,Ns,1]
    x_vec = tf.expand_dims(x, -1)

    # Apply precoder: (Nt x Ns) @ (Ns x 1) -> (Nt x 1)
    # G is [B,Nsym,Nsc,Nt,Ns] and x is [B,num_tx,Nsym,Nsc,Ns,1]
    x_precoded = tf.matmul(G, x_vec)  # [B,num_tx,Nsym,Nsc,Nt,1] if num_tx==1? see note below
    x_precoded = tf.squeeze(x_precoded, -1)

    if return_precoding_matrix:
        return x_precoded, G
    return x_precoded


def mumimo_slnr_equalizer(y, h, no, ue_indices):
    """
    MU-MIMO precoding based on signal-to-leakage-noise-ratio (SLNR) criterion

    :param y: data stream symbols
    :param h: channel coefficients
    :param no: noise variance
    :param ue_indices: receiver antenna indices for all users
    :return: precoded data symbols
    """

    # Input dimensions:
    # y: [batch_size, num_rx, num_ofdm_sym, fft_size, num_rx_ant/num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rxs_ant, num_txs_ant]

    total_rx_ant, num_tx_ant = h.shape[-2:]
    num_user = len(ue_indices)

    G_all = []
    for k in range(num_user):
        # number of antennas for user k
        num_rx_ant = len(ue_indices[k])
        # antenna indices for users other than k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), ue_indices[k], axis=0)
        # effective channel for user k
        H_k = tf.gather(h, indices=ue_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
        # complement channels to user k
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        # compute inputs to the SLNR algorithm
        A_k = tf.matmul(tf.linalg.adjoint(H_k), H_k)  # [..., num_tx_ant, num_tx_ant]
        scaled_sigma = tf.linalg.diag((total_rx_ant/num_rx_ant * no) * tf.ones((num_tx_ant), dtype=tf.complex64))
        scaled_sigma = tf.reshape(scaled_sigma, (1, 1, 1, 1, *A_k.shape[-2:]))
        C_k = A_k + scaled_sigma + tf.matmul(tf.linalg.adjoint(H_t), H_t)  # [..., num_tx_ant, num_tx_ant]
        # step 1: compute Cholesky decomposition on C_k and obtain Q_k
        G_k = tf.linalg.cholesky(C_k)  # [..., num_tx_ant, num_tx_ant]
        Q_k = tf.linalg.adjoint(tf.linalg.inv(G_k))
        # step 2: compute eigen-decomposition on A_p and obtain U_k
        A_p = tf.matmul(tf.matmul(tf.linalg.adjoint(Q_k), A_k), Q_k)  # [..., num_tx_ant, num_tx_ant]
        s_k, u_k, v_k = tf.linalg.svd(A_p)
        # make the signs of eigen vectors consistent
        v_sign = tf.linalg.diag(tf.sign(v_k[..., 0, :]))  # [..., num_tx_ant, num_tx_ant]
        u_k = tf.matmul(u_k, v_sign)
        # Step 3: compute P_k
        P_k = tf.matmul(Q_k, u_k)  # [..., num_tx_ant, num_tx_ant]
        F_k = P_k[..., :, :num_rx_ant]  # [..., num_tx_ant, num_rx_ant]
        # normalization to unit power
        norm = tf.sqrt(tf.reduce_sum(tf.abs(F_k)**2, axis=[-2, -1], keepdims=True))
        F_k = F_k/tf.cast(norm, F_k.dtype)
        # compute equalizer matrix
        G_k = tf.linalg.adjoint(tf.matmul(H_k, F_k))  # [..., num_rx_ant, num_rx_ant]
        G_all.append(G_k)

    # Expand last dim of `y` for equalization
    y_equalized = tf.expand_dims(y, -1)

    # Equalizing
    y_equalized = [tf.squeeze(tf.matmul(G_all[k], y_equalized[:, k:k+1]), -1) for k in range(num_user)]
    y_equalized = tf.concat(y_equalized, axis=1)

    return y_equalized
