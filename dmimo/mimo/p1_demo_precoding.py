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

    w_k_i = np.ones((num_user, num_user_streams))
    best_w_k_i = w_k_i.copy()
    best_SINR_k_i = np.zeros((num_user, num_user_streams))
    starting_SINR = np.zeros((num_user, num_user_streams))

    SINR_k_i = np.zeros((num_user, num_user_streams))

    weight_floor_constant = 0.1
    weight_update_constant = 0.5
    gram_constant = 0.01
    
    if num_iterations > 0:
        for iter in range(num_iterations+1):

            w_k_i_reshaped = tf.reshape(w_k_i, (num_user, num_user_streams, 1))
            w_k_i_reshaped = tf.cast(w_k_i_reshaped, h.dtype)

            # print("\niteration {}".format(iter))

            gram = tf.matmul(h * np.sqrt(w_k_i_reshaped), h * np.sqrt(w_k_i_reshaped), adjoint_a=True)
            gram_normalizer = gram_constant * np.eye(gram.shape[1])
            sum_gram = tf.reduce_sum(gram, axis=0) + tf.cast(gram_normalizer, gram.dtype)
            eigvals, v = tf.linalg.eigh(sum_gram)
            # eigvals = tf.sort(tf.abs(eigvals), direction='DESCENDING')
            v = tf.gather(v, tf.argsort(tf.abs(eigvals), direction='DESCENDING'), axis=1)[:, :num_user_streams]
            # norm = tf.sqrt(tf.reduce_sum(tf.abs(v) ** 2, axis=0, keepdims=True))
            # v = v / tf.cast(norm, v.dtype)

            prev_SINR_k_i = SINR_k_i.copy()

            for k in range(num_user):

                h_ue = tf.gather(h, indices=k, axis=0)

                H_k_v = tf.matmul(h_ue, v)

                if num_user_streams > 1:
                    SINR_k_i[k,0] = tf.abs(tf.linalg.norm(H_k_v[:, 0]))**2 / (tf.abs(tf.linalg.norm(H_k_v[:, 1]))**2 + n_var[k])
                    SINR_k_i[k,1] = tf.abs(tf.linalg.norm(H_k_v[:, 1]))**2 / (tf.abs(tf.linalg.norm(H_k_v[:, 0]))**2 + n_var[k])
                else:
                    SINR_k_i[k,0] = tf.abs(tf.linalg.norm(H_k_v))**2 / (n_var[k])

            if iter == 0:
                starting_SINR = SINR_k_i.copy()

            if np.min(SINR_k_i) < 0.0001:
                hold = 1
                SINR_k_i[np.argmin(SINR_k_i)] += 0.0001

            # print("all SINRs at step {} of precoding:\n".format(iter), SINR_k_i)
            # print("min SINR at step {} of precoding: ".format(iter), np.min(SINR_k_i))
            # print("max SINR at step {} of precoding: ".format(iter), np.max(SINR_k_i))
            # print("mean SINR at step {} of precoding: ".format(iter), np.mean(SINR_k_i))

            if np.min(SINR_k_i) > np.min(best_SINR_k_i):
                best_w_k_i = w_k_i.copy()
                best_SINR_k_i = SINR_k_i.copy()

            prev_w_k_i = w_k_i.copy()
            w_k_i = 1 / np.sqrt(SINR_k_i)
            # w_k_i = weight_update_constant*w_k_i + (1-weight_update_constant)*prev_w_k_i
            # print("w_k_i before flooring:", w_k_i)
            floor_value = weight_floor_constant * np.mean(w_k_i)
            w_k_i = np.where(w_k_i < floor_value, floor_value, w_k_i)
            # print("w_k_i after flooring:", w_k_i)
            # w_k_i /= np.sum(w_k_i)

            hold = 1

    # if num_iterations > 0:
    #     print("\nmin SINR after precoding: ", np.min(best_SINR_k_i))
    #     print("max SINR after precoding: ", np.max(best_SINR_k_i))
    #     print("mean SINR after precoding: ", np.mean(best_SINR_k_i))
    #     print("all SINRs after precoding: ", best_SINR_k_i)

    #     hold = 1

    w_k_i_reshaped = tf.reshape(best_w_k_i, (num_user, num_user_streams, 1))
    w_k_i_reshaped = tf.cast(w_k_i_reshaped, h.dtype)

    gram = tf.matmul(h * np.sqrt(w_k_i_reshaped), h * np.sqrt(w_k_i_reshaped), adjoint_a=True)
    gram_normalizer = gram_constant * np.eye(gram.shape[1])
    sum_gram = tf.reduce_sum(gram, axis=0) + tf.cast(gram_normalizer, gram.dtype)
    eigvals, v = tf.linalg.eigh(sum_gram)
    # eigvals = tf.sort(tf.abs(eigvals), direction='DESCENDING')
    v = tf.gather(v, tf.argsort(tf.abs(eigvals), direction='DESCENDING'), axis=1)[:, :num_user_streams]
    # norm = tf.sqrt(tf.reduce_sum(tf.abs(v) ** 2, axis=0, keepdims=True))
    # v = v / tf.cast(norm, v.dtype)

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(v)**2, axis=-2, keepdims=True))
    g = v/tf.cast(norm, v.dtype)
    g = tf.cast(g, x.dtype)

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g, starting_SINR, best_SINR_k_i
    else:
        return x_precoded, None, starting_SINR, best_SINR_k_i
    


def wmmse_precoder(x,                  # [B, N_t, Ns]  – symbols to precode
                   h,                  # [K, N_r, N_t] – channels
                   rx_snr_db,          # scalar or [K]  – per-UE SNR
                   num_iterations=10,
                   p_max=1.0,          # total power budget
                   return_precoding_matrix=False):  # True → also return V, per-UE rates
    """
    Multicast WMMSE precoding (max-min fairness) for K users, Ns common streams.
    """
    K, N_r, N_t = h.shape
    Ns          = x.shape[-1]
    dtype       = x.dtype
    h = tf.cast(h, x.dtype)

    # noise variance per UE
    n0 = tf.cast(1 / (10.0 ** (rx_snr_db / 10.0)), dtype)      # [K] or scalar

    ## -- initial precoder: eigenmodes of mean Gram -------------------------
    g  = tf.reduce_sum(tf.matmul(h, h, adjoint_a=True), axis=0)
    _, V0 = tf.linalg.eigh(g)
    V     = V0[:, -Ns:]                          # Nt × Ns
    V     = tf.linalg.l2_normalize(V, axis=0)
    V = tf.cast(V, x.dtype)

    for _ in range(num_iterations):
        # ---------- step 1: MMSE receivers U_k ----------------------------
        HV   = tf.matmul(h, V)                   # [K, N_r, N_s]
        I_r   = tf.eye(N_r, dtype=dtype)[None, :, :]
        I_n = n0[:, None, None] * I_r
        inv_term = tf.linalg.inv(tf.matmul(HV, HV, adjoint_b=True) + I_n)
        U = tf.matmul(HV, inv_term, adjoint_a=True) # [K, N_s, N_r]

        # ---------- step 2: error & weights W_k ---------------------------
        UHV = tf.matmul(U, HV) # [K, N_s, N_s]
        I_s   = tf.eye(Ns, dtype=dtype)[None, :, :]
        I_UHV = I_s - UHV
        N_I_UH = tf.matmul(I_n, U, adjoint_b=True)
        E_k = tf.matmul(I_UHV, I_UHV, adjoint_b=True) + tf.matmul(U, N_I_UH)

        # ----------- J(V,U,W) monitor -----------------------------------------
        # trace_k = tr(W_k E_k) for each user
        W_k = tf.linalg.inv(E_k)
        mse_k   = tf.linalg.trace(E_k)        # [K]
        J_value = tf.reduce_sum(mse_k)        # scalar

        tf.print("iter", _, ":  J =", tf.abs(J_value), summarize=-1)

        # ---------- step 3: build A and B ----------------------------------
        UH      = tf.matmul(U, h)
        A_k     = tf.matmul(UH, W_k, adjoint_a=True)
        A_k     = tf.matmul(A_k, UH)
        B_k     = tf.matmul(h, tf.matmul(U, W_k, adjoint_a=True), adjoint_a=True)

        # ---------- step 4: sum over users & solve for V -----------------
        A = tf.reduce_sum(A_k, axis=0)                      # [N_t, N_t]
        B = tf.reduce_sum(B_k, axis=0)                      # [N_t, N_s]

        # regularise A very slightly to avoid numerical singularities
        eps = tf.cast(1e-9, dtype)
        A  += eps * tf.eye(N_t, dtype=dtype)

        V_un  = tf.linalg.solve(A, B)                       # unconstrained V

        # enforce the total-power budget: ‖V‖_F^2 = p_max
        p_now = tf.reduce_sum(tf.abs(V_un)**2)
        V     = V_un * tf.cast(tf.sqrt(p_max / p_now), dtype)               # scaled V (Nt×Ns)

    # ---------- precoding --------------------------------------------------
    x = tf.expand_dims(x, -1)
    x_precoded = tf.squeeze(tf.matmul(V, x), -1)   # [B, Ns] broadcasting

    if return_precoding_matrix:
        return x_precoded, V, HV                          # HV gives per-UE gains
    return x_precoded
