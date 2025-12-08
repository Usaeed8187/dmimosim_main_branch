import numpy as np
import tensorflow as tf


def phase_3_sic_lmmse_decoding(y_rg, h_hat, MU_MIMO_RG_populated, rg, noise_var):

    batch_size, _, num_rx_ants, num_tx, num_tx_streams, num_ofdm_syms, num_effective_scs = h_hat.shape

    num_rbs_per_RBG, _, num_ues_per_rb = MU_MIMO_RG_populated.shape

    assert num_effective_scs % num_rbs_per_RBG == 0 # subcarriers divisible into RBs

    num_subcarriers_per_RB = num_effective_scs // num_rbs_per_RBG

    left_rbg_data_syms_idx = tf.range(rg.num_ofdm_symbols//2)
    left_rbg_data_syms_idx = left_rbg_data_syms_idx[left_rbg_data_syms_idx != rg._pilot_ofdm_symbol_indices[0]]

    right_rbg_data_syms_idx = tf.range(rg.num_ofdm_symbols//2, rg.num_ofdm_symbols)
    right_rbg_data_syms_idx = right_rbg_data_syms_idx[right_rbg_data_syms_idx != rg._pilot_ofdm_symbol_indices[1]]

    num_RBs_per_UE = np.prod(MU_MIMO_RG_populated.shape) // num_tx
    total_syms_per_UE_per_stream = num_RBs_per_UE * num_subcarriers_per_RB * len(left_rbg_data_syms_idx)

    num_syms_updated_per_RB = num_subcarriers_per_RB * len(left_rbg_data_syms_idx)

    x_hat = tf.zeros([batch_size, num_tx, num_tx_streams, total_syms_per_UE_per_stream], dtype=y_rg.dtype)
    no_eff = tf.zeros([batch_size, num_tx, num_tx_streams, total_syms_per_UE_per_stream], dtype=y_rg.dtype.real_dtype)

    ue_wise_syms_pointer = np.zeros(num_tx, dtype=int)

    for rb_idx in range(num_rbs_per_RBG):
        abs_sc_idx = tf.range(rg.num_guard_carriers[0] + rb_idx*num_subcarriers_per_RB, 
                            rg.num_guard_carriers[0] + (rb_idx+1)*num_subcarriers_per_RB)
        rel_sc_idx = tf.range(rb_idx*num_subcarriers_per_RB, 
                            (rb_idx+1)*num_subcarriers_per_RB)
        
        # Getting DMRS channel estimate
        curr_h_hat = tf.gather(h_hat, rel_sc_idx, axis=-1)
        curr_h_hat_left = tf.gather(curr_h_hat, left_rbg_data_syms_idx, axis=-2)
        curr_h_hat_right = tf.gather(curr_h_hat, right_rbg_data_syms_idx, axis=-2)

        # Getting received signal
        curr_y_rg = tf.gather(y_rg, abs_sc_idx, axis = -1)
        curr_y_rg_left = tf.gather(curr_y_rg, left_rbg_data_syms_idx, axis=-2)
        curr_y_rg_right = tf.gather(curr_y_rg, right_rbg_data_syms_idx, axis=-2)

        ############################ LEFT RBG PROCESSING ######################################
        curr_UEs_left = tf.cast(MU_MIMO_RG_populated[rb_idx, 0, :], tf.int64)
        if curr_UEs_left[0] == curr_UEs_left[1]:
            curr_UEs_left = [curr_UEs_left[0]]

        curr_h_hat_left = tf.gather(curr_h_hat_left, curr_UEs_left, axis=3)
        curr_x_hat, curr_no_eff = sic_lmmse_equalize(curr_y_rg_left, curr_h_hat_left, noise_var)
        
        assert curr_x_hat.shape == curr_no_eff.shape

        for i in range(curr_x_hat.shape[1]):
            UE_idx = curr_UEs_left[i]
            syms_idx = tf.range(ue_wise_syms_pointer[UE_idx] , ue_wise_syms_pointer[UE_idx] + num_syms_updated_per_RB )

            curr_curr_x_hat = curr_x_hat[:,i,:,:]
            curr_curr_no_eff = curr_no_eff[:,i,:,:]

            x_hat = scatter_update_single_ue(x_hat, curr_curr_x_hat, UE_idx, syms_idx)
            no_eff = scatter_update_single_ue(no_eff, curr_curr_no_eff, UE_idx, syms_idx)

            assert np.all(curr_curr_x_hat == x_hat[:,UE_idx,:,syms_idx[0]:syms_idx[0]+num_syms_updated_per_RB])

            ue_wise_syms_pointer[UE_idx] += num_syms_updated_per_RB

        ############################ RIGHT RBG PROCESSING ######################################
        curr_UEs_right = tf.cast(MU_MIMO_RG_populated[rb_idx, 1, :], tf.int64)
        if curr_UEs_right[0] == curr_UEs_right[1]:
            curr_UEs_right = [curr_UEs_right[0]]

        curr_h_hat_right = tf.gather(curr_h_hat_right, curr_UEs_right, axis=3)
        curr_x_hat, curr_no_eff = sic_lmmse_equalize(curr_y_rg_right, curr_h_hat_right, noise_var)

        assert curr_x_hat.shape == curr_no_eff.shape

        for i in range(curr_x_hat.shape[1]):
            UE_idx = curr_UEs_right[i]
            syms_idx = tf.range(ue_wise_syms_pointer[UE_idx] , ue_wise_syms_pointer[UE_idx] + num_syms_updated_per_RB )

            curr_curr_x_hat = curr_x_hat[:,i,:,:]
            curr_curr_no_eff = curr_no_eff[:,i,:,:]

            x_hat = scatter_update_single_ue(x_hat, curr_curr_x_hat, UE_idx, syms_idx)
            no_eff = scatter_update_single_ue(no_eff, curr_curr_no_eff, UE_idx, syms_idx)

            assert np.all(curr_curr_x_hat == x_hat[:,UE_idx,:,syms_idx[0]:syms_idx[0]+num_syms_updated_per_RB])

            ue_wise_syms_pointer[UE_idx] += num_syms_updated_per_RB
        
    assert np.all(ue_wise_syms_pointer == total_syms_per_UE_per_stream), "Not all UEs decoded their symbols"

    return x_hat, no_eff


def lmmse_equalize(
    curr_y_rg,                             # [Nsf, 1, Nr, T, F] complex
    curr_h_hat,                            # [Nsf, 1, Nr, Nt, S, T, F] complex
    noise_var=1e-6
):
    """
    Returns:
      x_hat:  [Nsf, Nt, S, T*F] complex equalized symbol estimates
      no_eff: [Nsf, Nt, S, T*F] real effective noise variance for demapper
    """
    # ----- reshape / fold dimensions -----
    Nsf, _, Nr, T, F = curr_y_rg.shape
    _, _, Nr2, Nt, S, T2, F2 = curr_h_hat.shape
    assert int(Nr2) == int(Nr) and int(T2) == int(T) and int(F2) == int(F)

    K = int(Nt) * int(S)  # total layers

    y = tf.squeeze(curr_y_rg, axis=1)     # [Nsf, Nr, T, F]
    H = tf.squeeze(curr_h_hat, axis=1)    # [Nsf, Nr, Nt, S, T, F]

    H = tf.reshape(H, [Nsf, Nr, K, T, F]) # [Nsf, Nr, K, T, F]
    y = tf.transpose(y, [0, 2, 3, 1])     # [Nsf, T, F, Nr]
    H = tf.transpose(H, [0, 3, 4, 1, 2])  # [Nsf, T, F, Nr, K]

    B = Nsf*T*F
    y_b = tf.reshape(y, [B, Nr])[..., tf.newaxis]   # [B, Nr, 1]
    H_b = tf.reshape(H, [B, Nr, K])                 # [B, Nr, K]

    # LMMSE inverse (G = (H^H H + σ² I)^(-1) H^H)
    Hh = tf.linalg.adjoint(H_b)                     # [B, K, Nr]
    I = tf.eye(K, dtype=H_b.dtype)
    W = tf.linalg.inv(tf.matmul(Hh, H_b) + noise_var*I)
    G = tf.matmul(W, Hh)                            # [B, K, Nr]

    # Equalize
    x_hat_b = tf.matmul(G, y_b)                     # [B, K, 1]
    x_hat_b = tf.squeeze(x_hat_b, axis=-1)          # [B, K]

    # Compute effective noise variance per layer & tone:
    # σ_eff² = σ² * sum_r |G_{k,r}|²
    G_abs2 = tf.math.real(tf.math.conj(G) * G)
    no_eff_b = noise_var * tf.reduce_sum(G_abs2, axis=2)  # [B, K]

    # reshape output back
    x_hat = tf.reshape(x_hat_b, [Nsf, T, F, Nt, S])
    x_hat = tf.transpose(x_hat, [0, 3, 4, 1, 2])           # [Nsf, Nt, S, T, F]
    x_hat = tf.reshape(x_hat, [Nsf, Nt, S, T*F])

    no_eff = tf.reshape(no_eff_b, [Nsf, T, F, Nt, S])
    no_eff = tf.transpose(no_eff, [0, 3, 4, 1, 2])         # [Nsf, Nt, S, T, F]
    no_eff = tf.reshape(no_eff, [Nsf, Nt, S, T*F])

    return x_hat, no_eff


def sic_lmmse_equalize(
    curr_y_rg,                             # [Nsf, 1, Nr, T, F] complex
    curr_h_hat,                            # [Nsf, 1, Nr, Nt, S, T, F] complex
    noise_var=1e-3                         # scalar or broadcastable to [B]
):
    """
    Returns:
      xhat:   [Nsf, Nt, S, T*F] complex, LMMSE-SIC equalized symbols
      no_eff: [Nsf, Nt, S, T*F] real, per-layer effective noise variance
              (MSE of MMSE estimate at the step that stream was decoded)
    """
    Nsf, _, Nr, T, F = curr_y_rg.shape
    _, _, Nr2, Nt, S, T2, F2 = curr_h_hat.shape
    assert int(Nr2) == int(Nr) and int(T2) == int(T) and int(F2) == int(F)

    K = int(Nt) * int(S)

    # Remove UE axis
    y = tf.squeeze(curr_y_rg, axis=1)              # [Nsf, Nr, T, F]
    H = tf.squeeze(curr_h_hat, axis=1)             # [Nsf, Nr, Nt, S, T, F]

    # Flatten (Nt,S)->K and fold (T,F) into batch
    H = tf.reshape(H, [Nsf, Nr, K, T, F])          # [Nsf, Nr, K, T, F]
    y = tf.transpose(y, [0, 2, 3, 1])              # [Nsf, T, F, Nr]
    H = tf.transpose(H, [0, 3, 4, 1, 2])           # [Nsf, T, F, Nr, K]

    B = int(Nsf) * int(T) * int(F)

    y_b = tf.reshape(y, [B, Nr])                   # [B, Nr]
    H_b = tf.reshape(H, [B, Nr, K])                # [B, Nr, K]

    dtype = H_b.dtype
    y_b   = tf.cast(y_b, dtype)
    H_b   = tf.cast(H_b, dtype)

    # Noise broadcast
    nv = tf.cast(noise_var, dtype)                 # scalar or [B]
    if tf.rank(nv) == 0:
        nv = tf.broadcast_to(nv, [B])              # [B]

    # Residual, outputs, and containers
    r = y_b[..., None]                             # [B, Nr, 1]
    xhat_b = tf.zeros([B, K], dtype=dtype)         # [B, K]
    noeff_b = tf.zeros([B, K], dtype=tf.float32)   # [B, K] real

    # Active column mask (complex for safe multiplies)
    rem_mask = tf.ones([K], dtype=dtype)           # [K], 1+0j active / 0+0j removed

    def mmse_W(Hrem, nv_):                         # Hrem: [B, Nr, Krem]  (Krem=K but masked cols are zero)
        Hh = tf.linalg.adjoint(Hrem)               # [B, Krem, Nr]
        G  = tf.matmul(Hh, Hrem)                   # [B, Krem, Krem]
        I  = tf.eye(tf.shape(G)[-1], dtype=dtype)[None, ...]
        A  = G + nv_[:, None, None] * I
        return tf.matmul(tf.linalg.inv(A), Hh), G  # W [B,Krem,Nr], G [B,Krem,Krem]

    def pick_next_idx(Hrem, active_mask_real):     # active_mask_real: [K] float32
        power = tf.reduce_sum(tf.abs(Hrem)**2, axis=1)            # [B, K]
        huge = tf.cast(1e12, power.dtype)
        power = power + (1.0 - active_mask_real)[None, :] * huge
        idx_b = tf.argmin(power, axis=1, output_type=tf.int32)    # [B]
        return tf.math.argmax(tf.math.bincount(idx_b))            # scalar idx

    for _ in range(K):
        # Apply mask (zeroed columns are effectively removed)
        Hrem = H_b * rem_mask[None, None, :]       # [B, Nr, K]
        idx = pick_next_idx(Hrem, tf.cast(tf.math.real(rem_mask), tf.float32))  # scalar

        # MMSE filter for current active set + Gram
        W, G = mmse_W(Hrem, nv)                    # W: [B, K, Nr], G: [B, K, K]

        # Nulling vector & soft symbol
        w_k = tf.gather(W, idx, axis=1)            # [B, Nr]
        z_k = tf.matmul(w_k[:, None, :], r)        # [B, 1, 1]
        s_k = tf.squeeze(z_k, axis=[1, 2])         # [B] complex

        # Cancel selected stream
        h_k = tf.gather(H_b, idx, axis=2)          # [B, Nr]
        r = r - tf.expand_dims(h_k, -1) * s_k[:, None, None]

        # Save symbol
        onehot_c = tf.one_hot(idx, K, dtype=dtype)     # [K] complex
        xhat_b = xhat_b + s_k[:, None] * onehot_c[None, :]

        # ----- compute no_eff for THIS decoded stream at this step -----
        # MSE_k = diag( (I + (Hrem^H Hrem)/N0)^(-1) )[k]
        I = tf.eye(K, dtype=dtype)[None, ...]
        A_over_N0 = G / nv[:, None, None]              # [B, K, K]
        M = tf.linalg.inv(A_over_N0 + I)               # [B, K, K]
        mse_diag = tf.math.real(tf.linalg.diag_part(M))# [B, K]
        mse_k = tf.gather(mse_diag, idx, axis=1)       # [B]

        onehot_r = tf.one_hot(idx, K, dtype=tf.float32)  # [K] real
        noeff_b = noeff_b + mse_k[:, None] * onehot_r[None, :]

        # Mark removed
        rem_mask = tf.tensor_scatter_nd_update(
            rem_mask,
            indices=tf.reshape(idx, [1, 1]),
            updates=tf.zeros([1], dtype=dtype)
        )

    # Reshape back
    xhat = tf.reshape(xhat_b, [Nsf, T, F, Nt, S])   # [Nsf, T, F, Nt, S]
    xhat = tf.transpose(xhat, [0, 3, 4, 1, 2])      # [Nsf, Nt, S, T, F]
    xhat = tf.reshape(xhat, [Nsf, Nt, S, T*F])      # [Nsf, Nt, S, T*F]

    no_eff = tf.reshape(noeff_b, [Nsf, T, F, Nt, S])
    no_eff = tf.transpose(no_eff, [0, 3, 4, 1, 2])  # [Nsf, Nt, S, T, F]
    no_eff = tf.reshape(no_eff, [Nsf, Nt, S, T*F])  # real

    return xhat, no_eff



def scatter_update_single_ue(x_hat, curr_curr_x_hat, UE_idx, syms_idx):
    """
    x_hat:            [1, U, S, F]
    curr_curr_x_hat:  [1, S, L]
    UE_idx:           scalar int (can be tf.Tensor)
    syms_idx:         [L] int tensor
    
    Returns:
        updated x_hat
    """
    UE_idx = tf.cast(UE_idx, tf.int32)
    syms_idx = tf.cast(syms_idx, tf.int32)

    B, U, S, F = x_hat.shape
    L = tf.shape(syms_idx)[0]

    # Build scatter indices for all (b=0, ue=UE_idx, stream=s, freq=k)
    b_idx = tf.zeros([S * L], dtype=tf.int32)                       # batch dim = 0
    ue_idx = tf.fill([S * L], UE_idx)                              # target UE
    s_idx = tf.repeat(tf.range(S, dtype=tf.int32), repeats=L)      # stream index
    f_idx = tf.tile(syms_idx, multiples=[S])                       # frequency indices

    scatter_indices = tf.stack([b_idx, ue_idx, s_idx, f_idx], axis=1)   # [S*L, 4]
    updates = tf.reshape(curr_curr_x_hat, [-1])                         # [S*L]

    return tf.tensor_scatter_nd_update(x_hat, scatter_indices, updates)

