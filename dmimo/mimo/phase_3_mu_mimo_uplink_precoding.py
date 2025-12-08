import numpy as np
import tensorflow as tf


def phase_3_mu_mimo_uplink_precoding(x, x_rg_placeholder, precoding_matrices, rg, MU_MIMO_RG_populated, num_subcarriers_per_RB):
    """
    MU-MIMO Uplink precoding using TPMI

    # x has shape
    # [batch_size, num_UE, Nsyms_per_UE, num_streams, 1]
    # Here Nsyms_per_UE means the total number of qam symbols the UE has to transmit
    #
    # precoding_matrices has shape
    # [num_UE, num_ofdm_syms, fft_size, num_tx_ant, num_streams]
    # 
    # MU_MIMO_RG_populated has shape
    # [num_RBs, num_ofdm_symbols / num_ofdm_syms_per_RBG, num_UEs_per_RBG]
    # 
    # 
    """

    batch_size, num_UEs, _, num_streams, _ = x.shape
    
    num_RBs = MU_MIMO_RG_populated.shape[0] # this is the number of RBs in one half of the OFDM grid
    num_RBs_per_UE = np.prod(MU_MIMO_RG_populated.shape) / x.shape[1]
    assert x.shape[2] % num_RBs_per_UE == 0  # assure that the data symbols in x are divisible into the RBGs we have

    left_rbg_data_syms_idx = tf.range(rg.num_ofdm_symbols//2)
    left_rbg_data_syms_idx = left_rbg_data_syms_idx[left_rbg_data_syms_idx != rg._pilot_ofdm_symbol_indices[0]]

    right_rbg_data_syms_idx = tf.range(rg.num_ofdm_symbols//2, rg.num_ofdm_symbols)
    right_rbg_data_syms_idx = right_rbg_data_syms_idx[right_rbg_data_syms_idx != rg._pilot_ofdm_symbol_indices[1]]
    
    num_subcarriers_per_RB = tf.cast(num_subcarriers_per_RB, tf.int64)

    qam_syms_per_RBG = num_subcarriers_per_RB * len(left_rbg_data_syms_idx)

    ue_wise_syms_pointer = np.zeros(num_UEs, dtype=int)

    # Precode pilots
    for ue_idx in range(num_UEs):
        curr_x_rg_placeholder = x_rg_placeholder[:, ue_idx, :, :, :, :]
        curr_precoding_matrices = precoding_matrices[ue_idx, ...]
        curr_precoding_matrices = curr_precoding_matrices[tf.newaxis, ...]
        curr_precoding_matrices = tf.tile(curr_precoding_matrices, [batch_size, 1, 1, 1, 1])

        curr_x_rg_placeholder_precoded = tf.matmul(curr_precoding_matrices, curr_x_rg_placeholder)

        x_rg_placeholder = tf.where(
            tf.equal(tf.range(tf.shape(x_rg_placeholder)[1])[None, :, None, None, None, None],
                    ue_idx),
            tf.expand_dims(curr_x_rg_placeholder_precoded, axis=1),
            x_rg_placeholder
        )

    # Iterate over RBs and precode data
    for rb_idx in range(num_RBs):
        sc_idx = tf.range(rg.num_guard_carriers[0] + rb_idx*num_subcarriers_per_RB, 
                            rg.num_guard_carriers[0] + (rb_idx+1)*num_subcarriers_per_RB)
        
        # Getting precoders for the current RB
        curr_precoding_matrices = tf.gather(precoding_matrices, sc_idx, axis=2)
        
        ############################ LEFT RBG PROCESSING ######################################
        curr_UEs_left = tf.cast(MU_MIMO_RG_populated[rb_idx, 0, :], tf.int64)
        if curr_UEs_left[0] == curr_UEs_left[1]:
            curr_UEs_left = [curr_UEs_left[0]]
        
        # Getting precoders for the current RBG
        curr_precoding_matrices_left = tf.gather(curr_precoding_matrices, curr_UEs_left, axis=0)
        curr_precoding_matrices_left = tf.gather(curr_precoding_matrices_left, left_rbg_data_syms_idx, axis=1)
        curr_precoding_matrices_left = curr_precoding_matrices_left[tf.newaxis, ...]
        curr_precoding_matrices_left = tf.tile(curr_precoding_matrices_left, [batch_size, 1, 1, 1, 1, 1])

        # Getting current information to precode
        curr_x = tf.gather(x, curr_UEs_left, axis=1)
        if len(curr_UEs_left) == 2:
            sym_idx_first_UE = tf.range(ue_wise_syms_pointer[curr_UEs_left][0]
                                        ,ue_wise_syms_pointer[curr_UEs_left][0] + qam_syms_per_RBG)
            sym_idx_second_UE = tf.range(ue_wise_syms_pointer[curr_UEs_left][1]
                                        ,ue_wise_syms_pointer[curr_UEs_left][1] + qam_syms_per_RBG)
            curr_x_first_UE = tf.gather(curr_x, sym_idx_first_UE, axis=2)
            curr_x_first_UE = curr_x_first_UE[:,0,...]
            curr_x_first_UE = curr_x_first_UE[:, tf.newaxis, ...]
            curr_x_second_UE = tf.gather(curr_x, sym_idx_second_UE, axis=2)
            curr_x_second_UE = curr_x_second_UE[:,1,...]
            curr_x_second_UE = curr_x_second_UE[:, tf.newaxis, ...]
            curr_x = tf.concat([curr_x_first_UE, curr_x_second_UE], axis=1)
            curr_x = tf.reshape(curr_x, [curr_x.shape[0], curr_x.shape[1], len(left_rbg_data_syms_idx), curr_precoding_matrices.shape[2], curr_x.shape[-2], curr_x.shape[-1]])
        else:
            sym_idx = tf.range(ue_wise_syms_pointer[curr_UEs_left]
                               ,ue_wise_syms_pointer[curr_UEs_left] + qam_syms_per_RBG)
            curr_x = tf.gather(curr_x, sym_idx, axis=2)
            curr_x = tf.reshape(curr_x, [curr_x.shape[0], curr_x.shape[1], len(left_rbg_data_syms_idx), curr_precoding_matrices.shape[2], curr_x.shape[-2], curr_x.shape[-1]])
        
        # Precoding data
        curr_x_precoded = tf.matmul(curr_precoding_matrices_left, curr_x)

        # assert np.count_nonzero(curr_x_precoded) == np.prod(curr_x_precoded.shape), "qam symbols across at least 1 pair of streams the same. collided with a precoder row that was [v -v]"

        # Placing precoded data into the resource grid
        x_rg_placeholder = scatter_into_rg(x_rg_placeholder, curr_x_precoded, curr_UEs_left, left_rbg_data_syms_idx, sc_idx)

        # Incrementing pointer
        ue_wise_syms_pointer[curr_UEs_left] += qam_syms_per_RBG

        ############################ RIGHT RBG PROCESSING ######################################
        curr_UEs_right = tf.cast(MU_MIMO_RG_populated[rb_idx, 1, :], tf.int64)
        if curr_UEs_right[0] == curr_UEs_right[1]:
            curr_UEs_right = [curr_UEs_right[0]]

        # Getting precoders for the current RBG
        curr_precoding_matrices_right = tf.gather(curr_precoding_matrices, curr_UEs_right, axis=0)
        curr_precoding_matrices_right = tf.gather(curr_precoding_matrices_right, right_rbg_data_syms_idx, axis=1)
        curr_precoding_matrices_right = curr_precoding_matrices_right[tf.newaxis, ...]
        curr_precoding_matrices_right = tf.tile(curr_precoding_matrices_right, [batch_size, 1, 1, 1, 1, 1])

        # Getting current information to precode
        curr_x = tf.gather(x, curr_UEs_right, axis=1)
        if len(curr_UEs_right) == 2:
            sym_idx_first_UE = tf.range(ue_wise_syms_pointer[curr_UEs_right][0]
                                        ,ue_wise_syms_pointer[curr_UEs_right][0] + qam_syms_per_RBG)
            sym_idx_second_UE = tf.range(ue_wise_syms_pointer[curr_UEs_right][1]
                                        ,ue_wise_syms_pointer[curr_UEs_right][1] + qam_syms_per_RBG)
            curr_x_first_UE = tf.gather(curr_x, sym_idx_first_UE, axis=2)
            curr_x_first_UE = curr_x_first_UE[:,0,...]
            curr_x_first_UE = curr_x_first_UE[:, tf.newaxis, ...]
            curr_x_second_UE = tf.gather(curr_x, sym_idx_second_UE, axis=2)
            curr_x_second_UE = curr_x_second_UE[:,1,...]
            curr_x_second_UE = curr_x_second_UE[:, tf.newaxis, ...]
            curr_x = tf.concat([curr_x_first_UE, curr_x_second_UE], axis=1)
            curr_x = tf.reshape(curr_x, [curr_x.shape[0], curr_x.shape[1], len(right_rbg_data_syms_idx), curr_precoding_matrices.shape[2], curr_x.shape[-2], curr_x.shape[-1]])
        else:
            sym_idx = tf.range(ue_wise_syms_pointer[curr_UEs_right]
                               ,ue_wise_syms_pointer[curr_UEs_right] + qam_syms_per_RBG)
            curr_x = tf.gather(curr_x, sym_idx, axis=2)
            curr_x = tf.reshape(curr_x, [curr_x.shape[0], curr_x.shape[1], len(right_rbg_data_syms_idx), curr_precoding_matrices.shape[2], curr_x.shape[-2], curr_x.shape[-1]])

        # Precoding data
        curr_x_precoded = tf.matmul(curr_precoding_matrices_right, curr_x)

        # Placing precoded data into the resource grid
        x_rg_placeholder = scatter_into_rg(x_rg_placeholder, curr_x_precoded, curr_UEs_right, right_rbg_data_syms_idx, sc_idx)

        # Incrementing pointer
        ue_wise_syms_pointer[curr_UEs_right] += qam_syms_per_RBG

    assert np.all(ue_wise_syms_pointer == x.shape[2]), "Not All UEs were able to put all their symbols into the RG"

    return x_rg_placeholder


def scatter_into_rg(x_rg_placeholder,
                    curr_x_precoded,
                    curr_UEs,                   # shape [2] (int32/int64)
                    rbg_data_syms_idx,          # shape [6]
                    sc_idx):                    # shape [12]

    # Ensure integer dtype for indices
    curr_UEs               = tf.cast(curr_UEs, tf.int32)
    rbg_data_syms_idx = tf.cast(rbg_data_syms_idx, tf.int32)
    sc_idx                 = tf.cast(sc_idx, tf.int32)

    # Build full index grid for all elements to be updated
    B  = tf.constant([0], dtype=tf.int32)      # batch axis (size 1)
    A4 = tf.range(2, dtype=tf.int32)           # axis 4 (size 2)
    A5 = tf.range(1, dtype=tf.int32)           # axis 5 (size 1)

    # Meshgrid over all target indices (ij indexing preserves axis order)
    Bg, Ug, Tg, Fg, Gg, Hg = tf.meshgrid(
        B, curr_UEs, rbg_data_syms_idx, sc_idx, A4, A5, indexing='ij'
    )
    idx = tf.stack([Bg, Ug, Tg, Fg, Gg, Hg], axis=-1)          # [..., 6]
    idx = tf.reshape(idx, [-1, 6])                             # [N, 6]

    # Flatten updates to match N scalar writes
    updates = tf.reshape(curr_x_precoded, [-1])

    # Scatter-update (returns a new tensor)
    x_rg_updated = tf.tensor_scatter_nd_update(
        x_rg_placeholder, idx, updates
    )
    return x_rg_updated

