# Zero-Forcing (ZF) Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv
from ..utils import complex_pinv


def sumimo_zf_precoder(x, h, return_precoding_matrix=False):
    """
    SU/MU-MIMO precoding using ZF method, treating all receiving antennas as independent ones.

    :param x: data stream symbols
    :param h: channel coefficients
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
    num_streams_per_tx = x.shape[-1]
    num_rx_ant, num_tx_ant = h.shape[-2:]
    assert num_streams_per_tx <= num_rx_ant, \
        "Number of stream should not be larger than number of Tx/Rx antennas"

    # Use only the first num_streams_per_tx antennas
    if num_streams_per_tx < num_rx_ant:
        h = h[..., :num_streams_per_tx, :]

    # Compute pseudo inverse for precoding
    g = tf.matmul(h, h, adjoint_b=True)
    g = tf.matmul(h, matrix_inv(g), adjoint_a=True)

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
    
def sumimo_zf_precoder_modified(x, h, ue_ranks, return_precoding_matrix=False):
    """
    SU/MU-MIMO precoding using ZF method, treating all receiving antennas as independent ones. 
    This function picks antennas according to the selected UEs, instead of picking the first num_streams_per_tx antennas

    :param x: data stream symbols
    :param h: channel coefficients
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
    num_streams_per_tx = x.shape[-1]
    num_rx_ant, num_tx_ant = h.shape[-2:]
    assert num_streams_per_tx <= num_rx_ant, \
        "Number of stream should not be larger than number of Tx/Rx antennas"

    # Use only the first num_streams_per_tx antennas
    if num_streams_per_tx < num_rx_ant and ue_ranks == 1:
        ants_idx = np.arange(0,h.shape[-2],2)
        h = tf.gather(h, ants_idx, axis=-2)
    elif num_streams_per_tx < num_rx_ant and ue_ranks == 2:
        h = h[..., :num_streams_per_tx, :]

    # Compute pseudo inverse for precoding
    g = tf.matmul(h, h, adjoint_b=True)
    g = tf.matmul(h, matrix_inv(g), adjoint_a=True)

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

def mumimo_zf_precoder_new_testing(x, h, ue_indices, ue_ranks, return_precoding_matrix=False):
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
    total_rx_ant, total_tx_ant = h.shape[-2:]
    num_user = len(ue_indices)
    num_user_streams = np.sum(ue_ranks)
    num_user_ant = np.sum(len(val) for val in ue_indices)
    assert num_user_streams == num_streams_per_tx, "total number of streams must match"
    assert num_user_ant == total_rx_ant, "number Rx antennas must match"
    assert (num_user_streams <= total_tx_ant) and (num_user_streams <= total_rx_ant), \
        "total number of streams must be less than total number of Tx/Rx antennas"

    if total_rx_ant == num_user_streams:
        # Compute pseudo inverse for precoding
        g = tf.matmul(h, h, adjoint_b=True)
        g = tf.matmul(h, matrix_inv(g), adjoint_a=True)
    else:
        # Rank adaptation support
        h_all = []
        for k in range(num_user):
            # Update effective channels for all users
            num_rx_ant = len(ue_indices[k])  # number of antennas for user k
            h_ue = tf.gather(h, indices=ue_indices[k], axis=-2)
            if ue_ranks[k] == num_rx_ant:
                # full rank
                h_all.append(h_ue)
            else:
                # support only one stream adaptation
                assert(ue_ranks[k] == 1)
                # Calculate combining weights
                Nr_k = int(h_ue.shape[-2])
                Ik   = tf.eye(Nr_k, dtype=h_ue.dtype)[None,None,None,None,:,:]
                
                Rk   = tf.matmul(h_ue, h_ue, adjoint_b=True) + tf.cast(1e-6, h_ue.dtype)*Ik
                eigvals, eigvecs = tf.linalg.eigh(Rk) # eigvecs: [..., Nr_k, Nr_k]
                uk_raw = eigvecs[..., -1]                   # principal eigenvector [B,1,S,K,Nr_k]

                # -------- Sign/phase continuity along frequency (axis K) --------
                # We align uk[k] to uk[k-1] so that <uk[k-1], uk[k]> becomes real and >= 0.
                # Works for complex (phase alignment) and real (sign flip) cases.

                # Split along frequency axis (K is the 4th dim: index -2)
                uk_slices = tf.unstack(uk_raw, axis=-2)     # list of K tensors, each [B,1,S,Nr_k]

                aligned = []
                u_prev = uk_slices[0]
                aligned.append(u_prev)

                for i in range(1, len(uk_slices)):
                    u_curr = uk_slices[i]

                    if u_curr.dtype.is_complex:
                        # complex phase alignment
                        dot = tf.reduce_sum(tf.math.conj(u_prev) * u_curr, axis=-1, keepdims=True)  # [B,1,S,1]
                        phase = tf.math.angle(dot)                                                  # [B,1,S,1]
                        rot = tf.complex(tf.math.cos(phase), -tf.math.sin(phase))                   # e^{-j*phase}
                        u_curr = u_curr * rot                                                       # broadcast on last dim
                        # after rotation, <u_prev, u_curr> ≈ |dot| (real, >= 0)
                    else:
                        # real sign flip
                        sign = tf.sign(tf.reduce_sum(u_prev * u_curr, axis=-1, keepdims=True) + 1e-12)
                        u_curr = u_curr * sign

                    aligned.append(u_curr)
                    u_prev = u_curr

                uk_aligned = tf.stack(aligned, axis=-2)     # [B,1,S,K,Nr_k]
                # ----------------------------------------------------------------

                # Use phase-aligned eigenvector for effective channel
                uk = uk_aligned[..., tf.newaxis]            # [B,1,S,K,Nr_k,1]

                heff_k = tf.matmul(uk, h_ue, adjoint_a=True)
                h_all.append(heff_k)
        # Combine h_eff for all users
        h_zf = tf.concat(h_all, axis=-2) # [..., num_tx_ant, num_streams_per_tx]

        U = x.shape[-1]
        Heff = h_zf # [B,1,S,K,U,Nt]
        HeffH = tf.linalg.adjoint(Heff) # [B,1,S,K,Nt,U]
        gram = tf.matmul(Heff, HeffH) # [B,1,S,K,U,U]
        Iu = tf.eye(U, dtype=gram.dtype)[None,None,None,None,:,:]

        noise_var_scalar = 1e-3
        sigma2 = tf.cast(noise_var_scalar, gram.dtype) # or estimate
        lam = tf.cast(U/total_tx_ant, gram.dtype)
        alpha = lam * sigma2

        # pick alpha (see next point)
        A = gram + alpha * Iu # [B,1,S,K,U,U]
        L = tf.linalg.cholesky(A) # batched Cholesky
        # Solve A X = Heff (note: shapes must align on last two dims)
        g = tf.linalg.cholesky_solve(L, Heff) # [B,1,S,K,U,Nt]
        g = tf.linalg.adjoint(g) # [B,1,S,K,Nt,U]

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

def mumimo_zf_precoder(x, h, ue_indices, ue_ranks, return_precoding_matrix=False):
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
    total_rx_ant, total_tx_ant = h.shape[-2:]
    num_user = len(ue_indices)
    num_user_streams = np.sum(ue_ranks)
    num_user_ant = np.sum(len(val) for val in ue_indices)
    assert num_user_streams == num_streams_per_tx, "total number of streams must match"
    assert num_user_ant == total_rx_ant, "number Rx antennas must match"
    assert (num_user_streams <= total_tx_ant) and (num_user_streams <= total_rx_ant), \
        "total number of streams must be less than total number of Tx/Rx antennas"
    

    arr_indices = tf.reshape(tf.range(num_user_ant), (-1, 2))

    if total_rx_ant == num_user_streams:
        # Compute pseudo inverse for precoding
        g = tf.matmul(h, h, adjoint_b=True)
        g = tf.matmul(h, matrix_inv(g), adjoint_a=True)
    else:
        # Rank adaptation support
        h_all = []
        for k in range(num_user):
            # Update effective channels for all users
            num_rx_ant = len(arr_indices[k])  # number of antennas for user k
            h_ue = tf.gather(h, indices=arr_indices[k], axis=-2)
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


def mumimo_zf_precoder_quantized(x, h_quantized, ue_indices, ue_ranks, return_precoding_matrix=False):
    """
    MU-MIMO zero-forcing precoding supporting rank adaptation.

    :param x: data stream symbols of shape [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    :param h_quantized: quantized channel coefficients of shape [batch_size, num_tx_ants, num_streams, num_ofdm_symbols, num_subcarriers]
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
    
    

    arr_indices = tf.reshape(tf.range(num_user_ant), (-1, 2))

    h_zf = tf.transpose(h_quantized, perm=[0,3,4,1,2])  # [B, num_ofdm_symbols, num_subcarriers, num_streams, num_tx_ants]

    # Compute pseudo inverse for precoding
    # The following lines are commented out to handle singular matrix cases
    # g = tf.matmul(h_zf, h_zf, adjoint_b=True) # [B, num_ofdm_symbols, num_subcarriers, num_streams, num_streams]
    # g = tf.matmul(h_zf, matrix_inv(g), adjoint_a=True) # [B, num_ofdm_symbols, num_subcarriers, num_tx_ants, num_streams]
    # Handle cases that are even singular
    g = complex_pinv(h_zf)  # [B, num_ofdm_symbols, num_subcarriers, num_tx_ants, num_streams]

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    g = g/tf.cast(norm, g.dtype)
    g = tf.cast(g, x.dtype)
    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g
    else:
        return x_precoded
    

def mumimo_zf_precoder_quantized_new(x, h_quantized, ue_indices, ue_ranks, return_precoding_matrix=False):
    """
    MU-MIMO zero-forcing precoding supporting rank adaptation.

    :param x: data stream symbols of shape [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    :param h_quantized: quantized channel coefficients of shape [batch_size, num_streams, num_tx_ants, num_ofdm_symbols, num_subcarriers]
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

    h_zf = tf.transpose(h_quantized, perm=[0,3,4,2,1])  # [B, num_ofdm_symbols, num_subcarriers, num_tx_ants, num_streams]

    reg = 0.0
    gram = tf.matmul(h_zf, h_zf, adjoint_a=True)  # W^H W: [B,Nsym,Nsc,Ns,Ns]
    I = tf.eye(h_zf.shape[-1], batch_shape=tf.shape(gram)[:-2], dtype=gram.dtype)
    gram_inv = tf.linalg.inv(gram + tf.cast(reg, gram.dtype) * I)
    g = tf.matmul(h_zf, gram_inv)  # [B,Nsym,Nsc,Nt,Ns]
    g = g[tf.newaxis, ...]

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    g = g/tf.cast(norm, g.dtype)
    g = tf.cast(g, x.dtype)

    if reg == 0.0:
        WG = tf.matmul(h_zf, g, adjoint_a=True)
        I = tf.eye(WG.shape[-1], dtype=WG.dtype)
        off_diag_idx = 1 - I
        off_diag_vals = WG * off_diag_idx
        max_off_diag = tf.reduce_max(tf.abs(off_diag_vals))
        tf.debugging.assert_less(max_off_diag, tf.cast(1e-3, max_off_diag.dtype), message="orthogonalization failed")

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g
    else:
        return x_precoded