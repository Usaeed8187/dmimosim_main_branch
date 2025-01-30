"""
Orthogonal Space-time Block Codes (OSTBC)
"""

import tensorflow as tf
import numpy as np
import sympy as sp


class OSTBC:
    
    def __init__(self, scheme: str = None, S_matrix: sp.matrices.dense.MutableDenseMatrix = None) -> None:
        """
        A class that handles orthogonal space-time block codes (OSTBC) -- Incomplete. DO NOT USE.

        Arguments
        ---------

            ``scheme``:
            Currently supported schemes are 'alamouti', '434' and '848'. If None, the user has to provide the STBC
             characteristic matrix ``S_matrix``.

            ``S_matrix`` : The STBC characteristic matrix. This is a T by Nt matrix where each row represents
              a time slot and each column represents a transmitter antenna

        Attributes
        ----------

            ``S_transpose_S``: The result of multiplying the transpose of a row of S by another row of S.
            The result is a numpy array of shape (T,T,Nt,Nt)

            ``S_hermitian_S``: The result of multiplying the hermitian of a row of S by another row of S.
            The result is a numpy array of shape (T,T,Nt,Nt)
        """
        
        assert ((scheme is not None) and (S_matrix is None)), 'User input S_matrix is not supported yet.'
        if scheme == 'alamouti':
            s1, s2 = sp.symbols('s_1,s_2')
            s1_ = sp.conjugate(s1)
            s2_ = sp.conjugate(s2)
            S = sp.Matrix([[s1, s2], [-s2_, s1_]])
            self.info_syms = [s1, s2]
        elif scheme == '434':
            s1, s2, s3 = sp.symbols('s_1,s_2,s_3')
            s1_ = sp.conjugate(s1)
            s2_ = sp.conjugate(s2)
            s3_ = sp.conjugate(s3)
            S = sp.Matrix([[s1, s2, s3, 0],
                           [-s2_, s1_, 0, s3],
                           [-s3_, 0, s1_, -s2],
                           [0, -s3_, s2_, s1]])
            self.info_syms = [s1, s2, s3]
        elif scheme == '848':
            s1, s2, s3, s4 = sp.symbols('s_1,s_2,s_3,s_4')
            s1_ = sp.conjugate(s1)
            s2_ = sp.conjugate(s2)
            s3_ = sp.conjugate(s3)
            s4_ = sp.conjugate(s4)
            S = sp.Matrix([[s1, s2, s3, 0, s4, 0, 0, 0],
                           [-s2_, s1_, 0, s3, 0, s4, 0, 0],
                           [s3_, 0, -s1_, s2, 0, 0, s4, 0],
                           [0, s3_, -s2_, -s1, 0, 0, 0, s4],
                           [s4_, 0, 0, 0, -s1_, s2, -s3, 0],
                           [0, s4_, 0, 0, -s2_, -s1, 0, -s3],
                           [0, 0, s4_, 0, -s3_, 0, s1, s2],
                           [0, 0, 0, s4_, 0, -s3_, -s2_, s1_]])
            self.info_syms = [s1, s2, s3, s4]
        else:
            raise ValueError('scheme should be either alamouti, 434 or 848')
        A = sp.conjugate(S.T) * S
        assert A == sp.diag(sp.diag(A)), 'This is not an orthogonal STBC...'
        self.S = S
        self.T = S.shape[0]  # Number of STBC time slots
        self.Nt = S.shape[1]  # Number of STBC transmit antennas
        # Distinct symbols in the matrix (S.atoms() returns all the distinct elements of S)
        info_syms = ([element for element in S.atoms() if type(element) == sp.core.symbol.Symbol])
        self.num_STBC_info_syms = len(info_syms)

        self.S_transpose_S = np.zeros([S.shape[0], S.shape[0], S.shape[1], S.shape[1]])
        for i_row in range(S.shape[0]):
            for j_row in range(S.shape[0]):
                A = S[j_row, :].T * S[i_row, :]
                self.S_transpose_S[i_row, j_row] = np.array(self._take_expectation(A))
                pass
            pass
        self.S_hermitian_S = np.zeros([S.shape[0], S.shape[0], S.shape[1], S.shape[1]])
        for i_row in range(S.shape[0]):
            for j_row in range(S.shape[0]):
                A = sp.conjugate(S[j_row, :].T) * S[i_row, :]
                self.S_hermitian_S[i_row, j_row] = np.array(self._take_expectation(A)).astype(int)
                pass
            pass

        # Here, we will figure out the receiver processing
        S_array = np.array(S)
        self.which_index_h = np.zeros([len(self.info_syms), S_array.shape[0]],
                                      dtype=int)  # As large as the number of time slots
        self.conjugate_h = np.zeros([len(self.info_syms), S_array.shape[0]],
                                    dtype=bool)  # Same shape as h_indices_processed
        self.negate_h = np.zeros([len(self.info_syms), S_array.shape[0]],
                                 dtype=bool)  # Same shape as h_indices_processed

        for i_symbol, symbol in enumerate(self.info_syms):
            # h_tilde, a vector that should get multiplied by the vector [r,r.conjugate] to give z.
            h_indices_processed = np.zeros(S_array.shape[0], dtype=int)  # As large as the number of time slots
            conjugation = np.zeros(S_array.shape[0], dtype=bool)  # Same shape as h_indices_processed
            negation = np.zeros(S_array.shape[0], dtype=bool)  # Same shape as h_indices_processed
            for conjugate_flag in [False, True]:
                if conjugate_flag:
                    s = sp.conjugate(symbol)
                else:
                    s = symbol
                for i_time in range(S_array.shape[0]):
                    for i_antenna in range(S_array.shape[1]):
                        if S_array[i_time, i_antenna] == s:
                            h_indices_processed[i_time] = i_antenna
                            conjugation[i_time] = (not conjugate_flag)  # Whatever conjugation on r, the reverse on h.
                        elif S_array[i_time, i_antenna] == -s:
                            h_indices_processed[i_time] = i_antenna
                            conjugation[i_time] = (not conjugate_flag)
                            negation[i_time] = True
            self.which_index_h[i_symbol] = h_indices_processed
            self.conjugate_h[i_symbol] = conjugation
            self.negate_h[i_symbol] = negation
        self.conjugate_r = ~self.conjugate_h  # Whenever h is conjugated, r is not and vice versa

    def _take_expectation(self, A: sp.matrices.dense.MutableDenseMatrix):
        E_A = sp.zeros(*A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                a = A[i, j]
                if a != 0:
                    assert a.func == sp.core.mul.Mul or a.func == sp.core.power.Pow
                    if a.func == sp.core.mul.Mul:  # If it is power, the expectation should be zero
                        first_multiplier = a.args[0]
                        second_multiplier = a.args[1]
                        conj_flag_1 = conj_flag_2 = False
                        if type(first_multiplier) == sp.conjugate:
                            conj_flag_1 = True
                            first_multiplier = sp.conjugate(first_multiplier)
                        if type(second_multiplier) == sp.conjugate:
                            conj_flag_2 = True
                            second_multiplier = sp.conjugate(second_multiplier)
                        if conj_flag_1 == conj_flag_2:
                            E_A[i, j] = 0
                        else:
                            if first_multiplier != second_multiplier:
                                E_A[i, j] = 0
                            else:
                                E_A[i, j] = 1
        return E_A

    def find_interference_power(self, h_intended: tf.Tensor, h_not_intended: tf.Tensor):
        """
        Computes the interference power of one cluster on the signal coming from
        the other cluster.

        WARNING: This function assumes that the intended signal and the interference
        are both using the same STBC scheme. Different schemes should be supported
        in the future.

        Arguments
        ---------

            ``h_intended``: A tensor of shape (...,Mr, Nt) where Mr and Nt is
            the number of receive and transmit antennas. This is the channel
            from the intended cluster to the receiver

            ``h_not_intended``: A tensor of shape (...,Mr, Nt) where Mr and Nt is
            the number of receive and transmit antennas. This is the channel
            from the interfering cluster to the receiver.

        Output
        ------

            ``per_symbol_intf_power``: a tensor of size (..., self.num_STBC_info_syms)
            which represents the interference power for each info symbol

        """
        assert h_intended.shape[-1] == h_not_intended.shape[-1] == self.Nt
        assert h_intended.shape[-2] == h_not_intended.shape[-2]
        # Create h_tilde
        h_tilde_all = []
        for i_sym in range(self.num_STBC_info_syms):
            h_tilde = []
            for t in range(self.T):
                if self.conjugate_r[i_sym, t]:
                    h_tilde.append(tf.zeros_like(h_intended[..., 0]))
                elif self.conjugate_h[i_sym, t]:
                    conjugate_or_not = (lambda x: tf.math.conj(x)) if self.conjugate_h[i_sym, t] else (lambda x: x)
                    h_tilde.append(conjugate_or_not(h_intended[..., self.which_index_h[i_sym, t]]))
            for t in range(self.T):
                if self.conjugate_r[i_sym, t]:
                    conjugate_or_not = (lambda x: tf.math.conj(x)) if self.conjugate_h[i_sym, t] else (lambda x: x)
                    h_tilde.append(conjugate_or_not(h_intended[..., self.which_index_h[i_sym, t]]))
                elif self.conjugate_h[i_sym, t]:
                    h_tilde.append(tf.zeros_like(h_intended[..., 0]))

            h_tilde_all.append(tf.stack(h_tilde, axis=-1))  # tf.stack(h_tilde,axis=-1): shape = (...,Mr,2T)
            pass
        del h_tilde
        h_tilde_all = tf.stack(h_tilde_all, axis=-2)  # shape = (..., Mr, num_info_syms, 2T)
        assert h_tilde_all.shape[-1] == 2 * self.T

        # Figure out 𝔼{r rᴴ} for the interference r
        shape = [self.T, self.T, *[1 for _ in range(h_intended.ndim - 1)], self.Nt, self.Nt]
        S_hermitian_S = tf.reshape(tf.convert_to_tensor(self.S_hermitian_S, dtype=h_intended.dtype), shape)
        S_transpose_S = tf.reshape(tf.convert_to_tensor(self.S_transpose_S, dtype=h_intended.dtype), shape)

        W = (tf.math.conj(h_not_intended[..., tf.newaxis, :]) @
             S_hermitian_S @
             (h_not_intended[..., :, tf.newaxis]))  # (T,T,...,1,1)
        # W = tf.transpose()
        W_hat = ((h_not_intended[..., tf.newaxis, :]) @
                 S_transpose_S @
                 (h_not_intended[..., :, tf.newaxis]))  # (T,T,...,1,1)
        Q = tf.concat([tf.concat((W, W_hat), axis=1),
                       tf.concat((tf.math.conj(W_hat), tf.math.conj(W)), axis=1)],
                      axis=0)

        Q = tf.transpose(Q, (*range(2, Q.ndim), 0, 1))  # (...,2T,2T)

        # Interference Power
        per_symbol_intf_power = h_tilde_all[..., tf.newaxis, tf.newaxis, :] @ Q @ tf.math.conj(
            h_tilde_all[..., tf.newaxis, :, tf.newaxis])
        per_symbol_intf_power = per_symbol_intf_power[..., 0, 0, 0]  # (...,Mr,num_info_syms)
        per_symbol_intf_power = tf.reduce_sum(per_symbol_intf_power, axis=-2)  # (...,num_info_syms)

        return per_symbol_intf_power
