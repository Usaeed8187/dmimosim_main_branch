import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from sionna.mapping import Constellation


class HardLogLikelihood(Layer):
    """
    This function computes log2 P(``received_symbol`` | ``candidate_symbol``)
    for all possible choices of ``candidate_symbol`` when the received
    symbol is coming from detection of a symbol passed over an AWGN channel with
    SNR ``SNR``.
    If ``sum_over_Rx_Syms`` is True, the summation of the log2 of probabilities will
    be calculated and the ``candidate_symbol`` that has the most sum log likelihood
    will be returned.


    :param tf.Tensor received_symbol: the received detected symbol of shape (...,M,N),
        where N is the number of copies of the same symbol.
    :param tf.Tensor SNR: The signal-to-noise power ratio. Has to have the same shape as ``received_symbol``.
    :param int k_constellation: The number of bits per symbol of the QAM symbol constellation
    :param bool return_sum: whether to sum over the log likelihoods over the last dimension of ``received_symbol``
    :return most_likely_symbols:
        if `sum_over_Rx_Syms==True` and `return_bit_LLRs==False` the summation of the log2 of probabilities will
        be calculated and the ``candidate_symbol`` that has the most sum log likelihood
        will be returned. The shape would be (...,M)
    :return log_prob:
        if `sum_over_Rx_Syms==False` and `return_bit_LLRs==False` the log2 of the probability of
        receiving ``received_symbol`` given, we had sent ``candidate_symbol`` through an AWGN channel
        for all possible choices of candidate symbols is returned. The shape would be (...,M,N,2**k)
    :return LLR:
        if `return_bit_LLRs==True` then the loglikelihood of each bit would be returned.
        The shape would be  (...,M,k_constellation)
    """

    def __init__(self, constel: Constellation, signal_dims,
                 sum_over_rx_syms=True, return_bit_llrs=False, max_num_copies = 11, **kwargs):

        super().__init__(trainable=False, **kwargs)

        # constel = Constellation('qam',k_constellation)
        self.candidate_symbols = constel.points
        self.k_constellation = constel.num_bits_per_symbol
        self.sum_over_rx_syms = sum_over_rx_syms
        self.return_bit_llrs = return_bit_llrs

        valid_points = tf.reshape(self.candidate_symbols, [*[1 for _ in range(signal_dims)], -1])  # shape= [1,...,1,2**k]
        self.candidate_symbols_reshaped = valid_points[..., tf.newaxis]  # (1,...,1,2**k,1)

        candidate_real = tf.sort(tf.unique((tf.math.real(self.candidate_symbols)))[0])  # (2**(k/2),)
        candidate_imag = tf.sort(tf.unique((tf.math.imag(self.candidate_symbols)))[0])  # (2**(k/2),)
        self.candidate_real_reshaped = tf.constant(tf.reshape(candidate_real, [1, -1]))
        self.candidate_imag_reshaped = tf.constant(tf.reshape(candidate_imag, [1, -1]))

        middle_points_real = (candidate_real[..., 1:] + candidate_real[..., :-1]) / 2.0  # (2**(k/2)-1,)
        middle_points_imag = (candidate_imag[..., 1:] + candidate_imag[..., :-1]) / 2.0  # (2**(k/2)-1,)

        self.left_inf_real = tf.concat(([tf.float32.min], middle_points_real), axis=0)  # shape [2**(k/2),]
        self.left_inf_imag = tf.concat(([tf.float32.min], middle_points_imag), axis=0)  # shape [2**(k/2),]
        self.right_inf_real = tf.concat((middle_points_real, [tf.float32.max]), axis=0)  # shape [2**(k/2),]
        self.right_inf_imag = tf.concat((middle_points_imag, [tf.float32.max]), axis=0)  # shape [2**(k/2),]
        self.EPS = (tf.experimental.numpy.finfo(tf.float32.as_numpy_dtype).tiny)#**(1/max_num_copies)

    # tf.where does not support XLA
    @tf.function(jit_compile=True)  # Enable graph execution to speed things up
    def call(self, inputs, SNR) -> tf.Tensor:

        assert inputs.shape == SNR.shape

        # assert tf.reduce_all(tf.reduce_any((received_symbol[...,tf.newaxis] == valid_points),axis=-1)), \
        #    'The received symbol should be part of the QAM constellation with unit average power'

        received_symbols = inputs[..., tf.newaxis, tf.newaxis]  # (...,N,1,1)
        SNRs = tf.math.real(SNR[..., tf.newaxis, tf.newaxis])  # (...,N,1,1)

        # Mapping each received symbol to an interval.
        # # def fn_v2(x: tf.Tensor):
        # x_reshaped = tf.reshape(received_symbols, [-1, 1])
        # # Explanation: the first column of tf.where's output is just a range from 0 to total_symbols. We don't need it.
        # real_indices = tf.where(tf.math.real(x_reshaped) == self.candidate_real_reshaped)[:, 1]  # shape = (total_symbols,)
        # imag_indices = tf.where(tf.math.imag(x_reshaped) == self.candidate_imag_reshaped)[:, 1]  # shape = (total_symbols,)


        # # print(real_indices.shape)
        # # print(imag_indices.shape)

        # intervals_real = tf.stack([tf.gather(self.left_inf_real, real_indices),
        #                            tf.gather(self.right_inf_real, real_indices)],
        #                           axis=1)  # shape = (total_symbols,2)
        # intervals_imag = tf.stack([tf.gather(self.left_inf_imag, imag_indices),
        #                            tf.gather(self.right_inf_imag, imag_indices)],
        #                           axis=1)  # shape = (total_symbols,2)

        # intervals = tf.concat((intervals_real, intervals_imag), axis=1)
        
        x_reshaped = tf.reshape(received_symbols, [-1, 1])
        # print('x_reshaped.shape=',x_reshaped.shape)
        mask_real_lower = (tf.math.real(x_reshaped)<self.right_inf_real[tf.newaxis,:]) # (total_symbols,2**(k/2))
        mask_real_upper = (tf.math.real(x_reshaped)>=self.left_inf_real[tf.newaxis,:]) # (total_symbols,2**(k/2))
        mask_real = tf.logical_and(mask_real_lower,mask_real_upper) # (total_symbols,2**(k/2)) -- This should have exactly one True value per row
        indices_real = tf.argmax(mask_real,axis=1) #  (total_symbols,)
        mask_imag_lower = (tf.math.imag(x_reshaped)<self.right_inf_imag[tf.newaxis,:]) # (total_symbols,2**(k/2))
        mask_imag_upper = (tf.math.imag(x_reshaped)>=self.left_inf_imag[tf.newaxis,:]) # (total_symbols,2**(k/2))
        mask_imag = tf.logical_and(mask_imag_lower,mask_imag_upper) # (total_symbols,2**(k/2)) -- This should have exactly one True value per row
        indices_imag = tf.argmax(mask_imag,axis=1) #  (total_symbols,)

        intervals=tf.stack([tf.gather(self.left_inf_real, indices_real),
                            tf.gather(self.right_inf_real, indices_real),
                            tf.gather(self.left_inf_imag, indices_imag),
                            tf.gather(self.right_inf_imag, indices_imag)],
                            axis=-1) # (total_symbols, 4)
        
        intervals = tf.reshape(intervals, (*received_symbols.shape, 4))  # (...,N,1,1,4)

        # Note that P(a<X<b) = 1/2 * ( erf((b-μ)/(σ√2)) - erf((a-μ)/(σ√2))) for X ~ 𝒩(μ,σ²)
        # value1 = (tf.math.erf((intervals[..., 1] - tf.math.real(self.candidate_symbols_reshaped)) * (
        #     tf.sqrt(SNRs))))
        # value2 = tf.math.erf((intervals[..., 0] - tf.math.real(self.candidate_symbols_reshaped)) * (
        #                      tf.sqrt(SNRs)))
        # value3 = tf.math.erf((intervals[..., 3] - tf.math.imag(self.candidate_symbols_reshaped)) * (
        #             tf.sqrt(SNRs)))
        # value4 = tf.math.erf((intervals[..., 2] - tf.math.imag(self.candidate_symbols_reshaped)) * (
        #              tf.sqrt(SNRs)))
        # print('value1 is of shape',value1.shape,'and is equal to','=\n',value1[...,0])
        # print('value2 is of shape',value2.shape,'and is equal to','=\n',value2[...,0])
        # print('value3 is of shape',value3.shape,'and is equal to','=\n',value3[...,0])
        # print('value4 is of shape',value4.shape,'and is equal to','=\n',value4[...,0])
        
        prob = (1 / 4 * (tf.math.erf((intervals[..., 1] - tf.math.real(self.candidate_symbols_reshaped)) * (
            tf.sqrt(SNRs))) -  # real part, right of interval
                         tf.math.erf((intervals[..., 0] - tf.math.real(self.candidate_symbols_reshaped)) * (
                             tf.sqrt(SNRs)))) *  # real part, left of interval
                (tf.math.erf((intervals[..., 3] - tf.math.imag(self.candidate_symbols_reshaped)) * (
                    tf.sqrt(SNRs))) -  # imag part, right of interval
                 tf.math.erf((intervals[..., 2] - tf.math.imag(self.candidate_symbols_reshaped)) * (
                     tf.sqrt(SNRs)))))  # imag part, left of interval
        prob = prob[..., 0]  # (...,N,2**k)
        log_prob = tf.math.log(prob)
        if self.return_bit_llrs is False:
            if self.sum_over_rx_syms:
                log_prob_summed = tf.reduce_sum(log_prob, axis=-2)  # (...,2**k)
                most_likely_indices = tf.argmax(log_prob_summed,
                                                axis=-1)  # (...) and the values are between 0 (inclusive) to 2**k (exclusive)
                most_likely_symbols = tf.gather(self.candidate_symbols, most_likely_indices)  # (...)
                return most_likely_symbols  # , log_prob_summed # (...) and (...,2**k)
            else:
                return log_prob  # (...,N,2**k) # N is the number of Rx Nodes and 2**k is the number of candidate symbols.
        else:
            prob_new = prob  # Add with the smallest positive number to avoid log(0) issues
            prob_multiplied = tf.reduce_prod(prob_new, axis=-2)  # (...,2**k)
            prob_multiplied = tf.maximum(prob_multiplied,self.EPS) # To avoid log(0) issues
            LLR_list = []
            for k in range(self.k_constellation - 1, -1, -1):  # [k_constellation-1, k_constellation-2, ... , 1, 0]
                # # e.g. [8,9,10,11,12,13,14,15]
                # syms_indices_that_have_bit1 = tf.where((tf.range(2 ** self.k_constellation) // (2 ** k)) % 2 == 1)[:, 0]
                # # e.g. [0,1,2,3,4,5,6,7]
                # syms_indices_that_have_bit0 = tf.where((tf.range(2 ** self.k_constellation) // (2 ** k)) % 2 == 0)[:, 0]
                # # (...,2**(k-1)) after tf.gather and (...) after tf.reduce_sum
                # sum_prob_bit1 = tf.reduce_sum(tf.gather(prob_multiplied, syms_indices_that_have_bit1, axis=-1), axis=-1)
                sum_prob_bit1 = tf.reduce_sum(tf.boolean_mask(prob_multiplied,mask=((tf.range(2 ** self.k_constellation) // (2 ** k)) % 2 == 1), axis=prob_multiplied.ndim-1),axis=-1)
                # (...,2**(k-1)) after tf.gather and (...) after tf.reduce_sum
                # sum_prob_bit0 = tf.reduce_sum(tf.gather(prob_multiplied, syms_indices_that_have_bit0, axis=-1), axis=-1)
                sum_prob_bit0 = tf.reduce_sum(tf.boolean_mask(prob_multiplied,mask=((tf.range(2 ** self.k_constellation) // (2 ** k)) % 2 == 0), axis=prob_multiplied.ndim-1),axis=-1)
                LLR_list.append((tf.math.log(sum_prob_bit1) - tf.math.log(sum_prob_bit0))/(tf.math.log(2.0)))
                # LLR_list.append((tf.math.log(sum_prob_bit1) - tf.math.log(sum_prob_bit0)))
                pass
            LLR = tf.stack(LLR_list, axis=-1)  # (...,k_constellation)
            return LLR
