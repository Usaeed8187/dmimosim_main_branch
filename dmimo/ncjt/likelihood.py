import tensorflow as tf
import numpy as np
from sionna.mapping import Constellation


def hard_log_likelihood(received_symbol: tf.Tensor,
                        SNR: tf.Tensor,
                        k_constellation=2,
                        sum_over_Rx_Syms=True,
                        return_bit_LLRs=False) -> tf.Tensor:
    """
    This function computes log2 P(``received_symbol`` | ``candidate_symbol``)
    for all possible choices of ``candidate_symbol`` when the received
    symbol is coming from detection of a symbol passed over an AWGN channel with
    SNR ``SNR``.
    If ``sum_over_Rx_Syms`` is True, the summation of the log2 of probabilities will
    be calculated and the ``candidate_symbol`` that has the most sum log likelihood
    will be returned.
    
    
    :param tf.Tensor received_symbol: the received detected symbol of shape (...,M,N), where N is the number of copies of the same symbol.
    :param tf.Tensor SNR: The signal-to-noise power ratio. Has to have the same shape as ``received_symbol``.
    :param int k_constellation: The number of bits per symbol of the QAM symbol constellation
    :param bool return_sum: whether to sum over the log likelihoods over the last dimension of ``received_symbol``
    :return most_likely_symbols:
        if `sum_over_Rx_Syms==True` and `return_bit_LLRs==False` the summation of the log2 of probabilities will
        be calculated and the ``candidate_symbol`` that has the most sum log likelihood
        will be returned. The shape would be (...,M)
    :return log_prob: 
        if `sum_over_Rx_Syms==False` and `return_bit_LLRs==False` the log2 of the probability of receiving ``received_symbol`` given
        we had sent ``candidate_symbol`` through an AWGN channel for all possible choices of candidate symbols is returned.
        The shape would be (...,M,N,2**k)
    :return LLR:
        if `return_bit_LLRs==True` then the loglikelihood of each bit would be returned.
        The shape would be  (...,M,k_constellation)
    """

    assert received_symbol.shape == SNR.shape
    constel = Constellation('qam',k_constellation)
    valid_points = tf.reshape(constel.points,[*[1 for _ in range(received_symbol.ndim)],-1]) # shape= [1,...,1,2**k]
    
    assert tf.reduce_all(tf.reduce_any((received_symbol[...,tf.newaxis] == valid_points),axis=-1)), 'The received symbol should be part of the constellation with ``k_constellation``-QAM with unit average power'

    received_symbols = received_symbol[...,tf.newaxis,tf.newaxis] # (...,N,1,1)
    SNRs = tf.math.real(SNR[...,tf.newaxis,tf.newaxis]) # (...,N,1,1)
    candidate_symbols = tf.squeeze(valid_points) # (2**k,)
    candidate_real = tf.sort(tf.unique((tf.math.real(candidate_symbols)))[0]) # (2**(k/2),)
    candidate_imag = tf.sort(tf.unique((tf.math.imag(candidate_symbols)))[0]) # (2**(k/2),)
    
    middle_points_real = ((candidate_real[...,1:] + candidate_real[...,:-1])/2) # (2**(k/2)-1,)
    middle_points_imag = ((candidate_imag[...,1:] + candidate_imag[...,:-1])/2) # (2**(k/2)-1,)
    
    shape_real = [1 for _ in range(received_symbol.ndim)]+[-1,1]
    shape_imag = [1 for _ in range(received_symbol.ndim)]+[1,-1]
    left_inf_real = tf.reshape(tf.convert_to_tensor([-np.inf, *(middle_points_real)]),(shape_real)) # (1,...,1,2**(k/2),1)
    left_inf_imag = tf.reshape(tf.convert_to_tensor([-np.inf, *(middle_points_imag)]),(shape_imag)) # (1,...,1,1,2**(k/2))
    right_inf_real = tf.reshape(tf.convert_to_tensor([*(middle_points_real), np.inf]),(shape_real)) # (1,...,1,2**(k/2),1)
    right_inf_imag = tf.reshape(tf.convert_to_tensor([*(middle_points_imag), np.inf]),(shape_imag)) # (1,...,1,1,2**(k/2))

    # Mapping each received symbol to an interval. 
    # Function "fn" is not used, "fn_v2" is used because it's faster.

    # @tf.function
    def fn(x:tf.Tensor):
        '''
        Not being used. Too slow.
        This function receives a symbol, and then determines in which interval it is located.
        It is supposed to be used with tf.map_fn .
        '''
        real_idx = tf.squeeze(tf.where(tf.math.real(x) == candidate_real))
        imag_idx = tf.squeeze(tf.where(tf.math.imag(x) == candidate_imag))
        
        return tf.cast(tf.convert_to_tensor([tf.squeeze(left_inf_real)[real_idx],tf.squeeze(right_inf_real)[real_idx],
                                     tf.squeeze(left_inf_imag)[imag_idx],tf.squeeze(right_inf_imag)[imag_idx]]),dtype=x.dtype)
    # intervals = tf.reshape(tf.map_fn(fn,tf.reshape(received_symbols,[-1])),shape=[*received_symbols.shape,4])

    # @tf.function
    def fn_v2(x:tf.Tensor):
        x_reshaped = tf.reshape(x,[-1,1])
        candidate_real_reshaped = tf.reshape(candidate_real,[1,-1])
        real_indices = tf.where(tf.math.real(x_reshaped) == candidate_real_reshaped)[:,1] # shape = (total_symbols,)
        # Explanation: the first column of output of tf.where used here, is just a range from 0 to total_symbols. We don't need it.
        candidate_imag_reshaped = tf.reshape(candidate_imag,[1,-1])
        imag_indices = tf.where(tf.math.imag(x_reshaped) == candidate_imag_reshaped)[:,1] # shape = (total_symbols,)

        intervals_real = tf.stack([tf.gather(tf.squeeze(left_inf_real),real_indices),
                                    tf.gather(tf.squeeze(right_inf_real),real_indices)],axis=1) # shape = (total_symbols,2)
        intervals_imag = tf.stack([tf.gather(tf.squeeze(left_inf_imag),imag_indices),
                                    tf.gather(tf.squeeze(right_inf_imag),imag_indices)],axis=1) # shape = (total_symbols,2)
        intervals = tf.concat((intervals_real,intervals_imag),axis=1)
        return tf.reshape(intervals,(*x.shape,4))
    
    intervals = fn_v2(received_symbols) # (...,N,1,1,4)

    candidate_symbols_reshaped = valid_points[...,tf.newaxis] # (1,...,1,2**k,1)

    # Note that P(a<X<b) = 1/2 * ( erf((b-μ)/(σ√2)) - erf((a-μ)/(σ√2))) for X ~ 𝒩(μ,σ²)

    prob = (1/4 * (tf.math.erf( (intervals[...,1]-tf.math.real(candidate_symbols_reshaped))*(tf.sqrt(SNRs)) ) - # real part, right of interval
                   tf.math.erf( (intervals[...,0]-tf.math.real(candidate_symbols_reshaped))*(tf.sqrt(SNRs)) )) * # real part, left of interval
                  (tf.math.erf( (intervals[...,3]-tf.math.imag(candidate_symbols_reshaped))*(tf.sqrt(SNRs)) ) - # imag part, right of interval
                   tf.math.erf( (intervals[...,2]-tf.math.imag(candidate_symbols_reshaped))*(tf.sqrt(SNRs)) ))) # imag part, left of interval
    prob = prob[...,0] # (...,N,2**k)
    log_prob = tf.math.log(prob)/tf.math.log(2.0) # Equivalent to log2. Shape= (...,N,2**k)
    if return_bit_LLRs == False:
        if sum_over_Rx_Syms:
            log_prob_summed = tf.reduce_sum(log_prob,axis=-2) # (...,2**k)
            most_likely_indices = tf.argmax(log_prob_summed,axis=-1) # (...) and the values are between 0 (inclusive) to 2**k (exclusive)
            most_likely_symbols = tf.gather(candidate_symbols,most_likely_indices) # (...)
            return most_likely_symbols #, log_prob_summed # (...) and (...,2**k)
        else:
            return log_prob # (...,N,2**k) # N is the number of Rx Nodes and 2**k is the number of candidate symbols.
    else:
        prob_multiplied = tf.reduce_prod(prob,axis=-2) # (...,2**k)
        LLR_list = []
        for k in range(k_constellation-1,-1,-1): # [k_constellation-1, k_constellation-2, ... , 1, 0]
            syms_indices_that_have_bit1 = tf.where((tf.range(2**k_constellation)//(2**k))%2 == 1)[:,0] # e.g. [8,9,10,11,12,13,14,15]
            syms_indices_that_have_bit0 = tf.where((tf.range(2**k_constellation)//(2**k))%2 == 0)[:,0] # e.g. [0,1,2,3,4,5,6,7]
            sum_prob_bit1 = tf.reduce_sum(tf.gather(prob_multiplied,syms_indices_that_have_bit1,axis=-1),axis=-1) # (...,2**(k-1)) after tf.gather and (...) after tf.reduce_sum
            sum_prob_bit0 = tf.reduce_sum(tf.gather(prob_multiplied,syms_indices_that_have_bit0,axis=-1),axis=-1) # (...,2**(k-1)) after tf.gather and (...) after tf.reduce_sum
            LLR_list.append(tf.math.log(sum_prob_bit1)-tf.math.log(sum_prob_bit0))
            pass
        LLR = tf.stack(LLR_list,axis=-1)  # (...,k_constellation)
        return LLR
            

if __name__ == '__main__':
    constellation = Constellation('qam', 4)  # 16QAM
    candidate_symbols = constellation.points  # (2**k,)
    real = tf.sort(tf.unique((tf.math.real(candidate_symbols)))[0])  # (2**(k/2),)
    imag = tf.sort(tf.unique((tf.math.imag(candidate_symbols)))[0])  # (2**(k/2),)
    a = tf.convert_to_tensor([tf.complex(real[1], imag[i]) for i in range(1, 4)])
    print(a)
    print(hard_log_likelihood(a, tf.ones_like(a), k_constellation=4, return_bit_LLRs=True))
    
