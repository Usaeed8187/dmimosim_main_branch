import tensorflow as tf
import numpy as np


def sttd_reshape(alamouti_txSignal: tf.Tensor, UeAxis=-3, SymbolAxis=-2) -> tf.Tensor:
    """
    Receives an alamouti-formatted array for multiple users and returns 
    a new array where the users' transmit symbols are put in the respective
    OFDM symbol corresponding to them.

    Parameters:
        alamouti_txSignal: the transmit signal of shape [..., numTxUe, numSyms, 2]
        UeAxis: the axis that indexes the transmit signal of each UE
        SymbolAxis: the axis that indexes the OFDM symbol number of the transmit signal
    Returns:
        STTD_format_signal: the transmit signal in the STTD format, shape [..., numTxUe, numTxUe * numSyms, 2]
    """

    # alamouti_txSignal = alamouti_txSignal.numpy()
    signal_shape = alamouti_txSignal.shape
    numTxUe = signal_shape[UeAxis]
    numSyms = signal_shape[SymbolAxis]
    # numAnt = signal_shape[-1]

    to_be_stacked = []
    for i_UE in range(numTxUe):
        padding = np.zeros((alamouti_txSignal.ndim, 2))  # Create the padding array for feeding to tf.pad
        padding[SymbolAxis, 0] = numSyms * i_UE  # These many zeros before, for the SymbolAxis dimension
        padding[SymbolAxis, 1] = numSyms * (numTxUe - i_UE - 1)  # These many zeros after, for the SymbolAxis dimension
        # Instead of this line:
        # to_be_stacked.append(tf.pad(alamouti_txSignal[...,i_UE:i_UE+1,:,:],padding))
        # We will use slicing to adjust for variety in axis corresponding to UE and symbols
        slicing = [slice(None) for _ in range(alamouti_txSignal.ndim)]  # equivalent to [:,:,...,:] if fed to a slicer
        slicing[UeAxis] = slice(i_UE, i_UE + 1)  # changing that specific slicing for that axis in the slicer
        # Separate out UE i_UE's data symbols and pad before and after it with zeros
        to_be_stacked.append(tf.pad(alamouti_txSignal[*slicing], padding))
        pass
    if numTxUe > 0:
        txSignal_reshaped = tf.concat(to_be_stacked, axis=-3)
    else:  # numTxUe == 0:
        new_shape = (*signal_shape[:-2], signal_shape[-3] * signal_shape[-2], signal_shape[-1])
        txSignal_reshaped = tf.reshape(alamouti_txSignal, new_shape)

    return txSignal_reshaped


def sttd_reshape_v2(alamouti_txSignal: tf.Tensor, SymbolAxis=-2) -> tf.Tensor:
    """
    Receives an alamouti-formatted array for multiple users and returns 
    a new array where the users' transmit symbols are put in the respective
    OFDM symbol corresponding to them.
    Parameters:
        alamouti_txSignal: the transmit signal of shape [... , numSyms, 2] 
        SymbolAxis: the axis that indexes the OFDM symbol number of the transmit signal
    Returns:
        STTD_format_signal: the transmit signal in the STTD format, shape= [..., numSyms , 2 * (numSyms/2)]
    """

    signal_shape = alamouti_txSignal.shape
    numSyms = signal_shape[SymbolAxis]

    txSig_reshaped = tf.reshape(alamouti_txSignal, (*(signal_shape[:-2]), numSyms // 2, 2, 2))
    # print('txSig_reshaped.shape=', txSig_reshaped.shape)
    txSig_sttd = sttd_reshape(txSig_reshaped)  # [..., numSyms/2 , numSyms , 2]
    transpose_dims = (*(range(txSig_sttd.ndim - 3)), txSig_sttd.ndim - 2, txSig_sttd.ndim - 3, txSig_sttd.ndim - 1)
    txSig_sttd2 = tf.transpose(txSig_sttd, transpose_dims)  # [..., numSyms , numSyms/2 , 2]
    # print('txSig_sttd.shape=', txSig_sttd.shape)
    txSig_sttd2 = tf.reshape(txSig_sttd2, (*signal_shape[:-2], numSyms, numSyms))
    # print('txSig_sttd.shape=', txSig_sttd2.shape)
    return txSig_sttd2
