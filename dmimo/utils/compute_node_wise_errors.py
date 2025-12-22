import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.utils.metrics import compute_ber, compute_bler



def compute_UE_wise_BER(info_bits, dec_bits, num_tx_streams_per_node, num_tx_streams):

    BS_indices = tf.range(0, num_tx_streams_per_node * 2)
    
    UE_indices_list = []
    start_value = num_tx_streams_per_node*2
    num_rx_UEs = int((num_tx_streams - num_tx_streams_per_node * 2) / num_tx_streams_per_node)
    for i in range(num_rx_UEs):
        row = tf.range(start_value, start_value + num_tx_streams_per_node)
        UE_indices_list.append(row)
        # Update start_value for the next row
        start_value = start_value + num_tx_streams_per_node
    UE_indices = tf.stack(UE_indices_list)

    node_wise_ber = 0.5*np.ones(num_rx_UEs + 1)
    node_wise_bler = 0.5*np.ones(num_rx_UEs + 1)
    for node_idx in range(num_rx_UEs + 1):
        if node_idx == 0:
            node_wise_ber[node_idx] = compute_ber(tf.gather(info_bits, BS_indices, axis=2), tf.gather(dec_bits, BS_indices, axis=2)).numpy()
            node_wise_bler[node_idx] = compute_bler(tf.gather(info_bits, BS_indices, axis=2), tf.gather(dec_bits, BS_indices, axis=2)).numpy()
        else:
            node_wise_ber[node_idx] = compute_ber(tf.gather(info_bits, UE_indices[node_idx-1], axis=2), tf.gather(dec_bits, UE_indices[node_idx-1], axis=2)).numpy()
            node_wise_bler[node_idx] = compute_bler(tf.gather(info_bits, UE_indices[node_idx-1], axis=2), tf.gather(dec_bits, UE_indices[node_idx-1], axis=2)).numpy()
    
    return node_wise_ber, node_wise_bler

def compute_UE_wise_SER(x ,x_hard, num_tx_streams_per_node, num_tx_streams):

    BS_indices = tf.range(0, num_tx_streams_per_node * 2)
    
    UE_indices_list = []
    start_value = num_tx_streams_per_node*2
    num_rx_UEs = int((num_tx_streams - num_tx_streams_per_node*2) / num_tx_streams_per_node)
    for i in range(num_rx_UEs):
        row = tf.range(start_value, start_value + num_tx_streams_per_node)
        UE_indices_list.append(row)
        # Update start_value for the next row
        start_value = start_value + num_tx_streams_per_node
    UE_indices = tf.stack(UE_indices_list)

    uncoded_ser = np.ones(num_rx_UEs + 1)
    for node_idx in range(num_rx_UEs + 1):
        if node_idx == 0:
            uncoded_ser[node_idx] = np.count_nonzero(tf.gather(x, BS_indices, axis=2) - tf.gather(x_hard, BS_indices, axis=2)) / np.prod(tf.gather(x, BS_indices, axis=2).shape)
        else:
            uncoded_ser[node_idx] = np.count_nonzero(tf.gather(x, UE_indices[node_idx-1], axis=2) - tf.gather(x_hard, UE_indices[node_idx-1], axis=2)) / np.prod(tf.gather(x, UE_indices[node_idx-1], axis=2).shape)

    return uncoded_ser
    