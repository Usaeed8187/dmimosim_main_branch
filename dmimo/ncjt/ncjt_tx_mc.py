import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.mapping import Mapper
from sionna.ofdm import ResourceGrid, ResourceGridMapper

from dmimo.config import SimConfig
from .stbc import alamouti_encode

from typing import List

class MC_NCJT_TxUE(Model):
    """
    Implement of the non-coherent transmission of symbols using the Alamouti scheme in the dMIMO phase.
    
    Creates NCJT TxUE object with Multi-cluster support.

    :param cfg: system settings
    :param int num_clusters: Number of clusters at the transmitter side, can be thought of as the number of simultaneous streams
    :param list modulation_order_list: A list of size num_clusters that shows the modulation order used by each cluster. 
    If None, it will revert to cfg.modulation_order. Default None.

    __call__() arguments
    --------------------
    :param tf.Tensor bit_stream_dmimo: bi   tstream to transmit. Shape: [batch_size, num_subcarriers, num_ofdm_syms * modulation_order]
    :param bool is_txbs: indicate transmitting from BS. Default False.
    :param int cluster_idx: the index of the transmitting cluster. Default 0.
    
    """
    def __init__(self, cfg: SimConfig, num_clusters=1, modulation_order_list:list=None, **kwargs):
        
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.nAntTxBs = 4  # fixed param for now
        self.nAntTxUe = 2  # fixed param for now

        self.num_clusters = num_clusters
        if modulation_order_list is None:
            self.modulation_orders = [cfg.modulation_order]
        else:
            self.modulation_orders = modulation_order_list

        self.mappers:List[Mapper] = [Mapper("qam", k) for k in self.modulation_orders]
        self.rg = ResourceGrid(num_ofdm_symbols=cfg.symbols_per_slot,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=2*num_clusters,
                               cyclic_prefix_length=cfg.cyclic_prefix_len,
                               num_guard_carriers=[0, 0],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=cfg.pilot_indices)

        self.rg_mapper = ResourceGridMapper(self.rg)

    # @tf.function(jit_compile=True)  # Enable graph execution to speed things up
    def call(self, bit_stream_dmimo:tf.Tensor, is_txbs:bool=False, cluster_idx:int = 0):
        """
        Transmission processing for Tx UEs

        param bit_stream_dmimo: bitstream to transmit
        param is_txbs: indicate transmitting from BS
        """
        assert bit_stream_dmimo.shape[-1] == (self.cfg.symbols_per_slot - len(self.cfg.pilot_indices))*(self.modulation_orders[cluster_idx]) and (bit_stream_dmimo.shape[-2]==self.cfg.fft_size) , \
            'Call to MC_NCJT_TxUE object: bit_stream_dmimo has the wrong shape.'
        num_tx_ant = self.nAntTxBs if is_txbs else self.nAntTxUe  # number of transmitting antennas
        batch_size = bit_stream_dmimo.shape[0]

        # Map data stream to QAM symbols
        x = self.mappers[cluster_idx](bit_stream_dmimo)  # [batch_size, num_subcarriers, num_data_ofdm_syms]
        x = alamouti_encode(x)  # [batch_size, num_subcarriers, num_data_ofdm_syms, 2]

        # Transpose to make the signal compatible with rg_mapper
        # New shape: [batch_size, 2, num_data_ofdm_syms, num_subcarriers]
        x = tf.transpose(x, [0, 3, 2, 1])

        padding_batch_size_axis = tf.zeros([1, 2], dtype=tf.int32) # Padding matrix 
        # On the num_ant axis, pad zeros as much as cluster_idx before the symbols and self.num_clusters-1-cluster_idx after the symbols 
        padding_ant_axis = tf.convert_to_tensor([[cluster_idx*2 , (self.num_clusters-1-cluster_idx)*2]]) # The *2 is because every cluster acts like an Alamouti transmitter with 2 antennas 
        padding_other_axes = tf.zeros([2,2], dtype=tf.int32)
        padding = tf.concat([padding_batch_size_axis, padding_ant_axis, padding_other_axes], axis=0) # shape = [4,2]
        x = tf.pad(x, padding) # shape = [batch_size, 2*(self.num_clusters), num_data_ofdm_syms, num_subcarriers]
        # New shape: [batch_size, num_tx, num_streams_per_tx, num_data_ofdm_syms * num_subcarriers]
        x = self.rg_mapper(tf.reshape(x, [batch_size, 1, 2*self.num_clusters, -1]))

        x = tf.reshape(x, [batch_size, 2*self.num_clusters, self.cfg.symbols_per_slot, -1])
        # New shape: [batch_size, num_subcarriers, num_ofdm_symbols, num_tx_ant*num_clusters]
        x = tf.transpose(x, [0, 3, 2, 1])

        x = x[...,2*(cluster_idx):2*cluster_idx+2] # shape: [batch_size, num_subcarriers, num_ofdm_symbols, num_tx_ant]

        # In case of more than 2 antennas on the BS
        if num_tx_ant > 2:
            x = tf.concat([x for _ in range(num_tx_ant // 2)], axis=-1)
    
        return x
