import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.mapping import Mapper
from sionna.ofdm import ResourceGrid, ResourceGridMapper

from dmimo.config import SimConfig
from .stbc import alamouti_encode


class NCJT_TxUE(Model):
    """
    Implement of the non-coherent transmission of symbols using the Alamouti scheme in the dMIMO phase.
    """
    def __init__(self, cfg: SimConfig, **kwargs):
        """
        Create NCJT TxUE object

        :param cfg: system settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.nAntTxBs = 4  # fixed param for now
        self.nAntTxUe = 2  # fixed param for now

        self.mapper = Mapper("qam", cfg.modulation_order)
        self.rg = ResourceGrid(num_ofdm_symbols=cfg.symbols_per_slot,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=2,
                               cyclic_prefix_length=cfg.cyclic_prefix_len,
                               num_guard_carriers=[0, 0],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=cfg.pilot_indices)

        self.rg_mapper = ResourceGridMapper(self.rg)

    @tf.function(jit_compile=True)  # Enable graph execution to speed things up
    def call(self, bit_stream_dmimo, is_txbs=False):
        """
        Transmission processing for Tx UEs

        param bit_stream_dmimo: bitstream to transmit
        param is_txbs: indicate transmitting from BS
        """

        num_tx_ant = self.nAntTxBs if is_txbs else self.nAntTxUe  # number of transmitting antennas
        batch_size = bit_stream_dmimo.shape[0]

        # Map data stream to QAM symbols
        x = self.mapper(bit_stream_dmimo)  # [batch_size, num_subcarriers, num_data_ofdm_syms]
        x = alamouti_encode(x)  # [batch_size, num_subcarriers, num_data_ofdm_syms, 2]

        # Transpose to make the signal compatible with rg_mapper
        # New shape: [batch_size, 2, num_data_ofdm_syms, num_subcarriers]
        x = tf.transpose(x, [0, 3, 2, 1])
        # New shape: [batch_size, num_tx, num_streams_per_tx, num_data_ofdm_syms * num_subcarriers]
        x = self.rg_mapper(tf.reshape(x, [batch_size, 1, 2, -1]))

        x = tf.reshape(x, [batch_size, 2, self.cfg.symbols_per_slot, -1])
        # New shape: [batch_size, num_subcarriers, num_ofdm_symbols, num_tx_ant]
        x = tf.transpose(x, [0, 3, 2, 1])

        # In case of more than 2 antennas on the BS
        if num_tx_ant > 2:
            x = tf.concat([x for _ in range(num_tx_ant // 2)], axis=-1)

        return x
