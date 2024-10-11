import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.mapping import Constellation
from sionna.mapping import Mapper, Demapper
from dmimo.config import SimConfig
from .loglikelihood import HardLogLikelihood


class NCJT_PostCombination(Model):

    def __init__(self, cfg: SimConfig, return_LLRs=False, perSC_SNR=False, **kwargs):
        """
        Create NCJT post detection combining
        :param cfg: system settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.return_LLRs = return_LLRs
        self.perSC_SNR = perSC_SNR
        self.constel = Constellation('qam', self.cfg.modulation_order)
        self.mapper = Mapper("qam", cfg.modulation_order)
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order, hard_out=True)
        self.signal_dims = 4
        self.loglikelihood = HardLogLikelihood(self.constel, self.signal_dims, return_bit_llrs=return_LLRs)

    # @tf.function(jit_compile=True)  # Enable graph execution to speed things up
    def call(self, rx_bit_streams, gains_list, no):

        if self.perSC_SNR:  # In case we have the exact gain of each subcarrier
            snrs = tf.stack(gains_list, axis=-1) / no
        else:  # In case only per subframe SNR gain is available at the Rx BS
            snrs = tf.stack(gains_list, axis=-1) / no
            # Averaging over OFDM symbols and subcarriers
            snrs = tf.ones_like(snrs) * tf.reduce_mean(snrs, axis=(-2, -3), keepdims=True)

        # Post detection combining
        symbol_streams_at_RxBs = [self.mapper(rx_node_bit_stream) for rx_node_bit_stream in rx_bit_streams]

        # y_combined has shape [cfg.num_subframes_phase2, num_ofdm_symbols]
        y_combined = self.loglikelihood(tf.stack(symbol_streams_at_RxBs, axis=-1), snrs)

        if self.return_LLRs:
            LLRs = y_combined
            return LLRs
        else:
            # detected_bits has shape [num_subframes, num_subcarriers, num_ofdm_symbols * cfg.num_bits_per_symbol]
            detected_bits = self.demapper([y_combined, no])
            # detected_bits = tf.reshape(detected_bits, (self.cfg.modulation_order, self.cfg.num_subcarriers, -1))
            return detected_bits
