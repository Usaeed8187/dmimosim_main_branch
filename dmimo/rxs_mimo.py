import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
from sionna.ofdm import LMMSEEqualizer, MMSEPICDetector
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import SimConfig, Ns3Config


class RxSquad(Model):
    """
    Implement Rx Squad data transmission in phase 3 (P3)

    For the Rx squad uplink, OFDMA is required for more than 2 UEs. Here we simulate
    the average performance for each pair of UEs uplink across all subcarriers, i.e.,
    two UEs are transmitting simultaneously to the BS with 4x4 MIMO channels.

    """

    def __init__(self, cfg: SimConfig, ns3cfg: Ns3Config, rxs_bits_per_frame: int, **kwargs):
        """
        Initialize RxSquad simulation

        :param cfg: simulation settings
        :param rxs_bits_per_frame: total number of bits per subframe/slot for all UEs
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.ns3cfg = ns3cfg
        self.batch_size = cfg.num_slots_p1  # batch processing for all slots in phase 1

        # The number of transmitted streams is equal to the number of BS antennas
        num_bs_ant = 4
        self.num_streams_per_tx = num_bs_ant

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = [[1]]

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        csi_effective_subcarriers = (cfg.fft_size // num_bs_ant) * num_bs_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

        effective_subcarriers = (csi_effective_subcarriers // self.num_streams_per_tx) * self.num_streams_per_tx
        guard_carriers_1 = (csi_effective_subcarriers - effective_subcarriers) // 2
        guard_carriers_2 = (csi_effective_subcarriers - effective_subcarriers) - guard_carriers_1
        guard_carriers_1 += csi_guard_carriers_1
        guard_carriers_2 += csi_guard_carriers_2

        self.rg_csi = ResourceGrid(num_ofdm_symbols=14,
                                   fft_size=cfg.fft_size,
                                   subcarrier_spacing=cfg.subcarrier_spacing,
                                   num_tx=1,
                                   num_streams_per_tx=num_bs_ant,
                                   cyclic_prefix_length=64,
                                   num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                                   dc_null=False,
                                   pilot_pattern="kronecker",
                                   pilot_ofdm_symbol_indices=[2, 11])

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=self.num_streams_per_tx,
                          cyclic_prefix_length=64,
                          num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

        self.coderate = 5/6  # fixed code rate for current design
        num_bits_per_stream = np.ceil(rxs_bits_per_frame * (cfg.num_slots_p2 / cfg.num_slots_p1) / self.num_streams_per_tx)
        bits_per_symbol = int(np.ceil(num_bits_per_stream / self.coderate / rg.num_data_symbols.numpy()))
        self.num_bits_per_symbol = max(2, bits_per_symbol)
        if self.num_bits_per_symbol % 2 != 0:
            self.num_bits_per_symbol += 1  # must be even
        self.num_bits_per_frame = int(rg.num_data_symbols.numpy() * self.num_bits_per_symbol * self.coderate * self.num_streams_per_tx)
        assert self.num_bits_per_symbol <= 12, "data bandwidth requires unsupported modulation order"

        self.num_codewords = self.num_bits_per_symbol  # number of codewords per frame
        self.ldpc_n = rg.num_data_symbols.numpy()      # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * self.coderate)  # Number of information bits
        self.ldpc_padding = self.num_bits_per_frame * cfg.num_slots_p1 - rxs_bits_per_frame * cfg.num_slots_p2

        # The encoder maps information bits to coded bits
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n)

        # LDPC interleaver
        self.intlvr = RowColumnInterleaver(3072, axis=-1)
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", self.num_bits_per_symbol)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(rg)
        self.rg_csi_mapper = ResourceGridMapper(self.rg_csi)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", self.num_bits_per_symbol)

        # The detector produces soft-decision output for all coded bits
        self.detector = MMSEPICDetector("bit", rg, sm, constellation_type="qam",
                                        num_bits_per_symbol=self.num_bits_per_symbol,
                                        num_iter=2, hard_out=False)

        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    def call(self, rxs_chans, info_bits, return_data_only=False):
        """
        Signal processing for RxSquad downlink transmission (P1)

        :param rxs_chans: RxSquad channels
        :param info_bits: information bits
        :return: decoded bits, LDPC BER, LDPC BLER
        """

        # payload padding
        b = tf.reshape(info_bits, (-1))
        b = tf.concat((b, tf.zeros(self.ldpc_padding)), axis=0)
        b = tf.reshape(b, [self.batch_size, 1, self.num_streams_per_tx, self.num_codewords, self.encoder.k])

        # LDPC encoding and interleaving
        c = self.encoder(b)
        c = tf.reshape(c, [self.batch_size, 1, self.num_streams_per_tx, self.num_codewords * self.encoder.n])
        d = self.intlvr(c)

        # QAM mapping on OFDM grid
        x = self.mapper(d)
        # x_rg has shape [batch_size, num_tx, num_tx_streams, num_ofdm_syms, fft_size)
        x_rg = self.rg_mapper(x)

        if return_data_only:
            return x_rg

        # check all UEs
        ue_data = []
        ue_ber_avg, ue_bler_avg = 0.0, 0.0
        ue_ber_max, ue_bler_max = 0.0, 0.0
        for rx_ue_idx in range(0, self.ns3cfg.num_rxue_sel, 2):

            # apply dMIMO channels to the resource grid in the frequency domain
            # only using the channel for the current UEs
            tx_ant_mask = np.arange(2 * rx_ue_idx, 2 * rx_ue_idx + self.num_streams_per_tx)
            y_rg, _ = rxs_chans([x_rg, self.cfg.first_slot_idx, tx_ant_mask, None])

            # LS channel estimation with linear interpolation
            no = 1e-5  # tunable param
            h_hat, err_var = self.ls_estimator([y_rg, no])

            prior = tf.zeros(d.shape)
            det_out = self.detector((y_rg, h_hat, prior, err_var, no))

            # LLR interleaving
            llr = tf.reshape(det_out, d.shape)
            llr = self.dintlvr(llr)

            # LDPC decoder
            llr = tf.reshape(llr, [self.batch_size, 1, self.num_streams_per_tx, self.num_codewords, self.encoder.n])
            dec_bits = self.decoder(llr)

            # Error statistics
            ber = compute_ber(b, dec_bits).numpy()
            bler = compute_bler(b, dec_bits).numpy()
            ue_ber_avg += ber/self.ns3cfg.num_rxue_sel
            ue_bler_avg += bler/self.ns3cfg.num_rxue_sel
            if ber > ue_ber_max:
                ue_ber_max = ber
            if bler > ue_bler_max:
                ue_bler_max = bler
            ue_data = dec_bits

        # Remove padding bits
        if self.ldpc_padding > 0:
            ue_data = tf.reshape(ue_data, (-1))
            ue_data = ue_data[:-self.ldpc_padding]
        # restore original shape
        ue_data = tf.reshape(ue_data, (self.cfg.num_slots_p2, -1))

        return ue_data, ue_ber_avg, ue_bler_avg, ue_ber_max, ue_bler_max
