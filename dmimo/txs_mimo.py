import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import SimConfig, Ns3Config
from dmimo.channel import lmmse_channel_estimation
from dmimo.mimo import ZFPrecoder


class TxSquad(Model):
    """
    Implement Tx Squad data transmission in phase 1 (P1)
    """

    def __init__(self, cfg: SimConfig, ns3cfg: Ns3Config, txs_bits_per_frame: int, **kwargs):
        """
        Initialize TxSquad simulation

        :param cfg: simulation settings
        :param txs_bits_per_frame: number of bits per subframe/slot for SU-MIMO operation
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.ns3cfg = ns3cfg
        self.batch_size = cfg.num_slots_p1  # batch processing for all slots in phase 1

        # Define the number of UE and BS antennas.
        num_bs_ant = 4
        num_ue_ant = 2

        # The number of transmitted streams is equal to the number of UE antennas
        self.num_streams_per_tx = num_ue_ant

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

        self.coderate = 11/12  # fixed code rate for current design
        num_bits_per_stream = np.ceil(txs_bits_per_frame * (cfg.num_slots_p2 / cfg.num_slots_p1)
                                      / self.num_streams_per_tx)
        bits_per_symbol = int(np.ceil(num_bits_per_stream / self.coderate / rg.num_data_symbols.numpy()))
        self.num_bits_per_symbol = max(2, bits_per_symbol)
        if self.num_bits_per_symbol % 2 != 0:
            self.num_bits_per_symbol += 1  # must be even
        self.num_bits_per_frame = int(rg.num_data_symbols.numpy() * self.num_bits_per_symbol * self.coderate * self.num_streams_per_tx)
        assert self.num_bits_per_symbol <= 12, "data bandwidth requires unsupported modulation order"

        self.num_codewords = self.num_bits_per_symbol  # number of codewords per frame
        self.ldpc_n = rg.num_data_symbols.numpy()      # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * self.coderate)  # Number of information bits
        self.ldpc_padding = self.num_bits_per_frame * cfg.num_slots_p1 - txs_bits_per_frame * cfg.num_slots_p2

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

        # The zero forcing precoder precodes the transmitter stream towards the intended antennas
        self.zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", self.num_bits_per_symbol)

        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    def call(self, txs_chans, info_bits):
        """
        Signal processing for TxSquad downlink transmission (P1)

        :param txs_chans: TxSquad channels
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
        x_rg = self.rg_mapper(x)

        # Perfect channel estimation or LMMSE channel estimation for precoding
        if self.cfg.perfect_csi:
            h_freq_csi, rx_snr_db, _ = txs_chans.load_channel(slot_idx=self.cfg.first_slot_idx, batch_size=self.batch_size)
        else:
            # LMMSE channel estimation
            h_freq_csi, err_var_csi = lmmse_channel_estimation(txs_chans, self.rg_csi, slot_idx=self.cfg.first_slot_idx,
                                                               batch_size=self.batch_size)

        # Apply basic ZF precoder (optimized precoder will be added later)
        x_precoded, g = self.zf_precoder([x_rg, h_freq_csi])

        # apply dMIMO channels to the resource grid in the frequency domain.
        y, _ = txs_chans([x_precoded, self.cfg.first_slot_idx])

        # check all UEs
        ue_data = []
        ue_ber = 1.0
        ue_bler = 1.0
        for ue_idx in range(self.ns3cfg.num_txue_sel):
            # Received signal for current UE
            y1 = y[:, :, 2*ue_idx:2*ue_idx+self.num_streams_per_tx]

            # LS channel estimation with linear interpolation
            no = 1e-5  # tunable param
            h_hat, err_var = self.ls_estimator([y1, no])

            # LMMSE equalization and demapping
            x_hat, no_eff = self.lmmse_equ([y1, h_hat, err_var, no])
            llr = self.demapper([x_hat, no_eff])

            # LLR interleaving
            llr = self.dintlvr(llr)

            # LDPC decoder
            llr = tf.reshape(llr, [self.batch_size, 1, self.num_streams_per_tx, self.num_codewords, self.encoder.n])
            dec_bits = self.decoder(llr)

            # Error statistics
            ber = compute_ber(b, dec_bits).numpy()
            bler = compute_bler(b, dec_bits).numpy()
            if ber < ue_ber:
                ue_ber = ber
                ue_bler = bler
                ue_data = dec_bits

        # Remove padding bits
        if self.ldpc_padding > 0:
            ue_data = tf.reshape(ue_data, (-1))
            ue_data = ue_data[:-self.ldpc_padding]
        # restore original shape
        ue_data = tf.reshape(ue_data, (self.cfg.num_slots_p2, -1))

        # To simplify simulation, we only return the results for the best case UE,
        # we can return results for all UEs if more accurate error modeling is needed.
        return ue_data, ue_ber, ue_bler
