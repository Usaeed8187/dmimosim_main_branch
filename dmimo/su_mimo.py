import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import Ns3Config, SimConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation
from dmimo.channel import standard_rc_pred_freq_mimo
from dmimo.mimo import SVDPrecoder, SVDEqualizer, rankAdaptation, linkAdaptation
from dmimo.mimo import ZFPrecoder
from dmimo.mimo import update_node_selection
from dmimo.utils import add_frequency_offset, add_timing_offset, cfo_val, sto_val

from .txs_mimo import TxSquad


class SU_MIMO(Model):

    def __init__(self, cfg: SimConfig, rg_csi: ResourceGrid, **kwargs):
        """
        Create SU-MIMO simulation object

        :param cfg: simulation settings
        :param rg_csi: Resource grid for CSI estimation
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.rg_csi = rg_csi
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2

        # CFO and STO settings
        self.sto_sigma = sto_val(cfg, cfg.sto_sigma)
        self.cfo_sigma = cfo_val(cfg, cfg.cfo_sigma)

        self.num_rx_ant = 8   # total number of Rx antennas for effective channel
        # The number of transmitted streams is less than or equal to the number of Rx UE antennas
        assert cfg.num_tx_streams <= self.num_rx_ant
        self.num_streams_per_tx = cfg.num_tx_streams

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.array([[1]])  # 1-Tx 1-RX for SU-MIMO

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        # Adjust guard subcarriers for different number of streams
        csi_effective_subcarriers = self.rg_csi.num_effective_subcarriers
        csi_guard_carriers_1 = self.rg_csi.num_guard_carriers[0]
        csi_guard_carriers_2 = self.rg_csi.num_guard_carriers[1]
        effective_subcarriers = (csi_effective_subcarriers // self.num_streams_per_tx) * self.num_streams_per_tx
        guard_carriers_1 = (csi_effective_subcarriers - effective_subcarriers) // 2
        guard_carriers_2 = (csi_effective_subcarriers - effective_subcarriers) - guard_carriers_1
        guard_carriers_1 += csi_guard_carriers_1
        guard_carriers_2 += csi_guard_carriers_2

        # OFDM resource grid (RG) for normal transmission
        self.rg = ResourceGrid(num_ofdm_symbols=14,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=self.num_streams_per_tx,
                               cyclic_prefix_length=64,
                               num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=[2, 11])

        # Update number of data bits and LDPC params
        cfg.ldpc_n = int(2 * self.rg.num_data_symbols)  # Number of coded bits
        cfg.ldpc_k = int(cfg.ldpc_n * cfg.code_rate)  # Number of information bits
        self.num_codewords = cfg.modulation_order // 2  # number of codewords per frame
        self.num_bits_per_frame = cfg.ldpc_k * self.num_codewords * self.num_streams_per_tx
        self.num_uncoded_bits_per_frame = cfg.ldpc_n * self.num_codewords * self.num_streams_per_tx

        # The encoder maps information bits to coded bits
        self.encoder = LDPC5GEncoder(cfg.ldpc_k, cfg.ldpc_n)

        # LDPC interleaver
        self.intlvr = RowColumnInterleaver(3072, axis=-1)  # fixed design for current RG config
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", cfg.modulation_order)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(self.rg)

        # The zero forcing precoder
        self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)

        # SVD-based precoder and equalizer
        self.svd_precoder = SVDPrecoder(self.rg, sm, return_effective_channel=True)
        self.svd_equalizer = SVDEqualizer(self.rg, sm)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(self.rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order)

        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    def call(self, dmimo_chans: dMIMOChannels, h_freq_csi, info_bits):
        """
        Signal processing for one SU-MIMO transmission cycle (P1/P2/P3)

        Effective channel models for phase 2 (P2) and phase 3 (P3) are used, phase 1 (P1) transmission
        is assumed to always provide enough data bandwidth for P2.

        :param dmimo_chans: dMIMO channels
        :param h_freq_csi: CSI feedback for precoding
        :param info_bits: information bits
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        # LDPC encoder processing
        info_bits = tf.reshape(info_bits, [self.batch_size, 1, self.rg.num_streams_per_tx,
                                           self.num_codewords, self.encoder.k])
        c = self.encoder(info_bits)
        c = tf.reshape(c, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords * self.encoder.n])

        # Interleaving for coded bits
        d = self.intlvr(c)

        # QAM mapping for the OFDM grid
        x = self.mapper(d)
        x_rg = self.rg_mapper(x)

        # apply precoding to OFDM grids
        if self.cfg.precoding_method == "ZF":
            x_precoded, g = self.zf_precoder([x_rg, h_freq_csi])
        elif self.cfg.precoding_method == "SVD":
            x_precoded, g = self.svd_precoder([x_rg, h_freq_csi])
        else:
            ValueError("unsupported precoding method")

        # add CFO/STO to simulate synchronization errors
        if self.sto_sigma > 0:
            x_precoded = add_timing_offset(x_precoded, self.sto_sigma)
        if self.cfo_sigma > 0:
            x_precoded = add_frequency_offset(x_precoded, self.cfo_sigma)

        # apply dMIMO channels to the resource grid in the frequency domain.
        y = dmimo_chans([x_precoded, self.cfg.first_slot_idx])

        # SVD equalization
        if self.cfg.precoding_method == "SVD":
            y = self.svd_equalizer([y, h_freq_csi, self.num_streams_per_tx])

        # LS channel estimation with linear interpolation
        no = 0.1  # initial noise estimation (tunable param)
        h_hat, err_var = self.ls_estimator([y, no])

        # LMMSE equalization
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])

        # Soft-output QAM demapper
        llr = self.demapper([x_hat, no_eff])

        # Hard-decision bit error rate
        d_hard = tf.cast(llr > 0, tf.float32)
        uncoded_ber = compute_ber(d, d_hard).numpy()

        # Hard-decision symbol error rate
        x_hard = self.mapper(d_hard)
        uncoded_ser = np.count_nonzero(x - x_hard) / np.prod(x.shape)

        # LLR deinterleaver for LDPC decoding
        llr = self.dintlvr(llr)
        llr = tf.reshape(llr, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords, self.encoder.n])

        # LDPC hard-decision decoding
        dec_bits = self.decoder(llr)

        return dec_bits, uncoded_ber, uncoded_ser, x_hat


def do_rank_link_adaptation(cfg, dmimo_chans, h_est, rx_snr_db):

    # Rank adaptation
    rank_adaptation = rankAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                     architecture='SU-MIMO', snrdb=rx_snr_db, fft_size=cfg.fft_size,
                                     precoder='SVD')

    rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')

    if rank_adaptation.use_mmse_eesm_method:
        rank = rank_feedback_report[0]
        rate = rank_feedback_report[1]

        print("\n", "rank (SU-MIMO) = ", rank, "\n")
        print("\n", "rate (SU-MIMO) = ", rate, "\n")

    else:
        rank = rank_feedback_report
        rate = []

        print("\n", "rank (SU-MIMO) = ", rank, "\n")

    # Link adaptation
    data_sym_position = np.arange(0, 14)
    link_adaptation = linkAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                     architecture='SU-MIMO', snrdb=rx_snr_db, nfft=cfg.fft_size,
                                     N_s=rank, data_sym_position=data_sym_position, lookup_table_size='short')

    mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO')

    if link_adaptation.use_mmse_eesm_method:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = mcs_feedback_report[1]

        # Majority vote for MCS selection for now
        values, counts = np.unique(qam_order_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        modulation_order = int(most_frequent_value)

        values, counts = np.unique(code_rate_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        code_rate = most_frequent_value

        print("\n", "Bits per stream (SU-MIMO) = ", cfg.modulation_order, "\n")
        print("\n", "Code-rate per stream (SU-MIMO) = ", cfg.code_rate, "\n")
    else:
        qam_order_arr = mcs_feedback_report[0]
        modulation_order = int(np.min(qam_order_arr))
        code_rate = []  # FIXME update code rate

        print("\n", "Bits per stream (SU-MIMO) = ", cfg.modulation_order, "\n")

    return rank, rate, modulation_order, code_rate


def sim_su_mimo(cfg: SimConfig):
    """
    Simulation of SU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits]
    """

    # dMIMO channels from ns-3 simulator
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO-Forward", add_noise=True)

    # UE selection
    if cfg.enable_ue_selection is True:
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
        ns3cfg.update_ue_mask(tx_ue_mask, rx_ue_mask)

    # Total number of antennas in the TxSquad, always use all gNB antennas
    num_txs_ant = 2 * cfg.num_tx_ue_sel + ns3cfg.num_bs_ant

    # Adjust guard subcarriers for channel estimation grid
    csi_effective_subcarriers = (cfg.fft_size // num_txs_ant) * num_txs_ant
    csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
    csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

    # Resource grid for channel estimation
    rg_csi = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=num_txs_ant,
                          cyclic_prefix_length=cfg.cyclic_prefix_len,
                          num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

    # Channel CSI estimation using channels in previous frames/slots
    if cfg.perfect_csi is True:
        # Perfect channel estimation
        h_freq_csi, rx_snr_db = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                         batch_size=cfg.num_slots_p2)
    elif cfg.csi_prediction is True:
        rc_predictor = standard_rc_pred_freq_mimo('SU_MIMO')
        # Get CSI history
        # TODO: optimize channel estimation and optimization procedures (currently very slow)
        h_freq_csi_history = rc_predictor.get_csi_history(cfg.first_slot_idx, cfg.csi_delay,
                                                          rg_csi, dmimo_chans)
        # Do channel prediction
        h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
    else:
        # LMMSE channel estimation
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_sigma=cfo_val(cfg, cfg.cfo_sigma),
                                                           sto_sigma=sto_val(cfg, cfg.sto_sigma))

    # Rank and link adaptation
    if cfg.rank_adapt and cfg.link_adapt and cfg.first_slot_idx == cfg.start_slot_idx:
        _, rx_snr_db = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                batch_size=cfg.num_slots_p2)
        rank, rate, modulation_order, code_rate = \
            do_rank_link_adaptation(cfg, dmimo_chans, h_freq_csi, rx_snr_db)

        # Update rank and total number of streams
        cfg.num_tx_streams = rank
        cfg.modulation_order = modulation_order
        cfg.code_rate = code_rate

    # Create SU-MIMO simulation
    su_mimo = SU_MIMO(cfg, rg_csi)

    # Generate information source
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, su_mimo.num_bits_per_frame])

    # TxSquad transmission (P1)
    if cfg.enable_txsquad is True:
        tx_squad = TxSquad(cfg, su_mimo.num_bits_per_frame)
        txs_chans = dMIMOChannels(ns3cfg, "TxSquad", add_noise=True)
        info_bits_new, txs_ber, txs_bler = tx_squad(txs_chans, info_bits)
        print("BER: {}  BLER: {}".format(txs_ber, txs_bler))
        assert txs_ber <= 1e-3, "TxSquad transmission BER too high"

    # SU-MIMO transmission (P2 & P3)
    dec_bits, uncoded_ber, uncoded_ser, x_hat = su_mimo(dmimo_chans, h_freq_csi, info_bits)

    # Update error statistics
    info_bits = tf.reshape(info_bits, dec_bits.shape)
    coded_ber = compute_ber(info_bits, dec_bits).numpy()
    coded_bler = compute_bler(info_bits, dec_bits).numpy()

    # Goodput and throughput estimation
    goodbits = (1.0 - coded_ber) * su_mimo.num_bits_per_frame
    userbits = (1.0 - coded_bler) * su_mimo.num_bits_per_frame
    ratedbits = (1.0 - uncoded_ser) * su_mimo.num_uncoded_bits_per_frame

    return [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits]


def sim_su_mimo_all(cfg: SimConfig):
    """"
    Simulation of SU-MIMO scenario according to the frame structure
    """

    total_cycles = 0
    uncoded_ber, ldpc_ber, goodput, throughput, bitrate = 0, 0, 0, 0, 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        bers, bits = sim_su_mimo(cfg)
        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        goodput += bits[0]
        throughput += bits[1]
        bitrate += bits[2]

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = cfg.num_slots_p2/(cfg.num_slots_p1 + cfg.num_slots_p2)
    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    bitrate = bitrate / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput, bitrate]
