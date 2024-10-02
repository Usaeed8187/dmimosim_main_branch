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

from dmimo.config import Ns3Config, SimConfig, NetworkConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation, standard_rc_pred_freq_mimo
from dmimo.mimo import SVDPrecoder, SVDEqualizer, rankAdaptation, linkAdaptation
from dmimo.mimo import ZFPrecoder
from dmimo.utils import add_frequency_offset, add_timing_offset, cfo_val, sto_val


class Baseline(Model):

    def __init__(self, cfg: SimConfig, **kwargs):
        """
        Create Baseline simulation object

        :param cfg: simulation settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2

        # dMIMO configuration
        self.num_bs_ant = 4  # Tx squad BB
        self.num_ue_ant = 4  # Rx squad BB

        # CFO and STO settings
        self.sto_sigma = sto_val(cfg, cfg.sto_sigma)
        self.cfo_sigma = cfo_val(cfg, cfg.cfo_sigma)

        # The number of transmitted streams is less than or equal to the number of UE antennas
        assert cfg.num_tx_streams <= self.num_ue_ant
        self.num_streams_per_tx = cfg.num_tx_streams

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.array([[1]])  # 1-Tx 1-RX for SU-MIMO

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        # Adjust guard subcarriers for channel estimation grid
        csi_effective_subcarriers = (cfg.fft_size // self.num_bs_ant) * self.num_bs_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

        # Resource grid for channel estimation
        self.rg_csi = ResourceGrid(num_ofdm_symbols=14,
                                   fft_size=cfg.fft_size,
                                   subcarrier_spacing=cfg.subcarrier_spacing,
                                   num_tx=1,
                                   num_streams_per_tx=self.num_bs_ant,
                                   cyclic_prefix_length=cfg.cyclic_prefix_len,
                                   num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                                   dc_null=False,
                                   pilot_pattern="kronecker",
                                   pilot_ofdm_symbol_indices=[2, 11])

        # Adjust guard subcarriers for different number of streams
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

        # Update number of data bits and LDPC codewords
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

    def call(self, dmimo_chans: dMIMOChannels, info_bits):
        """
        Signal processing for baseline one transmission cycle

        :param dmimo_chans: dMIMO channels
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

        if self.cfg.perfect_csi is True:
            # Perfect channel estimation
            h_freq_csi, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                             batch_size=self.batch_size)
        elif self.cfg.csi_prediction is True:
            rc_predictor = standard_rc_pred_freq_mimo('baseline')
            # Get CSI history
            # TODO: optimize channel estimation and optimization procedures (currently very slow)
            h_freq_csi_history = rc_predictor.get_csi_history(self.cfg.first_slot_idx, self.cfg.csi_delay,
                                                                self.rg_csi, dmimo_chans)
            # Do channel prediction
            h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
            _, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay, batch_size=self.batch_size)
        else:
            # LMMSE channel estimation
            h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, self.rg_csi,
                                                               slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                               cfo_sigma=self.cfo_sigma, sto_sigma=self.sto_sigma)
            _, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay, batch_size=self.batch_size)

        if self.cfg.return_estimated_channel:
            return h_freq_csi, rx_snr_db

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

        # Hard-decision for uncoded bits
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

        if self.cfg.rank_adapt and self.cfg.link_adapt:
            do_rank_link_adaptation(self.cfg, h_freq_csi, rx_snr_db, self.cfg.first_slot_idx)

        return dec_bits, uncoded_ber, uncoded_ser, x_hat


def do_rank_link_adaptation(cfg, h_est=None, rx_snr_db=None, start_slot_idx=None, baseline=None, dmimo_chans=None, info_bits=None):

    assert cfg.start_slot_idx >= cfg.csi_delay

    if start_slot_idx == None:
        cfg.first_slot_idx = cfg.start_slot_idx
    else:
        cfg.first_slot_idx = start_slot_idx

    if np.any(h_est == None) or np.any(rx_snr_db == None):
        
        cfg.return_estimated_channel = True
        h_est, rx_snr_db = baseline(dmimo_chans, info_bits)
        cfg.return_estimated_channel = False

    network_config = NetworkConfig()

    # Rank adaptation test
    rank_adaptation = rankAdaptation(network_config.num_bs_ant, network_config.num_ue_ant, architecture='SU-MIMO',
                                        snrdb=rx_snr_db, fft_size=cfg.fft_size, precoder='SVD')

    rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')

    if rank_adaptation.use_mmse_eesm_method:
        rank = rank_feedback_report[0]
        rate = rank_feedback_report[1]

        cfg.num_tx_streams = int(rank)
        
        print("\n", "rank (baseline) = ", rank, "\n")
        print("\n", "rate (baseline) = ", rate, "\n")

    else:
        rank = rank_feedback_report
        rate = []

        cfg.num_tx_streams = int(rank)

        print("\n", "rank (baseline) = ", rank, "\n")

    # Link adaptation test
    data_sym_position = np.arange(0, 14)
    link_adaptation = linkAdaptation(network_config.num_bs_ant, network_config.num_ue_ant, architecture='SU-MIMO',
                                        snrdb=rx_snr_db, nfft=cfg.fft_size, N_s=rank, data_sym_position=data_sym_position, lookup_table_size='long')
    
    mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO')

    if link_adaptation.use_mmse_eesm_method:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = mcs_feedback_report[1]

        cfg.modulation_order = int(np.min(qam_order_arr))
        cfg.code_rate = np.min(code_rate_arr)

        print("\n", "Bits per stream (baseline) = ", cfg.modulation_order, "\n")
        print("\n", "Code-rate per stream (baseline) = ", cfg.code_rate, "\n")
    else:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = []

        cfg.modulation_order = int(np.min(qam_order_arr))


        print("\n", "Bits per stream (SU-MIMO) = ", cfg.modulation_order, "\n")
    
    return rank, rate, qam_order_arr, code_rate_arr

def sim_baseline(cfg: SimConfig):
    """
    Simulation of baseline scenarios using 4x4 MIMO channels

    :param cfg: simulation settings
    :return: [uncoded BER, LDPC BER], [goodput, throughput]
    """

    # dMIMO channels from ns-3 simulator
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3cfg, "Baseline", add_noise=True)

    # Create Baseline simulation
    baseline = Baseline(cfg)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, baseline.num_bits_per_frame])

    # Initial rank and link adaptation
    if cfg.rank_adapt and cfg.link_adapt and cfg.first_slot_idx == cfg.start_slot_idx:
        do_rank_link_adaptation(cfg, baseline=baseline, dmimo_chans=dmimo_chans, info_bits=info_bits)
    else:
        cfg.return_estimated_channel = False

    # Baseline transmission
    dec_bits, uncoded_ber, uncoded_ser, x_hat = baseline(dmimo_chans, info_bits)
    ranks = int(cfg.num_tx_streams)

    # Update error statistics
    info_bits = tf.reshape(info_bits, dec_bits.shape)
    coded_ber = compute_ber(info_bits, dec_bits).numpy()
    coded_bler = compute_bler(info_bits, dec_bits).numpy()

    # Goodput and throughput estimation
    goodbits = (1.0 - coded_ber) * baseline.num_bits_per_frame
    userbits = (1.0 - coded_bler) * baseline.num_bits_per_frame
    ratedbits= (1.0 - uncoded_ser) * baseline.num_uncoded_bits_per_frame

    return [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits], [ranks]


def sim_baseline_all(cfg: SimConfig):
    """"
    Simulation of baseline scenario (BS-to-BS)
    """

    total_cycles = 0
    uncoded_ber, ldpc_ber, goodput, throughput, bitrate = 0, 0, 0, 0, 0

    ranks_list = []
    ldpc_ber_list = []
    uncoded_ber_list = []

    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p2):
        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        bers, bits, additional_KPIs = sim_baseline(cfg)

        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        uncoded_ber_list.append(bers[0])
        ldpc_ber_list.append(bers[1])

        goodput += bits[0]
        throughput += bits[1]
        bitrate += bits[2]

        ranks_list.append(additional_KPIs[0])

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    goodput = goodput / (total_cycles * slot_time * 1e6)  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6)  # Mbps
    bitrate = bitrate / (total_cycles * slot_time * 1e6)  # Mbps

    try:
        ranks = np.concatenate(ranks_list)
    except:
        ranks = np.asarray(ranks_list)

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput, bitrate, ranks, ldpc_ber_list, uncoded_ber_list]
