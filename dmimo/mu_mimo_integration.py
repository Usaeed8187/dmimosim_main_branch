import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, flatten_dims
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import Ns3Config, SimConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation
from dmimo.channel import standard_rc_pred_freq_mimo
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, SLNRPrecoder, SLNREqualizer
from dmimo.mimo import rankAdaptation, linkAdaptation
from dmimo.mimo import MUMIMOScheduler
from dmimo.mimo import update_node_selection
from dmimo.utils import add_frequency_offset, add_timing_offset, compute_UE_wise_BER, compute_UE_wise_SER

from .txs_mimo import TxSquad
from .rxs_mimo import RxSquad


class MU_MIMO(Model):

    def __init__(self, cfg: SimConfig, rg_csi: ResourceGrid, **kwargs):
        """
        Create MU-MIMO simulation object

        :param cfg: simulation settings
        :param rg_csi: Resource grid for CSI estimation
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.rg_csi = rg_csi
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2

        # To use sionna-compatible interface, regard TxSquad as one BS transmitter
        # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
        self.num_streams_per_tx = cfg.num_tx_streams

        self.num_ue_ant = 2  # assuming 2 antennas per UE for reshaping data/channels
        if cfg.ue_indices is None:
            # no rank/link adaptation
            self.num_rxs_ant = self.num_streams_per_tx
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
        else:
            # rank adaptation support
            self.num_rxs_ant = np.sum([len(val) for val in cfg.scheduled_rx_ue_indices])
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
            if cfg.ue_ranks is None:
                cfg.ue_ranks = self.num_ue_ant  # no rank adaptation

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.ones((self.num_rx_ue, 1))

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
        self.ldpc_n = int(2 * self.rg.num_data_symbols)  # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * cfg.code_rate)  # Number of information bits
        self.num_codewords = cfg.modulation_order // 2  # number of codewords per frame
        self.num_bits_per_frame = self.ldpc_k * self.num_codewords * self.num_streams_per_tx
        self.num_uncoded_bits_per_frame = self.ldpc_n * self.num_codewords * self.num_streams_per_tx

        # The encoder maps information bits to coded bits
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n)

        # LDPC interleaver
        self.intlvr = RowColumnInterleaver(3072, axis=-1)  # fixed design for current RG config
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", cfg.modulation_order)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(self.rg)

        if self.cfg.precoding_method == "ZF":
            # the zero forcing precoder
            self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)
        elif self.cfg.precoding_method == "BD":
            self.bd_precoder = BDPrecoder(self.rg, sm, return_effective_channel=True)
            self.bd_equalizer = BDEqualizer(self.rg, sm)
        elif self.cfg.precoding_method == "SLNR":
            self.slnr_precoder = SLNRPrecoder(self.rg, sm, return_effective_channel=True)
            self.slnr_equalizer = SLNREqualizer(self.rg, sm)

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
        Signal processing for one MU-MIMO transmission cycle (P2)

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

        # [batch_size, num_rx_ue, num_ue_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        # h_freq_csi = tf.gather(h_freq_csi, tf.reshape(self.cfg.scheduled_rx_ue_indices, (1,-1)), axis=2, batch_dims=1)

        # apply precoding to OFDM grids
        if self.cfg.precoding_method == "ZF":
            x_precoded, g = self.zf_precoder([x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "BD":
            x_precoded, g = self.bd_precoder([x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "SLNR":
            nvar = 5e-2  # TODO optimize value
            x_precoded, g = self.slnr_precoder([x_rg, h_freq_csi, nvar, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        else:
            ValueError("unsupported precoding method")

        # add CFO/STO to simulate synchronization errors
        if np.any(np.not_equal(self.cfg.random_sto_vals, 0)):
            x_precoded = add_timing_offset(x_precoded, self.cfg.random_sto_vals)
        if np.any(np.not_equal(self.cfg.random_cfo_vals, 0)):
            x_precoded = add_frequency_offset(x_precoded, self.cfg.random_cfo_vals)

        # apply dMIMO channels to the resource grid in the frequency domain.
        y, _ = dmimo_chans([x_precoded, self.cfg.first_slot_idx])

        # make proper shape
        # y = y[:, :, :self.num_rxs_ant, :, :]
        y = tf.gather(y, tf.reshape(self.cfg.scheduled_rx_ue_indices, [-1]), axis=2)
        y = tf.reshape(y, (self.batch_size, self.num_rx_ue, self.num_ue_ant, 14, -1))

        if self.cfg.precoding_method == "BD":
            y = self.bd_equalizer([y, h_freq_csi, self.cfg.ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "SLNR":
            y = self.slnr_equalizer([y, h_freq_csi, nvar, self.cfg.ue_indices, self.cfg.ue_ranks])

        # LS channel estimation with linear interpolation
        no = 5e-2  # initial noise estimation (tunable param)
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
        node_wise_uncoded_ser = compute_UE_wise_SER(x ,x_hard, self.cfg.ue_ranks[0], self.cfg.num_tx_streams)

        # LLR deinterleaver for LDPC decoding
        llr = self.dintlvr(llr)
        llr = tf.reshape(llr, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords, self.encoder.n])

        # LDPC hard-decision decoding
        dec_bits = self.decoder(llr)

        sinr_dB_arr = None
        return dec_bits, uncoded_ber, uncoded_ser, x_hat, node_wise_uncoded_ser, sinr_dB_arr


def do_rank_link_adaptation(cfg, dmimo_chans, h_est, rx_snr_db):

    # Rank adaptation
    rank_adaptation = rankAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                     architecture='MU-MIMO', snrdb=rx_snr_db, fft_size=cfg.fft_size,
                                     precoder=cfg.precoding_method)

    rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')
    rank = rank_feedback_report[0]

    h_eff = rank_adaptation.calculate_effective_channel(rank_feedback_report[0], h_est)
    snr_linear = 10**(rx_snr_db/10)
    snr_linear = np.sum(snr_linear, axis=(2))
    snr_linear = np.mean(snr_linear)

    n_var = rank_adaptation.cal_n_var(h_eff, snr_linear)

    # Link adaptation
    data_sym_position = np.arange(0, 14)
    link_adaptation = linkAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                     architecture='MU-MIMO', snrdb=rx_snr_db, nfft=cfg.fft_size,
                                     N_s=rank, data_sym_position=data_sym_position, lookup_table_size='short')

    mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO')

    return rank_feedback_report, n_var, mcs_feedback_report


def sim_mu_mimo(cfg: SimConfig, ns3cfg: Ns3Config):
    """
    Simulation of MU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :param ns3cfg: ns-3 channel settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits]
    """

    # CFO and STO settings
    if cfg.gen_sync_errors:
        cfg.random_sto_vals = cfg.sto_sigma * np.random.normal(size=(ns3cfg.num_txue_sel, 1))
        cfg.random_cfo_vals = cfg.cfo_sigma * np.random.normal(size=(ns3cfg.num_txue_sel, 1))

    # dMIMO channels from ns-3 simulator
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True)

    # Reset UE selection
    ns3cfg.reset_ue_selection()
    tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
    ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)
    # cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))
    
    # if cfg.scheduling:
    # if cfg.enable_ue_selection is True:
    #     ns3cfg.reset_ue_selection()
    #     tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
    #     ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)
    #     cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

    # Total number of antennas in the TxSquad, always use all gNB antennas
    num_txs_ant = 2 * ns3cfg.num_txue_sel + ns3cfg.num_bs_ant

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
        h_freq_csi, rx_snr_db, rx_pwr_dbm = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                                     batch_size=cfg.num_slots_p2)
    elif cfg.csi_prediction is True:
        rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams)
        # Get CSI history
        # TODO: optimize channel estimation and optimization procedures (currently very slow)
        h_freq_csi_history = rc_predictor.get_csi_history(cfg.first_slot_idx, cfg.csi_delay,
                                                          rg_csi, dmimo_chans, 
                                                          cfo_vals=cfg.random_cfo_vals,
                                                          sto_vals=cfg.random_sto_vals)
        # Do channel prediction
        h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
    else:
        # LMMSE channel estimation
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_vals=cfg.random_cfo_vals,
                                                           sto_vals=cfg.random_sto_vals)
    _, rx_snr_db, _ = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                batch_size=cfg.num_slots_p2)


    ######################################################
    # Testing scheduler with no quantization for now
    ######################################################

    if cfg.scheduling:
        # Get effective channel estimate after MRC combining at the RxSquad Nodes
        rank_adaptation = rankAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                        architecture='MU-MIMO', snrdb=rx_snr_db, fft_size=cfg.fft_size,
                                        precoder=cfg.precoding_method)
        _, h_eff = rank_adaptation.generate_zf_precoding(cfg.ue_ranks[0], h_freq_csi) 

        # Get right singular vectors and singular values of effective channels 
        num_rx_nodes = cfg.ue_indices.shape[0]-1
        h_reconstructed = np.zeros((h_freq_csi.shape), dtype=complex)
        h_reconstructed = h_reconstructed[:, :, :h_eff.shape[-2], ...]
        for rx_node_idx in range(num_rx_nodes):

            if rx_node_idx == 0:
                num_streams = cfg.ue_ranks[0]*2
                stream_idx = np.arange(cfg.ue_ranks[0]*2)
            else:
                num_streams = cfg.ue_ranks[0]
                stream_idx = np.arange((rx_node_idx-1) * cfg.ue_ranks[0] + cfg.ue_ranks[0]*2, rx_node_idx * cfg.ue_ranks[0] + cfg.ue_ranks[0]*2)

            curr_h_eff = tf.gather(h_eff, stream_idx, axis=-2)

            s, _, Vh = tf.linalg.svd(curr_h_eff, full_matrices=False)
            s = tf.cast(s, Vh.dtype)
            s = tf.expand_dims(s, axis=-2)

            curr_h_reconstructed = Vh * s
            curr_h_reconstructed = curr_h_reconstructed[:,:,:,tf.newaxis,...]

            h_reconstructed[:, :, stream_idx, ...] = tf.transpose(curr_h_reconstructed, [0, 1, 6, 3, 5, 2, 4])
        
        h_reconstructed = tf.convert_to_tensor(h_reconstructed)

        # Scheduling
        mu_mimo_scheduler = MUMIMOScheduler(rx_snr_db)
        scheduled_rx_nodes = mu_mimo_scheduler(h_reconstructed)
        scheduled_rx_UEs = scheduled_rx_nodes[1:] - 1 # Assuming gNB was scheduled
        
        # Updating system parameters based on scheduling    
        # rx_ue_mask = np.zeros(cfg.num_rx_ue_sel)
        # rx_ue_mask[scheduled_rx_UEs] = 1
        # tx_ue_mask = np.ones(cfg.num_tx_ue_sel)
        # ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

        ue_indices = [[0, 1],[2, 3]] # Assuming gNB was scheduled
        for node in scheduled_rx_nodes[1:]:
            start = (node - 1) * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            end = node * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            ue_indices.append(list(np.arange(start, end)))
        cfg.scheduled_rx_ue_indices = np.array(ue_indices)
        cfg.num_scheduled_ues = scheduled_rx_UEs.size
        cfg.num_tx_streams = (cfg.num_scheduled_ues+2) * cfg.ue_ranks[0]
    else:
        cfg.scheduled_rx_ue_indices = cfg.ue_indices
        cfg.num_scheduled_ues = cfg.scheduled_rx_ue_indices.shape[0]-2
        if not cfg.rank_adapt:
            cfg.num_tx_streams = (cfg.num_scheduled_ues+2) * cfg.ue_ranks[0]
    
    # Rank and link adaptation
    h_freq_csi = tf.gather(h_freq_csi, tf.reshape(cfg.scheduled_rx_ue_indices, (1,-1)), axis=2, batch_dims=1)
    rank_feedback_report, n_var, mcs_feedback_report = \
        do_rank_link_adaptation(cfg, dmimo_chans, h_freq_csi, rx_snr_db)

    if cfg.rank_adapt and cfg.link_adapt:
        # Update rank and total number of streams
        rank = rank_feedback_report[0]
        cfg.ue_ranks = [rank]
        cfg.num_tx_streams = rank * (cfg.num_scheduled_ues + 2)  # treat BS as two UEs

        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = mcs_feedback_report[1]
        values, counts = np.unique(qam_order_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.modulation_order = int(most_frequent_value)

        print("\n", "rank per user (MU-MIMO) = ", rank, "\n")
        # print("\n", "rate per user (MU-MIMO) = ", rate, "\n")

        values, counts = np.unique(code_rate_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.code_rate = most_frequent_value

        print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order, "\n")
        print("\n", "Code-rate per stream per user (MU-MIMO) = ", cfg.code_rate, "\n")
    ranks_out = int(cfg.num_tx_streams / (cfg.num_scheduled_ues+2))

    # Create MU-MIMO simulation
    mu_mimo = MU_MIMO(cfg, rg_csi)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, mu_mimo.num_bits_per_frame])

    # TxSquad transmission (P1)
    if cfg.enable_txsquad is True:
        tx_squad = TxSquad(cfg, ns3cfg, mu_mimo.num_bits_per_frame)
        txs_chans = dMIMOChannels(ns3cfg, "TxSquad", add_noise=True)
        info_bits_new, txs_ber, txs_bler = tx_squad(txs_chans, info_bits)
        # print("BER: {}  BLER: {}".format(txs_ber, txs_bler))
        assert txs_ber <= 1e-3, "TxSquad transmission BER too high"

    # MU-MIMO transmission (P2)
    dec_bits, uncoded_ber, uncoded_ser, x_hat, node_wise_uncoded_ser, sinr_dB_arr = mu_mimo(dmimo_chans, h_freq_csi, info_bits)

    # Update error statistics
    info_bits = tf.reshape(info_bits, dec_bits.shape)
    coded_ber = compute_ber(info_bits, dec_bits).numpy()
    coded_bler = compute_bler(info_bits, dec_bits).numpy()

    node_wise_ber, node_wise_bler = compute_UE_wise_BER(info_bits, dec_bits, cfg.ue_ranks[0], cfg.num_tx_streams)

    # RxSquad transmission (P3)
    if cfg.enable_rxsquad is True:
        rxcfg = cfg.clone()
        rxcfg.csi_delay = 0
        rxcfg.perfect_csi = True
        rx_squad = RxSquad(rxcfg, ns3cfg, mu_mimo.num_bits_per_frame)
        print("RxSquad using modulation order {} for {} streams / {}".format(
            rx_squad.num_bits_per_symbol, mu_mimo.num_streams_per_tx, mu_mimo.mapper.constellation.num_bits_per_symbol))
        rxscfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
        rxs_chans = dMIMOChannels(rxscfg, "RxSquad", add_noise=True)
        received_bits, rxs_ber, rxs_bler, rxs_ber_max, rxs_bler_max = rx_squad(rxs_chans, dec_bits)
        # print("BER: {}  BLER: {}".format(rxs_ber, rxs_bler))
        assert rxs_ber <= 1e-3 and rxs_ber_max <= 1e-2, "RxSquad transmission BER too high"

    # Goodput and throughput estimation
    goodbits = (1.0 - coded_ber) * mu_mimo.num_bits_per_frame
    userbits = (1.0 - coded_bler) * mu_mimo.num_bits_per_frame
    ratedbits = (1.0 - uncoded_ser) * mu_mimo.num_uncoded_bits_per_frame

    node_wise_goodbits = (1.0 - node_wise_ber) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)
    node_wise_userbits = (1.0 - node_wise_bler) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)
    node_wise_ratedbits = (1.0 - node_wise_uncoded_ser) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)

    return [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits], [node_wise_goodbits, node_wise_userbits, node_wise_ratedbits, ranks_out, sinr_dB_arr]


def sim_mu_mimo_all(cfg: SimConfig, ns3cfg: Ns3Config):
    """"
    Simulation of MU-MIMO scenario according to the frame structure

    :param cfg: simulation settings
    :param ns3cfg: ns-3 channel settings
    """

    total_cycles = 0
    uncoded_ber, ldpc_ber, goodput, throughput, bitrate = 0, 0, 0, 0, 0
    nodewise_goodput = []
    nodewise_throughput = []
    nodewise_bitrate = []
    ranks_list = []
    ldpc_ber_list = []
    uncoded_ber_list = []
    sinr_dB_list = []
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        
        print("first_slot_idx: ", first_slot_idx, "\n")

        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        bers, bits, additional_KPIs = sim_mu_mimo(cfg, ns3cfg)
        
        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        uncoded_ber_list.append(bers[0])
        ldpc_ber_list.append(bers[1])
        
        goodput += bits[0]
        throughput += bits[1]
        bitrate += bits[2]
        
        nodewise_goodput.append(additional_KPIs[0])
        nodewise_throughput.append(additional_KPIs[1])
        nodewise_bitrate.append(additional_KPIs[2])
        ranks_list.append(additional_KPIs[3])
        sinr_dB_list.append(additional_KPIs[4])

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = cfg.num_slots_p2/(cfg.num_slots_p1 + cfg.num_slots_p2)
    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    nodewise_goodput = np.concatenate(nodewise_goodput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_throughput = np.concatenate(nodewise_throughput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_bitrate = np.concatenate(nodewise_bitrate) / (slot_time * 1e6) * overhead  # Mbps
    ranks = np.array(ranks_list).flatten()
    if sinr_dB_list[0] is not None:
        sinr_dB = np.concatenate(sinr_dB_list)
    else:
        sinr_dB = None

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput, bitrate, nodewise_goodput, nodewise_throughput, nodewise_bitrate, ranks, uncoded_ber_list, ldpc_ber_list, sinr_dB]
