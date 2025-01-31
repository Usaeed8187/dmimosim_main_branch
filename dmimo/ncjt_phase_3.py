import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from sionna.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper,  LSChannelEstimator, LMMSEEqualizer, MMSEPICDetector, RemoveNulledSubcarriers
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
from dmimo.channel import LMMSELinearInterp
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, SLNRPrecoder, SLNREqualizer, SICLMMSEEqualizer
from dmimo.mimo import update_node_selection
from dmimo.utils import add_frequency_offset, add_timing_offset
from dmimo.ncjt_demo_branch import MC_NCJT_RxUE

from .txs_mimo import TxSquad
from .rxs_mimo import RxSquad


class NCJT_phase_3(Model):

    def __init__(self, cfg: SimConfig, rg_csi: ResourceGrid, **kwargs):
        """
        Create MU-MIMO simulation object

        :param cfg: simulation settings
        :param rg_csi: Resource grid for CSI estimation
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.rg_csi = rg_csi
        self.batch_size = cfg.num_slots_p1  # batch processing for all slots in phase 2

        # To use sionna-compatible interface, regard TxSquad as one BS transmitter
        # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
        self.num_streams_per_tx = cfg.num_tx_streams // cfg.num_scheduled_rx_ue

        self.num_ue_ant = 2  # assuming 2 antennas per UE for reshaping data/channels
        if cfg.ue_indices is None:
            # no rank/link adaptation
            self.num_rxs_ant = self.num_streams_per_tx
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
        else:
            # rank adaptation support
            self.num_rxs_ant = np.sum([len(val) for val in cfg.ue_indices])
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
            if cfg.ue_ranks is None:
                cfg.ue_ranks = self.num_ue_ant  # no rank adaptation

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.ones((1, cfg.num_scheduled_rx_ue))

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
                               num_tx=cfg.num_scheduled_rx_ue,
                               num_streams_per_tx=self.num_streams_per_tx,
                               cyclic_prefix_length=64,
                               num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                               dc_null=cfg.dc_null,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=[2, 11])
        
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)

        # Update number of data bits and LDPC params
        self.ldpc_n = int(2 * self.rg.num_data_symbols)  # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * cfg.code_rate)  # Number of information bits
        self.num_codewords = cfg.modulation_order // 2  # number of codewords per frame
        self.num_bits_per_frame = self.ldpc_k * self.num_codewords * self.num_streams_per_tx  * self.cfg.num_scheduled_rx_ue
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

        # The detector produces soft-decision output for all coded bits
        self.detector = MMSEPICDetector("bit", self.rg, sm, constellation_type="qam",
                                        num_bits_per_symbol=cfg.modulation_order,
                                        num_iter=2, hard_out=False)
        
        # The SIC+LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.sic_lmmse_equ = SICLMMSEEqualizer(self.rg, sm, cfg.modulation_order)
        
        # Using LMMSE CE from NCJT Demo code
        self.ncjt_rx = MC_NCJT_RxUE(self.cfg, batch_size=self.batch_size , modulation_order_list=[self.cfg.modulation_order])

    def call(self, p3_chans_ul, h_freq_csi_dl, h_freq_csi_ul, err_var_csi_ul, info_bits):
        """
        Signal processing for one MU-MIMO transmission cycle (P2)

        :param dmimo_chans: dMIMO channels
        :param h_freq_csi: CSI feedback for precoding
        :param info_bits: information bits
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        # LDPC encoder processing
        info_bits = tf.reshape(info_bits, [self.batch_size, self.cfg.num_scheduled_rx_ue, self.rg.num_streams_per_tx,
                                           self.num_codewords, self.encoder.k])
        c = self.encoder(info_bits)
        c = tf.reshape(c, [self.batch_size, self.cfg.num_scheduled_rx_ue, self.rg.num_streams_per_tx, self.num_codewords * self.encoder.n])

        # Interleaving for coded bits
        d = self.intlvr(c)

        # QAM mapping for the OFDM grid
        x = self.mapper(d)
        x_rg = self.rg_mapper(x)

        # # [batch_size, num_rx, num_rxs_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        # h_freq_csi = h_freq_csi[:, :, :self.num_rxs_ant, :, :, :, :]

        # # [batch_size, num_rx_ue, num_ue_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        # h_freq_csi = tf.reshape(h_freq_csi, (-1, self.num_rx_ue, self.num_ue_ant, *h_freq_csi.shape[3:]))

        # apply precoding to OFDM grids
        if self.cfg.precoding_method == "ZF":
            x_precoded, g = self.zf_precoder([x_rg, h_freq_csi_dl, self.cfg.ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "BD":
            x_precoded, g = self.bd_precoder([x_rg, h_freq_csi_dl, self.cfg.ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "SLNR":
            nvar = 5e-2  # TODO optimize value
            x_precoded, g = self.slnr_precoder([x_rg, h_freq_csi_dl, nvar, self.cfg.ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "none":
            x_rg_shape = x_rg.shape
            padding_shape = tf.tensor_scatter_nd_update(x_rg_shape, [[2]], [h_freq_csi_dl.shape[-3] - x_rg.shape[2]])
            padding = tf.zeros(padding_shape, dtype=x_rg.dtype)
            x_precoded = tf.concat([x_rg, padding], axis=2)
        else:
            ValueError("unsupported precoding method")

        # add CFO/STO to simulate synchronization errors
        if np.any(np.not_equal(self.cfg.random_sto_vals, 0)):
            x_precoded = add_timing_offset(x_precoded, self.cfg.random_sto_vals)
        if np.any(np.not_equal(self.cfg.random_cfo_vals, 0)):
            x_precoded = add_frequency_offset(x_precoded, self.cfg.random_cfo_vals)

        # apply dMIMO channels to the resource grid in the frequency domain.
        y, _ = p3_chans_ul([x_precoded, self.cfg.first_slot_idx])

        if self.cfg.precoding_method == "BD":
            y = self.bd_equalizer([y, h_freq_csi_dl, self.cfg.ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "SLNR":
            y = self.slnr_equalizer([y, h_freq_csi_dl, nvar, self.cfg.ue_indices, self.cfg.ue_ranks])

        no = 5e-2  # initial noise estimation (tunable param)

        # Reshaping
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_effective_subcarriers]
        h_freq_csi_ul = tf.gather(h_freq_csi_ul, tf.range(self.cfg.num_scheduled_rx_ue * 2), axis=4)
        h_freq_csi_ul = tf.reshape(h_freq_csi_ul, tf.concat([h_freq_csi_ul.shape[:3], [-1], [self.cfg.num_scheduled_rx_ue], h_freq_csi_ul.shape[5:]], axis=0))
        h_freq_csi_ul = self.remove_nulled_scs(h_freq_csi_ul)
        
        # [batch_size, num_rx, 1, num_tx, num_tx_ant, num_ofdm_symbols, num_effective_subcarriers]
        _, err_var = self.ls_estimator([y, no])
        err_var = tf.reshape(err_var, tf.concat([err_var.shape[:3], [self.cfg.num_scheduled_rx_ue], [-1], err_var.shape[5:]], axis=0))

        if self.cfg.receiver == 'PIC':

            # PIC Detector
            prior = tf.zeros((self.batch_size, x_rg.shape[2] // self.cfg.num_scheduled_rx_ue, self.cfg.num_scheduled_rx_ue, h_freq_csi_ul.shape[-1] * self.cfg.modulation_order * (self.rg.num_ofdm_symbols - 2)))
            det_out = self.detector((y, h_freq_csi_ul, prior, err_var, no))
            
            # Hard-decision bit error rate
            d_hard = tf.cast(det_out > 0, tf.float32)

        elif self.cfg.receiver == 'LMMSE':

            # LMMSE equalization
            x_hat, no_eff = self.lmmse_equ([y, h_freq_csi_ul, err_var, no])
            
            # Soft-output QAM demapper
            llr = self.demapper([x_hat, no_eff])

            # Hard-decision bit error rate
            d_hard = tf.cast(llr > 0, tf.float32)       
        
        elif self.cfg.receiver == 'SIC':
            
            # SIC+LMMSE equalization
            x_hat, no_eff = self.sic_lmmse_equ([y, h_freq_csi_ul, err_var, no, self.num_streams_per_tx])

            # Soft-output QAM demapper
            llr = self.demapper([x_hat, no_eff])
            llr = tf.gather(llr, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13], axis=-2)
            llr = tf.reshape(llr, tf.concat([x_hat.shape[0:3], [-1]], axis=0))


            # Hard-decision bit error rate
            d_hard = tf.cast(llr > 0, tf.float32)
        
        uncoded_ber = compute_ber(d, d_hard).numpy()

        per_stream_ber = np.zeros((self.cfg.num_rx_ue_sel, self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel))
        for ue_idx in range(self.cfg.num_rx_ue_sel):
            for stream_idx in range(self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel):
                per_stream_ber[ue_idx, stream_idx] = compute_ber(d[:, ue_idx, stream_idx, :], d_hard[:,ue_idx, stream_idx,:]).numpy()

        # Hard-decision symbol error rate
        x_hard = self.mapper(d_hard)
        uncoded_ser = np.count_nonzero(x - x_hard) / np.prod(x.shape)

        # LLR deinterleaver for LDPC decoding
        llr = self.dintlvr(llr)
        llr = tf.reshape(llr, [self.batch_size, self.cfg.num_scheduled_rx_ue, self.rg.num_streams_per_tx, self.num_codewords, self.encoder.n])

        # LDPC hard-decision decoding
        dec_bits = self.decoder(llr)

        print("per_stream_ber = ", per_stream_ber, "\n \n")

        return dec_bits, uncoded_ber, uncoded_ser, x_hat

def ncjt_phase_3(cfg: SimConfig, ns3cfg: Ns3Config):
    """
    Simulation of NCJT Phase 3 scenarios using different settings

    :param cfg: simulation settings
    :param ns3cfg: ns-3 channel settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits]
    """

    # CFO and STO settings
    if cfg.gen_sync_errors:
        cfg.random_sto_vals = cfg.sto_sigma * np.random.normal(size=(ns3cfg.num_rxue_sel, 1))
        cfg.random_cfo_vals = cfg.cfo_sigma * np.random.normal(size=(ns3cfg.num_rxue_sel, 1))

    # Update UE selection
    if cfg.enable_ue_selection is True:
        ns3cfg.reset_ue_selection()
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
        ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)
        cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel) * 2), (ns3cfg.num_rxue_sel, -1))

    # dMIMO channels from ns-3 simulator
    p3_chans_ul = dMIMOChannels(ns3cfg, "RxSquad", forward=True, add_noise=True)
    p3_chans_dl = dMIMOChannels(ns3cfg, "RxSquad", forward=False, add_noise=True)

    # Total number of UE antennas in the RxSquad
    num_txs_ant = 2 * ns3cfg.num_rxue_sel

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
                          dc_null=cfg.dc_null,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

    cfg.num_guard_carriers = rg_csi.num_guard_carriers

    # Channel CSI estimation using channels in previous frames/slots
    if cfg.perfect_csi is True:
        # Perfect channel estimation
        h_freq_csi_dl, rx_snr_db_dl, rx_pwr_dbm_dl = p3_chans_dl.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                                            forward=False,
                                                                            batch_size=cfg.num_slots_p1)                                                                            
    # elif cfg.csi_prediction is True:
    #     rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams)
    #     # Get CSI history
    #     # TODO: optimize channel estimation and optimization procedures (currently very slow)
    #     h_freq_csi_history = rc_predictor.get_csi_history(cfg.first_slot_idx, cfg.csi_delay,
    #                                                       rg_csi, dmimo_chans, 
    #                                                       cfo_vals=cfg.random_cfo_vals,
    #                                                       sto_vals=cfg.random_sto_vals)
    #     # Do channel prediction
    #     h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
    else:
        # LMMSE channel estimation
        h_freq_csi_dl, err_var_csi_dl = lmmse_channel_estimation(p3_chans_dl, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_vals=cfg.random_cfo_vals,
                                                           sto_vals=cfg.random_sto_vals)
        h_freq_csi_ul, err_var_csi_ul = lmmse_channel_estimation(p3_chans_ul, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_vals=cfg.random_cfo_vals,
                                                           sto_vals=cfg.random_sto_vals)

    # Create MU-MIMO simulation
    ncjt_phase_3 = NCJT_phase_3(cfg, rg_csi)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p1, ncjt_phase_3.num_bits_per_frame])

    # Phase 3 NCJT transmission
    dec_bits, uncoded_ber, uncoded_ser = ncjt_phase_3(p3_chans_ul, h_freq_csi_dl, h_freq_csi_ul, err_var_csi_ul, info_bits)

    # Update average error statistics
    info_bits = tf.reshape(info_bits, dec_bits.shape)
    coded_ber = compute_ber(info_bits, dec_bits).numpy()
    coded_bler = compute_bler(info_bits, dec_bits).numpy()

    # Goodput and throughput estimation
    goodbits = (1.0 - coded_ber) * ncjt_phase_3.num_bits_per_frame
    userbits = (1.0 - coded_bler) * ncjt_phase_3.num_bits_per_frame
    ratedbits = (1.0 - uncoded_ser) * ncjt_phase_3.num_uncoded_bits_per_frame

    return [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits]



def sim_ncjt_phase_3_all(cfg: SimConfig, ns3cfg: Ns3Config):
    """"
    Testing of phase 3 receiver using USRP received signal 

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
        bers, bits, additional_KPIs = ncjt_phase_3(cfg, ns3cfg)
        
        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        uncoded_ber_list.append(bers[0])
        ldpc_ber_list.append(bers[1])
        
        goodput += bits[0]
        throughput += bits[1]
        bitrate += bits[2]


    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = 1 # Phase 3 results should not be compared with baseline
    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    bitrate = bitrate / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput, bitrate]