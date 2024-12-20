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
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, rankAdaptation, linkAdaptation
from dmimo.mimo import update_node_selection
from dmimo.utils import add_frequency_offset, add_timing_offset, cfo_val, sto_val, compute_UE_wise_BER, compute_UE_wise_SER

from .txs_mimo import TxSquad
from .rxs_mimo import RxSquad


class MU_MIMO(Model):

    def __init__(self, cfg: SimConfig, **kwargs):
        """
        Create MU-MIMO simulation object

        :param cfg: simulation settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2

        # CFO and STO settings
        self.sto_sigma = sto_val(cfg, cfg.sto_sigma)
        self.cfo_sigma = cfo_val(cfg, cfg.cfo_sigma)

        # To use sionna-compatible interface, regard TxSquad as one BS transmitter
        # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
        self.num_streams_per_tx = cfg.num_tx_streams

        self.num_txs_ant = 2 * cfg.num_tx_ue_sel + 4  # gNB always present with 4 antennas
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
        rx_tx_association = np.ones((self.num_rx_ue, 1))

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        # Adjust guard subcarriers for channel estimation grid
        csi_effective_subcarriers = (cfg.fft_size // self.num_txs_ant) * self.num_txs_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

        # Resource grid for channel estimation
        self.rg_csi = ResourceGrid(num_ofdm_symbols=14,
                                   fft_size=cfg.fft_size,
                                   subcarrier_spacing=cfg.subcarrier_spacing,
                                   num_tx=1,
                                   num_streams_per_tx=self.num_txs_ant,
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

        # The zero forcing and block diagonalization precoder
        self.bd_precoder = BDPrecoder(self.rg, sm, return_effective_channel=True)
        self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)
        self.bd_equalizer = BDEqualizer(self.rg, sm)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(self.rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order)

        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    def call(self, dmimo_chans: dMIMOChannels, info_bits=None):
        """
        Signal processing for one MU-MIMO transmission cycle (P2)

        :param dmimo_chans: dMIMO channels
        :param info_bits: information bits
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        if not self.cfg.return_estimated_channel:
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
            rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2)
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

        h_freq_csi_all = h_freq_csi

        # [batch_size, num_rx, num_rxs_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        h_freq_csi = h_freq_csi[:, :, :self.num_rxs_ant, :, :, :, :]

        # [batch_size, num_rx_ue, num_ue_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        h_freq_csi =tf.reshape(h_freq_csi, (-1, self.num_rx_ue, self.num_ue_ant, *h_freq_csi.shape[3:]))

        # apply precoding to OFDM grids
        if self.cfg.precoding_method == "ZF":
            x_precoded, g = self.zf_precoder([x_rg, h_freq_csi])
        elif self.cfg.precoding_method == "BD":
            x_precoded, g = self.bd_precoder([x_rg, h_freq_csi, self.cfg.ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "None":
            if self.cfg.ue_ranks[0] == 2:
                x_precoded = x_rg
            elif self.cfg.ue_ranks[0] == 1:
                x_precoded = tf.repeat(x_rg, repeats=2, axis=2)
            else:
                ValueError("unsupported number of streams")
        else:
            ValueError("unsupported precoding method")

        # add CFO/STO to simulate synchronization errors
        if self.sto_sigma > 0:
            x_precoded = add_timing_offset(x_precoded, self.sto_sigma)
        if self.cfo_sigma > 0:
            x_precoded = add_frequency_offset(x_precoded, self.cfo_sigma)

        # apply dMIMO channels to the resource grid in the frequency domain.
        y = dmimo_chans([x_precoded, self.cfg.first_slot_idx]) # Shape: [nbatches, num_rxs_antennas/2, 2, number of OFDM symbols, number of total subcarriers]

        # SINR calculation
        if self.cfg.precoding_method != "None":
            
            sinr_calculation = True
            sinr_dB_arr = np.zeros((self.cfg.num_rx_ue_sel+1, self.batch_size))
            dmimo_chans._add_noise = False

            num_UE_Ant = 2
            num_BS_Ant = 4

            for node_idx in range(self.cfg.num_rx_ue_sel+1):

                x_precoded_sinr, _ = self.zf_precoder([x_rg, h_freq_csi, self.cfg.ue_ranks[0]])
                y_sinr = dmimo_chans([x_precoded_sinr, self.cfg.first_slot_idx])
                
                if node_idx == 0:

                    if self.cfg.precoding_method == "ZF":
                        num_BS_streams = self.cfg.ue_ranks[0]*2
                        zeros_slice = tf.zeros_like(x_rg[:, :, num_BS_streams:, ...])
                        x_rg_tmp = tf.concat([x_rg[:, :, :num_BS_streams, ...], zeros_slice], axis=2)
                        x_precoded_tmp, _ = self.zf_precoder([x_rg_tmp, h_freq_csi, self.cfg.ue_ranks[0]])
                    else:
                        ValueError("unsupported precoding method for SINR calculation")
                    
                    if self.cfg.ue_ranks[0] == 1:
                        ant_indices = np.arange(0, num_BS_Ant,2)
                    elif self.cfg.ue_ranks[0] ==2:
                        ant_indices = np.arange(num_BS_Ant)

                    sig_all = tf.gather(y_sinr, ant_indices, axis=-3)
                    sig_intended = dmimo_chans([x_precoded_tmp, self.cfg.first_slot_idx])
                    sig_intended = tf.gather(sig_intended, ant_indices, axis=-3)
                    sig_pow = tf.reduce_sum(tf.square(tf.abs(sig_intended)), axis=[1, 2, 3, 4])
                    interf_pow = tf.reduce_sum(tf.square(tf.abs(sig_all - sig_intended)), axis=[1, 2, 3, 4])
                    rx_snr_linear = 10**(np.mean(rx_snr_db[:,:,:4,:], axis=(1,2,3)) / 10)
                    noise_pow = sig_pow / rx_snr_linear
                    sinr_linear = sig_pow / (interf_pow + noise_pow)
                    sinr_dB_arr[node_idx, :] = 10*np.log10(sinr_linear)

                    print("BS sinr_dB: ", sinr_dB_arr[node_idx, :], "\n")

                else:

                    if self.cfg.ue_ranks[0] == 1:
                        ant_indices = np.arange((node_idx-1)*num_UE_Ant  + num_BS_Ant, node_idx*num_UE_Ant + num_BS_Ant, 2)
                    elif self.cfg.ue_ranks[0] ==2:
                        ant_indices = np.arange((node_idx-1)*num_UE_Ant  + num_BS_Ant, node_idx*num_UE_Ant + num_BS_Ant)

                    
                    stream_indices = np.arange(self.cfg.ue_ranks[0]*2 + (node_idx-1)*self.cfg.ue_ranks[0], self.cfg.ue_ranks[0]*2 + node_idx*self.cfg.ue_ranks[0])

                    if self.cfg.precoding_method == "ZF":
                        mask = tf.reduce_any(tf.equal(tf.range(x_rg.shape[2])[..., tf.newaxis], stream_indices), axis=-1)
                        mask = tf.cast(mask, x_rg.dtype)
                        mask = tf.reshape(mask, [1, 1, -1, 1, 1])
                        x_rg_tmp = x_rg * mask
                        x_precoded_tmp, _ = self.zf_precoder([x_rg_tmp, h_freq_csi, self.cfg.ue_ranks[0]])
                    else:
                        ValueError("unsupported precoding method for SINR calculation")
                    
                    sig_all = tf.gather(y_sinr, ant_indices, axis=2)
                    sig_intended = dmimo_chans([x_precoded_tmp, self.cfg.first_slot_idx])
                    sig_intended = tf.gather(sig_intended, ant_indices, axis=2)
                    sig_pow = tf.reduce_sum(tf.square(tf.abs(sig_intended)), axis=[1, 2, 3, 4])
                    interf_pow = tf.reduce_sum(tf.square(tf.abs(sig_all - sig_intended)), axis=[1, 2, 3, 4])
                    rx_snr_linear = 10**(np.mean(rx_snr_db[:,:,ant_indices,:], axis=(1,2,3)) / 10)
                    noise_pow = sig_pow / rx_snr_linear
                    sinr_linear = sig_pow / (interf_pow + noise_pow)

                    if tf.reduce_any(tf.equal(sinr_linear, 0)):
                        sinr_linear = rx_snr_linear / 2
                    
                    sinr_dB_arr[node_idx, :] = 10*np.log10(sinr_linear)
                    
                    print("UE ", node_idx-1, " sinr_dB: ", sinr_dB_arr[node_idx, :], "\n")

                    
                    if (sinr_dB_arr[node_idx, :] == np.inf).any():
                        hold = 1

        else:
            sinr_dB_arr = None

        # make proper shape
        y = y[:, :, :self.num_rxs_ant, :, :]
        y = tf.reshape(y, (self.batch_size, self.num_rx_ue, self.num_ue_ant, 14, -1))

        if self.cfg.precoding_method == "BD":
            y = self.bd_equalizer([y, h_freq_csi, self.cfg.ue_indices, self.cfg.ue_ranks])

        # LS channel estimation with linear interpolation
        no = 0.1  # initial noise estimation (tunable param)
        h_hat, err_var = self.ls_estimator([y, no])

        # LMMSE equalization
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no]) # Shape: [nbatches, 1, number of streams, number of effective subcarriers * number of data OFDM symbols]

        # Soft-output QAM demapper
        llr = self.demapper([x_hat, no_eff])

        # Hard-decision bit error rate
        d_hard = tf.cast(llr > 0, tf.float32) # Shape: [nbatches, 1, number of streams, number of effective subcarriers * number of data OFDM symbols * QAM order]
        uncoded_ber = compute_ber(d, d_hard).numpy()

        # Hard-decision symbol error rate
        x_hard = self.mapper(d_hard)
        uncoded_ser = np.count_nonzero(x - x_hard) / np.prod(x.shape)
        num_tx_streams_per_node = int(self.cfg.num_tx_streams/(self.cfg.num_rx_ue_sel+2))
        node_wise_uncoded_ser = compute_UE_wise_SER(x ,x_hard, num_tx_streams_per_node, self.cfg.num_tx_streams)

        # LLR deinterleaver for LDPC decoding
        llr = self.dintlvr(llr)
        llr = tf.reshape(llr, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords, self.encoder.n])

        # LDPC hard-decision decoding
        dec_bits = self.decoder(llr) # Shape: [nbatches, 1, number of streams, 1, number of effective subcarriers * number of data OFDM symbols * QAM order * code rate]

        if self.cfg.rank_adapt and self.cfg.link_adapt:
            # h_freq_csi_reshaped = tf.reshape(h_freq_csi, shape_tmp)
            do_rank_link_adaptation(self.cfg, dmimo_chans, h_freq_csi_all, rx_snr_db, self.cfg.first_slot_idx)

        return dec_bits, uncoded_ber, uncoded_ser, node_wise_uncoded_ser, x_hat, sinr_dB_arr


def do_rank_link_adaptation(cfg, dmimo_chans, h_est=None, rx_snr_db=None, start_slot_idx=None, mu_mimo=None):

    assert cfg.start_slot_idx >= cfg.csi_delay

    if start_slot_idx == None:
        cfg.first_slot_idx = cfg.start_slot_idx
    else:
        cfg.first_slot_idx = start_slot_idx

    if np.any(h_est == None) or np.any(rx_snr_db == None):
        
        cfg.return_estimated_channel = True
        h_est, rx_snr_db = mu_mimo(dmimo_chans)
        cfg.return_estimated_channel = False

    # Rank adaptation
    rank_adaptation = rankAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant, architecture='MU-MIMO',
                                        snrdb=rx_snr_db, fft_size=cfg.fft_size, precoder='BD')

    rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')

    if rank_adaptation.use_mmse_eesm_method:
        rank = rank_feedback_report[0]
        rate = rank_feedback_report[1]

        cfg.num_tx_streams = int(rank)*(cfg.num_rx_ue_sel+2)
        cfg.ue_ranks = [rank]

        # cfg.num_rx_ue_sel = (cfg.num_tx_streams - 4) // 2
        # cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
        
        # print("\n", "rank per user (MU-MIMO) = ", rank, "\n")
        # print("\n", "rate per user (MU-MIMO) = ", rate, "\n")

    else:
        rank = rank_feedback_report
        rate = []

        cfg.num_tx_streams = int(rank)

        # print("\n", "rank per user (MU-MIMO) = ", rank, "\n")

    # Link adaptation
    data_sym_position = np.arange(0, 14)
    link_adaptation = linkAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant, architecture='MU-MIMO',
                                        snrdb=rx_snr_db, nfft=cfg.fft_size, N_s=rank, data_sym_position=data_sym_position, lookup_table_size='short')
    
    mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO')

    if link_adaptation.use_mmse_eesm_method:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = mcs_feedback_report[1]

        # Majority vote for MCS selection for now
        values, counts = np.unique(qam_order_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.modulation_order = int(most_frequent_value)

        values, counts = np.unique(code_rate_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.code_rate = most_frequent_value

        # print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order, "\n")
        # print("\n", "Code-rate per stream per user (MU-MIMO) = ", cfg.code_rate, "\n")
    else:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = []

        cfg.modulation_order = int(np.min(qam_order_arr))

        # print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order, "\n")
    
    return rank, rate, qam_order_arr, code_rate_arr


def sim_mu_mimo(cfg: SimConfig):
    """
    Simulation of MU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits]
    """

    # dMIMO channels from ns-3 simulator
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True)

    # UE selection
    if cfg.enable_ue_selection is True:
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg)
        ns3cfg.update_ue_mask(tx_ue_mask, rx_ue_mask)

    # Initial rank and link adaptation
    mu_mimo_tmp = MU_MIMO(cfg)
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, mu_mimo_tmp.num_bits_per_frame])
    ranks_list = []
    if cfg.rank_adapt and cfg.link_adapt and cfg.first_slot_idx == cfg.start_slot_idx:
        do_rank_link_adaptation(cfg, dmimo_chans=dmimo_chans, mu_mimo=mu_mimo_tmp)

    # Create MU-MIMO simulation
    mu_mimo = MU_MIMO(cfg)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, mu_mimo.num_bits_per_frame])

    # TxSquad transmission (P1)
    if cfg.enable_txsquad is True:
        tx_squad = TxSquad(cfg, mu_mimo.num_bits_per_frame)
        txs_chans = dMIMOChannels(ns3cfg, "TxSquad", add_noise=True)
        info_bits_new, txs_ber, txs_bler = tx_squad(txs_chans, info_bits)
        # print("BER: {}  BLER: {}".format(txs_ber, txs_bler))
        assert txs_ber <= 1e-3, "TxSquad transmission BER too high"

    # MU-MIMO transmission (P2)
    dec_bits, uncoded_ber, uncoded_ser,node_wise_uncoded_ser, x_hat, sinr_dB_arr = mu_mimo(dmimo_chans, info_bits)
    ranks_list.append(int(cfg.num_tx_streams / (cfg.num_rx_ue_sel+2)))

    # Update average error statistics
    info_bits = tf.reshape(info_bits, dec_bits.shape)
    coded_ber = compute_ber(info_bits, dec_bits).numpy()
    coded_bler = compute_bler(info_bits, dec_bits).numpy()

    # Update per-node error statistics
    num_tx_streams_per_node = int(cfg.num_tx_streams/(cfg.num_rx_ue_sel+2))
    node_wise_ber, node_wise_bler = compute_UE_wise_BER(info_bits, dec_bits, num_tx_streams_per_node, cfg.num_tx_streams)
    

    # RxSquad transmission (P3)
    if cfg.enable_rxsquad is True:
        rxcfg = cfg.clone()
        rxcfg.csi_delay = 0
        rxcfg.perfect_csi = True
        rx_squad = RxSquad(rxcfg, mu_mimo.num_bits_per_frame)
        # print("RxSquad using modulation order {} for {} streams / {}".format(
        #     rx_squad.num_bits_per_symbol, mu_mimo.num_streams_per_tx, mu_mimo.mapper.constellation.num_bits_per_symbol))
        rxscfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
        rxs_chans = dMIMOChannels(rxscfg, "RxSquad", add_noise=True)
        received_bits, rxs_ber, rxs_bler, rxs_ber_max, rxs_bler_max = rx_squad(rxs_chans, dec_bits)
        print("BER: {}  BLER: {}".format(rxs_ber, rxs_bler))
        assert rxs_ber <= 1e-3 and rxs_ber_max <= 1e-2, "RxSquad transmission BER too high"

    # Goodput and throughput estimation
    goodbits = (1.0 - coded_ber) * mu_mimo.num_bits_per_frame
    userbits = (1.0 - coded_bler) * mu_mimo.num_bits_per_frame
    ratedbits = (1.0 - uncoded_ser) * mu_mimo.num_uncoded_bits_per_frame

    node_wise_goodbits = (1.0 - node_wise_ber) * mu_mimo.num_bits_per_frame / (cfg.num_rx_ue_sel + 1)
    node_wise_userbits = (1.0 - node_wise_bler) * mu_mimo.num_bits_per_frame / (cfg.num_rx_ue_sel + 1)
    node_wise_ratedbits = (1.0 - node_wise_uncoded_ser) * mu_mimo.num_bits_per_frame / (cfg.num_rx_ue_sel + 1)

    return [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits], [node_wise_goodbits, node_wise_userbits, node_wise_ratedbits, ranks_list, sinr_dB_arr]


def sim_mu_mimo_all(cfg: SimConfig):
    """"
    Simulation of MU-MIMO scenario according to the frame structure
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
        bers, bits, additional_KPIs = sim_mu_mimo(cfg)
        
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
    bitrate = bitrate / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    nodewise_goodput = np.concatenate(nodewise_goodput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_throughput = np.concatenate(nodewise_throughput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_bitrate = np.concatenate(nodewise_bitrate) / (slot_time * 1e6) * overhead  # Mbps
    ranks = np.concatenate(ranks_list)
    if sinr_dB_list[0] is not None:
        sinr_dB = np.concatenate(sinr_dB_list)
    else:
        sinr_dB = None

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput, bitrate, nodewise_goodput, nodewise_throughput, nodewise_bitrate, ranks, uncoded_ber_list, ldpc_ber_list, sinr_dB]
