import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from sionna.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper,  LSChannelEstimator, LMMSEEqualizer, MMSEPICDetector
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


class Phase_3_RX(Model):

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
            self.num_rxs_ant = np.sum([len(val) for val in cfg.ue_indices])
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
            if cfg.ue_ranks is None:
                cfg.ue_ranks = self.num_ue_ant  # no rank adaptation

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        # rx_tx_association = np.ones((self.num_rx_ue, 1))
        bs_ut_association = np.zeros([1, cfg.num_rx_ue_sel])
        bs_ut_association[0, :] = 1
        rx_tx_association = bs_ut_association
        self.num_streams_per_tx = cfg.num_tx_streams // cfg.num_rx_ue_sel

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
                               num_tx=cfg.num_rx_ue_sel,
                               num_streams_per_tx=self.num_streams_per_tx,
                               cyclic_prefix_length=64,
                               num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                               dc_null=cfg.dc_null,
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
        self.rg_demapper = ResourceGridDemapper(self.rg, sm)

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
        self.demapper_hard = Demapper("maxlog", "qam", cfg.modulation_order, hard_out=True)

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

    def call(self, x_rg, y, rx_heltf=None):
        """
        Signal processing for one MU-MIMO transmission cycle (P2)

        :param dmimo_chans: dMIMO channels
        :param h_freq_csi: CSI feedback for precoding
        :param info_bits: information bits
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        # Pre-processing USRP rx data (from phase 3 gNB)
        y = tf.cast(y, dtype=x_rg.dtype)

        #############################################
        # Processing transmit data
        #############################################

        # Initializing LDPC encoding and interleaving (needed for decoding. the actual bits in these lines are not used)
        batch_size = x_rg.shape[0]
        num_codewords = self.cfg.modulation_order // 2
        encoder_k = self.ldpc_k
        info_bits = tf.cast(
                        tf.random.uniform(
                            [batch_size, 1, self.num_streams_per_tx, num_codewords, encoder_k],
                            minval=0,
                            maxval=2,  # Exclusive upper bound
                            dtype=tf.int32,
                        ), dtype=tf.float32)
        c_temp = self.encoder(info_bits)
        c_temp = tf.reshape(c_temp, [batch_size, 1, self.num_streams_per_tx, num_codewords * self.encoder.n])
        d_temp = self.intlvr(c_temp)

        # Reshaping USRP tx data (sent from phase 3 UEs)
        x = self.rg_demapper(x_rg)
        # x = tf.repeat(x, repeats=y.shape[0], axis=0)
        no_eff_x = 1e-5
        llr_tx = self.demapper([x, no_eff_x])
        tx_bits_coded_intlvd = tf.cast(llr_tx > 0, tf.float32)
        # Deinterleaving and LDPC decoding for transmit bits
        llr_tx = self.dintlvr(llr_tx)
        tx_bits = self.decoder(llr_tx)

        #############################################
        # Processing received data
        #############################################

        if self.cfg.lmmse_chest:

            # new shape: (batch_size, num_subcarrier, num_rx_ant, num_ltf)
            he_ltf = tf.squeeze(tf.transpose(rx_heltf, [1, 0, 4, 2, 3]))

            h1_hat = self.ncjt_rx.heltf_channel_estimate(he_ltf[..., 0:2])  # (num_batch, num_subcarrier, num_rx_ant, num_ss)
            h2_hat = self.ncjt_rx.heltf_channel_estimate(he_ltf[..., 2:4])  # (num_batch, num_subcarrier, num_rx_ant, num_ss)
            h_hat = tf.concat((h1_hat, h2_hat), axis=-1)
            # h_hat_averaged = insert_dims(h_hat, 1, axis=2)
            # h_hat_averaged = tf.repeat(h_hat_averaged, len(self.ncjt_rx.data_syms) // 2, axis=2)

            # Channel covariance statistics
            freq_cov = self.ncjt_rx.estimate_freq_cov(h_hat)
            lmmse_int = LMMSELinearInterp(self.rg.pilot_pattern, freq_cov)
            self.lmmse_est = LSChannelEstimator(self.rg, interpolator=lmmse_int)
            no = 5e-3
            h_hat, err_var = self.lmmse_est([y, no])

        else:
            # LS channel estimation with linear interpolation
            no = 1e-5  # initial noise estimation (tunable param)
            h_hat, err_var = self.ls_estimator([y, no])

        debug = True
        if debug:
            sym_idx = 11 
            plt.figure(figsize=(48, 12))
            plot_idx = 1
            for ue_idx in range(self.cfg.num_rx_ue_sel):
                for ue_ant_idx in range(2):
                    for bs_ant_idx in range(4):
                        plt.subplot(self.cfg.num_rx_ue_sel, 4*2, plot_idx)
                        plt.plot(np.real(h_hat[0, 0, bs_ant_idx, ue_idx, ue_ant_idx, sym_idx, :]), label='Real_Tx{}_Rx{}'.format(ue_ant_idx, bs_ant_idx), linewidth=2)
                        plt.title(f'UE_{ue_idx}_UEAnt_{ue_ant_idx}_BSAnt_{bs_ant_idx}', fontsize=18, fontweight='bold')
                        # plt.plot(np.imag(h_hat[0, 0, bs_ant_idx, ue_idx, ue_ant_idx, 0, :]), label='Imag_Tx{}_Rx{}'.format(ue_ant_idx, bs_ant_idx))
                        # plt.plot(np.abs(h_hat[0, 0, bs_ant_idx, ue_idx, ue_ant_idx, 0, :]), label='Abs_Tx{}_Rx{}'.format(ue_ant_idx, bs_ant_idx))
                        plot_idx += 1
            plt.legend()
            plt.suptitle("Channel Real Part for UEs and Antennas", fontsize=24, fontweight='bold', y=1.02)  # Main title
            plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to fit title
            plt.savefig('channel_estimates_all')


            # plt.figure()
            # plt.plot(np.real(h_hat[0,0,0,ue_idx,0,0,:]), label='Real')
            # # plt.plot(np.imag(h_hat[0,0,0,0,0,0,:]), label='Imag')
            # # plt.plot(np.abs(h_hat[0,0,0,0,0,0,:]), label='Abs')
            # plt.legend()
            # plt.title('Antenna_0_0')
            # plt.savefig('Antenna_0_0')

            # plt.figure()
            # plt.plot(np.real(h_hat[0,0,2,ue_idx,1,0,:]), label='Real')
            # # plt.plot(np.imag(h_hat[0,0,2,0,1,0,:]), label='Imag')
            # # plt.plot(np.abs(h_hat[0,0,2,0,1,0,:]), label='Abs')
            # plt.legend()
            # plt.title('Antenna_2_1')
            # plt.savefig('Antenna_2_1')

            # plt.figure()
            # plt.plot(np.real(h_hat[0,0,3,ue_idx,1,0,:]), label='Real')
            # # plt.plot(np.imag(h_hat[0,0,3,0,1,0,:]), label='Imag')
            # # plt.plot(np.abs(h_hat[0,0,3,0,1,0,:]), label='Abs')
            # plt.legend()
            # plt.title('Antenna_3_1')
            # plt.savefig('Antenna_3_1')

            # y_np = y.numpy()
            # x_rg_np = x_rg.numpy()
            # h_ls_np = np.zeros((y_np.shape[2], x_rg_np.shape[2], 2, len(self.rg.effective_subcarrier_ind)), dtype=np.complex64)

            # sym_ids = [2, 11]
            # for sym_idx, sym_id in enumerate(sym_ids):
            #     for rx_id in range(y.shape[2]):
            #         for tx_id in range(x_rg.shape[2]):

            #             h_ls_np[rx_id, tx_id, sym_idx, :] = y_np[0, 0, rx_id, sym_id, self.rg.effective_subcarrier_ind] / x_rg_np[0, 0, tx_id, sym_id, self.rg.effective_subcarrier_ind]
            
            # h_ls = tf.convert_to_tensor(h_ls_np)
            
            # num_ue_ants = 2
            # h_ls_reshaped = np.reshape(h_ls_np, [h_ls_np.shape[0], self.cfg.num_rx_ue_sel, num_ue_ants, len(sym_ids), self.rg.num_effective_subcarriers])
            # for sym_idx, sym_id in enumerate(sym_ids):
            #     for rx_ant_id in range(y.shape[2]):
            #         for tx_node_id in range(self.cfg.num_rx_ue_sel):
            #             for tx_ant_id in range(num_ue_ants):
                            
            #                 curr_h_ls_all = h_ls_reshaped[rx_ant_id, tx_node_id, tx_ant_id, sym_idx, :]
            #                 pilot_indices = np.where(np.isfinite(curr_h_ls_all))[0]
            #                 non_pilot_indices = np.where(~np.isfinite(curr_h_ls_all))[0]

            #                 curr_h_ls_pilots = curr_h_ls_all[pilot_indices]

            #                 interp_real = interp1d(pilot_indices, curr_h_ls_pilots.real, kind='linear', fill_value="extrapolate")
            #                 interp_imag = interp1d(pilot_indices, curr_h_ls_pilots.imag, kind='linear', fill_value="extrapolate")
            #                 interpolated_real = interp_real(non_pilot_indices)
            #                 interpolated_imag = interp_imag(non_pilot_indices)

            #                 h_ls_reshaped[rx_ant_id, tx_node_id, tx_ant_id, sym_idx, non_pilot_indices] = interpolated_real + 1j * interpolated_imag

            # h_ls_np = np.reshape(h_ls_reshaped, [h_ls_reshaped.shape[0], -1, h_ls_reshaped.shape[-2], h_ls_reshaped.shape[-1]])
            
            # plt.figure()
            # plt.plot(np.real(h_ls_np[0,0,0,:]))
            # plt.plot(np.real(h_hat[0,0,0,0,0,2,:]))
            # plt.savefig('a')
        debug = False

        if self.cfg.receiver == 'PIC':

            # PIC Detector
            prior = tf.zeros(llr_tx.shape[1:])
            prior = prior[tf.newaxis, ...]
            det_out = self.detector((y, h_hat, prior, err_var, no))
            
            # Hard-decision bit error rate
            d_hard = tf.cast(det_out > 0, tf.float32)

            # # LLR deinterleaving
            # llr = tf.reshape(det_out, prior.shape)
            # llr = self.dintlvr(llr)

            # # LDPC decoder
            # llr = tf.reshape(llr, [1, 1, self.num_streams_per_tx, num_codewords * self.encoder.n])
            # dec_bits = self.decoder(llr)

            uncoded_ber = 0.5
            start_subframe_idx = 0
            for curr_batch_idx in range(tx_bits_coded_intlvd.shape[0]):                
                curr_ber = compute_ber(tx_bits_coded_intlvd[curr_batch_idx:curr_batch_idx+1, ...], d_hard).numpy()

                if curr_ber < uncoded_ber:
                    uncoded_ber = curr_ber
                    start_subframe_idx = curr_batch_idx

            per_stream_ber = np.zeros((self.cfg.num_rx_ue_sel, self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel))
            for ue_idx in range(self.cfg.num_rx_ue_sel):
                for stream_idx in range(self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel):
                    per_stream_ber[ue_idx, stream_idx] = compute_ber(tx_bits_coded_intlvd[start_subframe_idx:start_subframe_idx+1, ue_idx, stream_idx, :], d_hard[:,ue_idx, stream_idx,:]).numpy()
        
        elif self.cfg.receiver == 'LMMSE':

            # LMMSE equalization
            x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])

            # Soft-output QAM demapper
            llr = self.demapper([x_hat, no_eff])

            # Hard-decision bit error rate
            d_hard = tf.cast(llr > 0, tf.float32)
            
            sc_ind = 6
            sym_ind = 1
            uncoded_ber = 0.5
            start_subframe_idx = 0
            for curr_batch_idx in range(tx_bits_coded_intlvd.shape[0]):                
                curr_ber = compute_ber(tx_bits_coded_intlvd[curr_batch_idx:curr_batch_idx+1, ...], d_hard).numpy()

                # expected_signal = h_hat[curr_batch_idx,0,:,0,:,sym_ind,sc_ind-6] @ x_rg[0,0,:,sym_ind,sc_ind:sc_ind+1]
                # actual_signal = y[curr_batch_idx, 0, :, sym_ind,sc_ind:sc_ind+1]

                if curr_ber < uncoded_ber:
                    uncoded_ber = curr_ber
                    start_subframe_idx = curr_batch_idx
            
            per_stream_ber = np.zeros((self.cfg.num_rx_ue_sel, self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel))
            for ue_idx in range(self.cfg.num_rx_ue_sel):
                for stream_idx in range(self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel):
                    per_stream_ber[ue_idx, stream_idx] = compute_ber(tx_bits_coded_intlvd[start_subframe_idx:start_subframe_idx+1, ue_idx, stream_idx, :], d_hard[:,ue_idx, stream_idx,:]).numpy()


            debug = True
            if debug:
                x_hat_np = x_hat[0,0,0,:].numpy()
                x_real = np.real(x_hat_np)
                x_imag = np.imag(x_hat_np)

                plt.figure()
                plt.scatter(x_real, x_imag, alpha=0.7)
                plt.grid(True)
                plt.savefig('b')
            debug = False
        
        elif self.cfg.receiver == 'SIC':
            
            # LMMSE equalization
            x_hat, no_eff = self.sic_lmmse_equ([y, h_hat, err_var, no, self.num_streams_per_tx])

            # Soft-output QAM demapper
            llr = self.demapper([x_hat, no_eff])

            # Hard-decision bit error rate
            d_hard = tf.cast(llr > 0, tf.float32)

            # Take the data carrying ofdm symbols of d_hard
            data_ofdm_syms = tf.range(self.rg.num_ofdm_symbols)
            mask = ~tf.reduce_any(tf.equal(tf.expand_dims(data_ofdm_syms, 1), self.rg._pilot_ofdm_symbol_indices), axis=1)
            data_ofdm_syms = tf.boolean_mask(data_ofdm_syms, mask)
            d_hard = tf.gather(d_hard, data_ofdm_syms, axis=-2)
            d_hard = tf.reshape(d_hard, (d_hard.shape[0], d_hard.shape[1], d_hard.shape[2], -1))
            
            sc_ind = 6
            sym_ind = 1
            uncoded_ber = 0.5
            start_subframe_idx = 0
            for curr_batch_idx in range(tx_bits_coded_intlvd.shape[0]):                
                curr_ber = compute_ber(tx_bits_coded_intlvd[curr_batch_idx:curr_batch_idx+1, ...], d_hard).numpy()

                # expected_signal = h_hat[curr_batch_idx,0,:,0,:,sym_ind,sc_ind-6] @ x_rg[0,0,:,sym_ind,sc_ind:sc_ind+1]
                # actual_signal = y[curr_batch_idx, 0, :, sym_ind,sc_ind:sc_ind+1]

                if curr_ber < uncoded_ber:
                    uncoded_ber = curr_ber
                    start_subframe_idx = curr_batch_idx
            
            per_stream_ber = np.zeros((self.cfg.num_rx_ue_sel, self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel))
            for ue_idx in range(self.cfg.num_rx_ue_sel):
                for stream_idx in range(self.cfg.num_tx_streams // self.cfg.num_rx_ue_sel):
                    per_stream_ber[ue_idx, stream_idx] = compute_ber(tx_bits_coded_intlvd[start_subframe_idx:start_subframe_idx+1, ue_idx, stream_idx, :], d_hard[:,ue_idx, stream_idx,:]).numpy()


        # Hard-decision symbol error rate
        x_hard = self.mapper(d_hard)
        uncoded_ser = np.count_nonzero(x[start_subframe_idx:start_subframe_idx+1, ...] - x_hard) / np.prod(x_hard.shape)

        print("subframe index = ", start_subframe_idx, "\n", "per_stream_ber = ", per_stream_ber, "\n \n")

        return uncoded_ber, uncoded_ser, per_stream_ber


def test_phase_3_rx(cfg: SimConfig, ns3cfg: Ns3Config, x_rg, y_rg, rx_heltf=None):
    """
    Simulation of MU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :param ns3cfg: ns-3 channel settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits]
    """
    
    # Total number of antennas in the TxSquad, always use all gNB antennas
    num_txs_ant = 2 * ns3cfg.num_rxue_sel

    # Adjust guard subcarriers for channel estimation grid
    # csi_effective_subcarriers = (cfg.fft_size // num_txs_ant) * num_txs_ant
    cfg.num_guard_carriers = [cfg.csi_guard_carriers_1, cfg.csi_guard_carriers_2]
    csi_effective_subcarriers = cfg.fft_size - (cfg.csi_guard_carriers_1 +  cfg.csi_guard_carriers_2 + cfg.dc_null)

    # Resource grid for channel estimation
    rg_csi = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=num_txs_ant,
                          cyclic_prefix_length=cfg.cyclic_prefix_len,
                          num_guard_carriers=cfg.num_guard_carriers,
                          dc_null=cfg.dc_null,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

    # Create MU-MIMO simulation
    phase_3_rx = Phase_3_RX(cfg, rg_csi)

    # MU-MIMO transmission (P2)
    uncoded_ber, uncoded_ser, per_stream_ber = phase_3_rx(x_rg, y_rg, rx_heltf)

    return uncoded_ber, uncoded_ser, per_stream_ber


def test_phase_3_rx_all(cfg: SimConfig, ns3cfg: Ns3Config, x_rg, y_rg, rx_heltf=None):
    """"
    Testing of phase 3 receiver using USRP received signal 

    :param cfg: simulation settings
    :param ns3cfg: ns-3 channel settings
    :param x_rg: transmit signal (in resource grid)
    :param y_rg: received signal (in resource grid)
    """

    total_cycles = 0
    uncoded_ber = 0
    uncoded_ser = 0
    per_stream_ber_all = []

    for batch_idx in np.arange(y_rg.shape[0]):

        curr_ber, curr_uncoded_ser, per_stream_ber = test_phase_3_rx(cfg, ns3cfg, x_rg, y_rg[batch_idx:batch_idx+1,...], rx_heltf)

        if curr_ber < 0.4: # May not always have enough received data corresponding to all subframes in x_rg. In this case we won't always find a y_rg corresponding to an entry in x_rg
            uncoded_ber = uncoded_ber + curr_ber
            uncoded_ser = uncoded_ser + curr_uncoded_ser
            total_cycles += 1

        per_stream_ber_all.append(per_stream_ber)
                                  
    uncoded_ber = uncoded_ber / total_cycles
    uncoded_ser = uncoded_ser / total_cycles
    per_stream_ber_all = np.asarray(per_stream_ber_all)
    

    return uncoded_ber, uncoded_ser, per_stream_ber_all
