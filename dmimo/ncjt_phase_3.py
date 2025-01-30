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

        # The detector produces soft-decision output for all coded bits
        self.detector = MMSEPICDetector("bit", self.rg, sm, constellation_type="qam",
                                        num_bits_per_symbol=cfg.modulation_order,
                                        num_iter=2, hard_out=False)
        
        # The SIC+LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.sic_lmmse_equ = SICLMMSEEqualizer(self.rg, sm, cfg.modulation_order)
        
        # Using LMMSE CE from NCJT Demo code
        self.ncjt_rx = MC_NCJT_RxUE(self.cfg, batch_size=self.batch_size , modulation_order_list=[self.cfg.modulation_order])

    def call(self, dmimo_chans_ul, dmimo_chans_dl, h_freq_csi_dl, info_bits):
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


def ncjt_phase_3(cfg: SimConfig, ns3cfg: Ns3Config):
    """
    Simulation of MU-MIMO scenarios using different settings

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
        cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

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

    # # Rank and link adaptation
    
    # _, rx_snr_db, _ = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
    #                                             batch_size=cfg.num_slots_p2)
    # rank_feedback_report, n_var, mcs_feedback_report = \
    #     do_rank_link_adaptation(cfg, dmimo_chans, h_freq_csi, rx_snr_db)

    # if cfg.rank_adapt and cfg.link_adapt:
    #     # Update rank and total number of streams
    #     rank = rank_feedback_report[0]
    #     cfg.ue_ranks = [rank]
    #     cfg.num_tx_streams = rank * (ns3cfg.num_rxue_sel + 2)  # treat BS as two UEs

    #     qam_order_arr = mcs_feedback_report[0]
    #     code_rate_arr = mcs_feedback_report[1]
    #     values, counts = np.unique(qam_order_arr, return_counts=True)
    #     most_frequent_value = values[np.argmax(counts)]
    #     cfg.modulation_order = int(most_frequent_value)

    #     print("\n", "rank per user (MU-MIMO) = ", rank, "\n")
    #     # print("\n", "rate per user (MU-MIMO) = ", rate, "\n")

    #     values, counts = np.unique(code_rate_arr, return_counts=True)
    #     most_frequent_value = values[np.argmax(counts)]
    #     cfg.code_rate = most_frequent_value

    #     print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order, "\n")
    #     print("\n", "Code-rate per stream per user (MU-MIMO) = ", cfg.code_rate, "\n")

    # Create MU-MIMO simulation
    ncjt_phase_3 = NCJT_phase_3(cfg, rg_csi)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p1, ncjt_phase_3.num_bits_per_frame])

    # Phase 3 NCJT transmission
    dec_bits, uncoded_ber, uncoded_ser, x_hat = ncjt_phase_3(p3_chans_ul, p3_chans_dl, h_freq_csi_dl, info_bits)

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
        
        nodewise_goodput.append(additional_KPIs[0])
        nodewise_throughput.append(additional_KPIs[1])
        nodewise_bitrate.append(additional_KPIs[2])
        ranks_list.append(additional_KPIs[3])
        sinr_dB_list.append(additional_KPIs[4])


    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = 1 # Phase 3 results should not be compared with baseline
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
