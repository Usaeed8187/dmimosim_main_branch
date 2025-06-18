import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt
import time
from scipy.stats import mode

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource
from sionna.utils.metrics import compute_ber, compute_bler
from sionna.channel import ApplyOFDMChannel
from sionna.utils import expand_to_rank, complex_normal, flatten_last_dims

from dmimo.config import Ns3Config, SimConfig, NetworkConfig, RCConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation
from dmimo.mimo import ZFPrecoder, rankAdaptation, linkAdaptation, quantized_CSI_feedback, P1DemoPrecoder
from dmimo.mimo import update_node_selection
from dmimo.utils import add_frequency_offset, add_timing_offset, compute_UE_wise_BER, compute_UE_wise_SER

from .txs_mimo import TxSquad
from .rxs_mimo import RxSquad


class Phase1(Model):

    def __init__(self, cfg: SimConfig, rg_csi: ResourceGrid, **kwargs):
        """
        Create MU-MIMO simulation object

        :param cfg: simulation settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.rg_csi = rg_csi
        self.batch_size = cfg.num_slots_p1  # batch processing for all slots in phase 2

        # dMIMO configuration
        self.num_bs_ant = 4  # Tx squad BB
        self.num_ue_ant = 4  # Rx squad BB

        # The number of transmitted streams is less than or equal to the number of UE antennas
        assert cfg.num_tx_streams <= self.num_ue_ant
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

        self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)

        self.p1_demo_precoder = P1DemoPrecoder(self.rg, sm, return_effective_channel=True)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(self.rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order)

        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

        self.apply_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(tf.complex64))

        self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)


    def call(self, dmimo_chans: dMIMOChannels, info_bits, precoding_matrices):

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

        # h_freq_csi = np.linalg.pinv(precoding_matrices)
        h_freq_csi = np.conj(np.swapaxes(precoding_matrices, -2, -1))

        h_freq_csi = h_freq_csi
        h_freq_csi = tf.transpose(h_freq_csi, perm=[0, 1, 2, 4, 3])

        all_symbols = tf.range(self.rg.num_ofdm_symbols)
        pilot_symbols = self.rg._pilot_ofdm_symbol_indices

        # Get the set difference: symbols not used for pilots
        data_symbol_indices = tf.sets.difference(
            tf.expand_dims(all_symbols, 0), tf.expand_dims(pilot_symbols, 0)
        ).values

        # apply dMIMO channels to the resource grid in the frequency domain.
        h_freq, _, _ = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx,
                                                            batch_size=self.batch_size)
        h_freq = tf.gather(h_freq, tf.range(0, self.cfg.num_scheduled_tx_ue*2), axis=2)

        num_variances = 30
        avg_SNR = 15
        max_snr = 25
        min_snr = 5
        max_variance = (max_snr - min_snr)**2 / 12
        variances = np.linspace(0.1, max_variance, num_variances)
        num_realizations = 10
        uncoded_bers = np.zeros((2, num_variances, num_realizations, self.cfg.num_scheduled_tx_ue))
        
        for var_idx in range(num_variances):

            for curr_method in range(2):

                hold = 1

                for realisation_idx in range(num_realizations):

                    width = np.sqrt(12 * variances[var_idx]) / 2
                    a = avg_SNR - width
                    b = avg_SNR + width
                    rx_snr_db = np.random.uniform(a, b, self.cfg.num_scheduled_tx_ue)
                
                    # apply precoding to OFDM grids
                    if curr_method == 0:
                        x_precoded, h_eff, _, _ = self.p1_demo_precoder([x_rg, h_freq_csi, rx_snr_db, 'baseline'])
                    elif curr_method == 1:
                        if self.cfg.precoding_method == "weighted_mean" or self.cfg.precoding_method == "power_allocation":
                            x_precoded, h_eff, starting_SINR, best_SINR = self.p1_demo_precoder([x_rg, h_freq_csi, rx_snr_db, self.cfg.precoding_method])
                        else:
                            ValueError("unsupported precoding method for phase 1 demo")
                    
                    y = self.apply_channel([x_precoded, h_freq])
                    no = np.power(10.0, rx_snr_db / (-10.0))
                    y = self.awgn([y, no])

                    # LS channel estimation with linear interpolation
                    # no = tf.reduce_mean(no)
                    no = tf.cast(no, tf.float32)

                    x_hat = np.zeros(x_rg.shape, dtype=np.complex64)
                    x_hat = x_hat[np.newaxis, ...]
                    x_hat = np.repeat(x_hat, self.cfg.num_scheduled_tx_ue, axis=0)
                    x_hat = x_hat[..., :self.rg.num_effective_subcarriers]
                    x_hat = x_hat[..., data_symbol_indices, :]
                    for rx_node in range(self.cfg.num_scheduled_tx_ue):
                        curr_y = tf.gather(y, tf.range(rx_node*2, rx_node*2+2), axis=2)

                        curr_h, err_var = self.ls_estimator([curr_y, no[rx_node]])

                        curr_y = tf.gather(curr_y, self.rg.effective_subcarrier_ind, axis=-1)
                        
                        # curr_h = tf.gather(h_hat, tf.range(rx_node*2, rx_node*2+2), axis=2)
                        # curr_h = tf.squeeze(curr_h)
                        # curr_h = curr_h[np.newaxis, np.newaxis, :, np.newaxis, ...]
                        
                        curr_x_hat, no_eff = self.lmmse_equ([curr_y, curr_h, err_var, no[rx_node]])
                        x_hat[rx_node, ...] = np.reshape(curr_x_hat, x_hat.shape[1:])

                        llr = self.demapper([curr_x_hat, no_eff])

                        d_hard = tf.cast(llr > 0, tf.float32)

                        uncoded_bers[curr_method, var_idx, realisation_idx, rx_node] = compute_ber(d, d_hard).numpy()

        plt.figure()
        mean_ber = np.mean(uncoded_bers, axis=2)
        # best_user = np.argmin(mean_ber[0, :])
        # plt.semilogy(SNR_range, uncoded_bers[0,:,best_user], label='BER for best user (user {}) after weighted mean precoding'.format(best_user))
        # best_user = np.argmin(mean_ber[1, :])
        # plt.semilogy(SNR_range, uncoded_bers[1,:,best_user], label='BER for best user (user {}) after mean precoding'.format(best_user))

        worst_users = np.argmax(mean_ber, axis=-1)
        worst_users  = mode(worst_users, axis=1, keepdims=False).mode
        plt.semilogy(variances, mean_ber[0,:,worst_users[0]], label='BER for worst user (user {}) after mean precoding'.format(worst_users[0]))
        plt.semilogy(variances, mean_ber[1,:,worst_users[0]], label='BER for worst user (user {}) after weighted mean precoding'.format(worst_users[0]))
        plt.legend()
        plt.grid()
        plt.xlabel('SNR Variances')
        plt.ylabel('BER')
        plt.title('')
        plt.savefig('Mean and Weighted Mean - {} Users'.format(self.cfg.num_scheduled_tx_ue))


        return uncoded_bers, x_hat
    

    def nmse(self, H_true, H_pred, standard=True):
        # Promote both inputs to the same backend first
        if isinstance(H_true, np.ndarray) and isinstance(H_pred, np.ndarray):
            backend = np
        else:                              # at least one is a tf.Tensor
            H_true = tf.convert_to_tensor(H_true)          # keep original dtype
            H_pred = tf.cast(H_pred, H_true.dtype)         # <-- safe cast
            backend = tf

        diff = backend.abs(H_true - H_pred) ** 2
        num  = backend.reduce_sum(diff) if backend is tf else backend.sum(diff)

        if standard:
            denom_term = backend.abs(H_true) ** 2
        else:
            denom_term = (backend.abs(H_true) + backend.abs(H_pred)) ** 2

        denom = backend.reduce_sum(denom_term) if backend is tf else backend.sum(denom_term)
        return backend.cast(num / denom, backend.float32 if backend is tf else backend.float64)

    def awgn(self, inputs):

        x, no = inputs

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(tf.shape(x), dtype=x.dtype)
        
        if isinstance(no, (tf.Tensor, np.ndarray)):

            num_ues = no.shape[0]
            y = np.zeros(x.shape, dtype=np.complex64)
            for ue_idx in range(num_ues):
                ant_idx = np.arange(ue_idx*2, ue_idx*2 + 2)
                curr_noise = tf.gather(noise, ant_idx, axis=2) * tf.cast(tf.sqrt(no[ue_idx]), noise.dtype)
                y[:,:, ant_idx, ...] = tf.gather(x, ant_idx, axis=2) + curr_noise
            
        else:

            # Add extra dimensions for broadcasting
            no = expand_to_rank(no, tf.rank(x), axis=-1)

            # Apply variance scaling
            noise *= tf.cast(tf.sqrt(no), noise.dtype)

            # Add noise to input
            y = x + noise

        return y


def sim_phase_1(cfg: SimConfig, ns3cfg: Ns3Config):

    # Update UE selection
    if cfg.enable_ue_selection is True:
        ns3cfg.reset_ue_selection()
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
        ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)
        cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel) * 2), (ns3cfg.num_rxue_sel, -1))

    # CFO and STO settings
    if cfg.gen_sync_errors:
        cfg.random_sto_vals = cfg.sto_sigma * np.random.normal(size=(ns3cfg.num_rxue_sel, 1))
        cfg.random_cfo_vals = cfg.cfo_sigma * np.random.normal(size=(ns3cfg.num_rxue_sel, 1))
        
    # dMIMO channels from ns-3 simulator
    p1_chans_dl = dMIMOChannels(ns3cfg, "TxSquad", forward=True, add_noise=True)

    # Total number of TX (gNB) antennas in the TxSquad
    num_txs_ant = 4

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
        h_freq_csi_dl, rx_snr_db_dl, rx_pwr_dbm_dl = p1_chans_dl.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
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
        h_freq_csi_dl, _ = lmmse_channel_estimation(p1_chans_dl, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_vals=cfg.random_cfo_vals,
                                                           sto_vals=cfg.random_sto_vals)
        precoding_channel = h_freq_csi_dl

        _, rx_snr_db, _ = p1_chans_dl.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                                            forward=False,
                                                                            batch_size=cfg.num_slots_p1)

    # print ("h_freq_dl", h_freq_csi_dl.shape)
    # print ("h_freq_ul", h_freq_csi_ul.shape)

    if cfg.CSI_feedback_method =='5G':
        generate_CSI_feedback = quantized_CSI_feedback(method='5G', codebook_selection_method='rate', num_tx_streams=cfg.num_tx_streams, architecture='dMIMO_phase1', 
                                                        snrdb=rx_snr_db, wideband=True)
        [PMI, rate_for_selected_precoder, quantized_channels] = generate_CSI_feedback(h_freq_csi_dl)
    else:
        quantized_channels = None

    quantized_channels = quantized_channels[:cfg.num_scheduled_tx_ue, ...]

    # Create MU-MIMO simulation
    phase_1 = Phase1(cfg, rg_csi)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p1, phase_1.num_bits_per_frame])

    # Phase 3 NCJT transmission
    uncoded_bers, x_hat = phase_1(p1_chans_dl, info_bits, quantized_channels)

    return uncoded_bers


def sim_phase_1_all(cfg: SimConfig, ns3cfg: Ns3Config):
    """"
    Simulation of MU-MIMO scenario according to the frame structure
    """

    total_cycles = 0
        
    uncoded_bers = []

    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):

        print("first_slot_idx: ", first_slot_idx, "\n")

        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx

        curr_uncoded_bers = sim_phase_1(cfg, ns3cfg)

        uncoded_bers.append(curr_uncoded_bers)

    return uncoded_bers