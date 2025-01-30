import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.mapping import Demapper
from sionna.ofdm import ResourceGrid, LSChannelEstimator, RemoveNulledSubcarriers
from sionna.utils import split_dim, flatten_last_dims, insert_dims

from dmimo.config import SimConfig
from dmimo.channel import LMMSELinearInterp
from .stbc import alamouti_decode

from scipy.io import savemat

class NCJT_RxUE(Model):
    """
    Implement of the reception of the Alamouti scheme in the dMIMO phase.
    """

    def __init__(self, cfg: SimConfig, lmmse_weights, batch_size, **kwargs):
        """
        Create NCJT RxUE object
        :param cfg: system settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.data_syms = np.delete(np.arange(0, cfg.symbols_per_slot, 1), cfg.pilot_indices)
        self.batch_size = batch_size
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order, hard_out=True)
        self.rg = ResourceGrid(num_ofdm_symbols=cfg.symbols_per_slot,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=2,
                               cyclic_prefix_length=cfg.cyclic_prefix_len,
                               num_guard_carriers=cfg.num_guard_carriers,
                               dc_null=cfg.dc_null,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=cfg.pilot_indices)

        # if self.cfg.perfect_csi is False:
        #     if self.cfg.lmmse_chest is False:
        #        self.ls_est = LSChannelEstimator(self.rg, interpolation_type=None)
        #        self.Wf = lmmse_weights
        #     else:
        #        lmmse_int = LMMSELinearInterp(self.rg.pilot_pattern, lmmse_weights)
        #        self.lmmse_est = LSChannelEstimator(self.rg, interpolator=lmmse_int)

        # For channel estimation using HE-LTF
        self.ltf_pmat = np.array([[1, -1], [1, 1]]).transpose()
        self.ltf_ref = 0.5 * np.array(
            [1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1,
             -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1,
             1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1,
             -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1,
             1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1,
             -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1,
             -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1,
             -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1,
             -1, 1, 1, -1]).reshape(1, -1, 1, 1)
        # HT-LTF cyclic shift compensation
        cshift = np.array([0, 8]).reshape(1, 1, 1, -1)
        pshift = (np.linspace(0, self.cfg.fft_size - 1, self.cfg.fft_size) - self.cfg.fft_size // 2)
        pshift = np.exp(-2j * np.pi * cshift / self.cfg.fft_size * pshift.reshape(1, -1, 1, 1))
        pshift = tf.convert_to_tensor(pshift, tf.complex64)
        self.pshift = tf.gather(pshift, self.rg.effective_subcarrier_ind, axis=1)  # (effective_subcarriers, num_rx)

    def estimate_freq_cov(self, h_freq):
        # chest has shape (num_batch, num_subcarrier, num_rx_ant, num_tx_ant)
        num_batch, num_sc = h_freq.shape[0:2]
        freq_cov_mat = tf.zeros([num_sc, num_sc], tf.complex64)

        for batch_idx in range(num_batch):
            # [num_batch, num_tx_ant, num_sc, num_rx_ant]
            h_samples = tf.transpose(h_freq, (0, 3, 1, 2))
            # [num_tx_ant, num_tx_ant, num_sc, num_sc]
            freq_cov_mat = tf.matmul(h_samples, h_samples, adjoint_b=True)
            # [num_sc, num_sc]
            freq_cov_mat = tf.reduce_mean(freq_cov_mat, axis=(0, 1))

        return freq_cov_mat

    def heltf_channel_estimate(self, heltf):
        # LS estimate using HT-LTF
        # heltf shape: (batch_size, num_subcarrier, num_rx_ant, num_ltf)
        heltf = tf.gather(heltf, self.rg.effective_subcarrier_ind, axis=1)  # remove null subcarriers
        ltfzf = tf.convert_to_tensor(heltf * self.ltf_ref, tf.complex64)
        hest = tf.matmul(ltfzf, self.ltf_pmat)
        # compensate for cyclic shifts
        hest = hest * self.pshift
        return hest

    # @tf.function(jit_compile=True)  # Enable graph execution to speed things up
    def call(self, ry_noisy=tf.Tensor, he_ltf=None, h_freq_ns3=None):

        # Extract data OFDM symbols
        # (num_subframes, num_subcarriers, len(data_syms), total_rx_antennas, 1)
        ry_stbc = tf.gather(ry_noisy, indices=self.data_syms, axis=2)
        ry_stbc = tf.gather(ry_stbc, indices=self.rg.effective_subcarrier_ind, axis=1)

        # TODO: accurate noise variance estimation
        nvar = tf.cast(2e-3, tf.float32)

        # Channel estimation use HE-LTF
        h_hat = self.heltf_channel_estimate(he_ltf)  # (num_batch, num_subcarrier, num_rx_ant, num_ss)
        h_hat_averaged = insert_dims(h_hat, 1, axis=2)
        h_hat_averaged = tf.repeat(h_hat_averaged, len(self.data_syms) // 2, axis=2)

        # Channel covariance statistics
        freq_cov = self.estimate_freq_cov(h_hat)
        lmmse_int = LMMSELinearInterp(self.rg.pilot_pattern, freq_cov)
        self.lmmse_est = LSChannelEstimator(self.rg, interpolator=lmmse_int)

        # Using perfect CSI
        if self.cfg.perfect_csi is True:
            # h_freq_ns3_estimated has shape
            #   (num_subframes, num_subcarriers, num_ofdm_symbols, total_rx_antennas, total_tx_antennas)
            h_freq_ns3_estimated = h_freq_ns3
            # (num_subframes, num_subcarriers, len(data_syms), total_rx_antennas, total_tx_antennas)
            h_freq_ns3_estimated = tf.gather(h_freq_ns3_estimated, indices=self.data_syms, axis=2)

            # Here we have an issue. Alamouti assumes that in two consecutive OFDM symbols the channel stays the same.
            # but that isn't generally true. In any case, we are going to feed the average of two consecutive OFDM symbol
            # channel to the STBC decoder.
            # (num_subframes, num_subcarriers, num_ofdm_symbols/2, total_rx_antennas, total_tx_antennas)
            h_freq_ns3_averaged = (h_freq_ns3_estimated[..., ::2, :, :] + h_freq_ns3_estimated[..., 1::2, :, :]) / 2

            # Now we need to sum over the respective transmit antennas
            total_tx_antennas = h_freq_ns3.shape[4]
            h_freq_ns3_averaged = tf.add_n([h_freq_ns3_averaged[..., i * 2:i * 2 + 2] for i in range(total_tx_antennas // 2)])
            # new shape is [num_subframes, num_subcarriers, len(data_syms)/2, total_rx_antennas, 2]

        # Simple channel estimation
        # elif self.cfg.lmmse_chest is False:
        #     ry_noisy = tf.transpose(ry_noisy, (0, 4, 3, 2, 1))
        #     # ry_noise shape [batch_size, num_rx_ant, num_tx_ant, num_ofdm_sym, nfft]
        #     h_hat = []
        #     for k in range(ry_noisy.shape[0]):
        #         # h_est shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, num_pilot_sym * nfft]
        #         h_est, err_var = self.ls_est([ry_noisy[k:k + 1], nvar])
        #         # new shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, num_pilot_sym, nfft]
        #         h_est = split_dim(h_est, [-1, self.rg.num_effective_subcarriers], axis=5)
        #         # average over time-domain, new shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, nfft]
        #         h_est = tf.reduce_mean(h_est, axis=5)
        #         # new shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, nfft/2, 2]
        #         h_est = split_dim(h_est, [self.rg.num_effective_subcarriers//2, 2], axis=5)
        #         # extract LS estimation for two Tx stream, new shape [..., num_tx_stream, nfft/2]
        #         h_est = tf.concat((h_est[..., 0:1, :, 0], h_est[..., 1:2, :, 1]), axis=4)
        #
        #         # interpolation function
        #         num_pt = 16  # fixed constant for now
        #         sfrm = tf.signal.frame(h_est, num_pt, 1)  # (num_batch, num_frame, 16)
        #         y_pre = h_est[..., :num_pt] @ self.Wf[:, :num_pt]
        #         y_main = sfrm @ self.Wf[:, num_pt:(num_pt + 2)]
        #         y_main = flatten_last_dims(y_main)
        #         y_post = h_est[..., -num_pt:] @ self.Wf[:, (num_pt + 2):]
        #         y_hat = tf.concat((y_pre, y_main, y_post), axis=-1)
        #         h_hat.append(y_hat)
        #
        #     h_hat = tf.concat(h_hat, axis=0)  # [num_batch, num_rx, num_rx_ant, num_tx, num_tx_stream, nfft]
        #     h_hat = tf.transpose(h_hat[:, 0], (0, 4, 2, 1, 3))  # [num_batch, nfft, 1, num_rx_ant, num_tx_stream]
        #     h_hat_averaged = tf.repeat(h_hat, len(self.data_syms)//2, axis=2)

        # LMMSE channel estimation
        else:
            ry_noisy = tf.transpose(ry_noisy, (0, 4, 3, 2, 1))
            # ry_noise shape [batch_size, num_rx_ant, num_tx_ant, num_ofdm_sym, nfft]
            h_hat = []
            for k in range(ry_noisy.shape[0]):
                h_est, err_var = self.lmmse_est([ry_noisy[k:k+1], 5e-3])
                h_hat.append(h_est[:, 0, :,  0, :, :, :])  # [1, num_rx_ant, num_tx_ant, num_ofdm_sym, nfft]
            h_hat = tf.concat(h_hat, axis=0)
            h_hat = tf.transpose(h_hat, (0, 4, 3, 1, 2))
            data_syms = [i for i in range(self.cfg.symbols_per_slot) if i not in self.cfg.pilot_indices]
            h_hat = tf.gather(h_hat, indices=data_syms, axis=2)
            h_hat_averaged = (h_hat[:, :, ::2] + h_hat[:, :, 1::2]) / 2.0
        
        # Reshape ry_stbc of shape [num_subframes, num_subcarriers, num_ofdm_symbols/2, 2, total_rx_antennas]
        num_ofdm_symbols = ry_stbc.shape[-3]
        total_rx_antennas = ry_stbc.shape[-2]
        ry_stbc = tf.reshape(ry_stbc, (*ry_stbc.shape[:-3], num_ofdm_symbols // 2, 2, total_rx_antennas))

        if self.cfg.perfect_csi:
            y, gains = alamouti_decode(ry_stbc[..., :], h_freq_ns3_averaged[..., :, :])
        else:
            y, gains = alamouti_decode(ry_stbc[..., :], h_hat_averaged[..., :, :])

        # y.shape = gains.shape = (num_subframes, num_subcarriers, num_ofdm_symbols)
        y = y / tf.cast(gains, y.dtype)

        # Turn into bits
        # y has shape = (num_subframes, num_subcarriers, num_ofdm_symbols*cfg.num_bits_per_symbol_phase2)
        yd = self.demapper([y, nvar])

        return yd, gains, nvar, y, h_hat
