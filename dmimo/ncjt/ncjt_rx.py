import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.mapping import Demapper
from sionna.ofdm import ResourceGrid, LSChannelEstimator
from sionna.utils import split_dim, flatten_last_dims

from dmimo.config import SimConfig
from .stbc import alamouti_decode


class NCJT_RxUE(Model):
    """
    Implement of the reception of the Alamouti scheme in the dMIMO phase.
    """

    def __init__(self, cfg: SimConfig, lmmse_weights, **kwargs):
        """
        Create NCJT RxUE object
        :param cfg: system settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.data_syms = np.delete(np.arange(0, cfg.symbols_per_slot, 1), cfg.pilot_indices)
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order, hard_out=True)
        self.rg = ResourceGrid(num_ofdm_symbols=cfg.symbols_per_slot,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=2,
                               cyclic_prefix_length=cfg.cyclic_prefix_len,
                               num_guard_carriers=[0, 0],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=cfg.pilot_indices)

        if self.cfg.perfect_csi is False:
            self.ls_est = LSChannelEstimator(self.rg, interpolation_type=None)
            self.Wf = lmmse_weights

    @tf.function(jit_compile=True)  # Enable graph execution to speed things up
    def call(self, ry_noisy=tf.Tensor, h_freq_ns3=None):

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

        # Extract data OFDM symbols
        # (num_subframes, num_subcarriers, len(data_syms), total_rx_antennas, 1)
        ry_stbc = tf.gather(ry_noisy, indices=self.data_syms, axis=2)

        # TODO: accurate noise variance estimation
        nvar = tf.cast(5e-2, tf.float32)

        # Channel estimation
        if self.cfg.perfect_csi is False:
            ry_noisy = tf.transpose(ry_noisy, (0, 4, 3, 2, 1))
            # ry_noise shape [batch_size, num_rx_ant, num_tx_ant, num_ofdm_sym, nfft]
            h_hat = []
            for k in range(ry_noisy.shape[0]):
                # h_est shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, num_pilot_sym * nfft]
                h_est, err_var = self.ls_est([ry_noisy[k:k + 1], nvar])
                # new shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, num_pilot_sym, nfft]
                h_est = split_dim(h_est, [-1, self.rg.num_effective_subcarriers], axis=5)
                # average over time-domain, new shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, nfft]
                h_est = tf.reduce_mean(h_est, axis=5)
                # new shape [num_batch, num_rx, rx_ant, num_tx, num_tx_stream, nfft/2, 2]
                h_est = split_dim(h_est, [self.rg.num_effective_subcarriers//2, 2], axis=5)
                # extract LS estimation for two Tx stream, new shape [..., num_tx_stream, nfft/2]
                h_est = tf.concat((h_est[..., 0:1, :, 0], h_est[..., 1:2, :, 1]), axis=4)
                # interpolation function
                num_pt = 16  # fixed constant for now
                sfrm = tf.signal.frame(h_est, num_pt, 1)  # (num_batch, num_frame, 16)
                y_pre = h_est[..., :num_pt] @ self.Wf[:, :num_pt]
                y_main = sfrm @ self.Wf[:, num_pt:(num_pt + 2)]
                y_main = flatten_last_dims(y_main)
                y_post = h_est[..., -num_pt:] @ self.Wf[:, (num_pt + 2):]
                y_hat = tf.concat((y_pre, y_main, y_post), axis=-1)
                h_hat.append(y_hat)

            h_hat = tf.concat(h_hat, axis=0)  # [num_batch, num_rx, num_rx_ant, num_tx, num_tx_stream, nfft]
            h_hat = tf.transpose(h_hat[:, 0], (0, 4, 2, 1, 3))  # [num_batch, nfft, 1, num_rx_ant, num_tx_stream]
            h_hat_averaged = tf.repeat(h_hat, len(self.data_syms)//2, axis=2)

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
        y = self.demapper([y, nvar])

        return y, gains, nvar
