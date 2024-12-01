# import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from typing import List

from sionna.mapping import Demapper, Mapper
from sionna.ofdm import ResourceGrid, LSChannelEstimator
from sionna.utils import split_dim, flatten_last_dims

from dmimo.config import SimConfig
from .stbc import alamouti_decode, alamouti_decode_zf_double, alamouti_encode

from dmimo.channel import dMIMOChannels

class MC_NCJT_RxUE(Model):
    """
    Implement of the reception of the Alamouti scheme in the dMIMO phase.
    """

    def __init__(self, cfg: SimConfig, lmmse_weights, batch_size ,modulation_order_list:list=None, **kwargs):
        """
        Create NCJT RxUE object
        :param cfg: system settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        # self.data_syms = np.delete(np.arange(0, cfg.symbols_per_slot, 1), cfg.pilot_indices)
        self.data_syms = [i for i in range(cfg.symbols_per_slot) if i not in cfg.pilot_indices]
        self.batch_size = batch_size
        self.modulation_orders = modulation_order_list
        self.num_clusters = len(modulation_order_list)

        self.mapper_list:List[Mapper] = [Mapper("qam", modulation_order_list[i]) for i in range(self.num_clusters)]
        self.demapper_list:List[Demapper] = [Demapper("maxlog", "qam", modulation_order_list[i], hard_out=True) for i in range(self.num_clusters)]
        # Recreate the resource grid object that was used at the transmitter side. This is for channel estimation purposes
        self.rg = ResourceGrid(num_ofdm_symbols=cfg.symbols_per_slot,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=2*self.num_clusters,
                               cyclic_prefix_length=cfg.cyclic_prefix_len,
                               num_guard_carriers=[0, 0],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=cfg.pilot_indices)

        if self.cfg.perfect_csi is False:
            self.ls_est = LSChannelEstimator(self.rg, interpolation_type=None)
            self.Wf = lmmse_weights
        ## Begin edit by Ramin
        from dmimo.channel import LMMSELinearInterp
        lmmse_int = LMMSELinearInterp(self.rg.pilot_pattern, lmmse_weights)
        self.lmmse_est = LSChannelEstimator(self.rg, interpolator=lmmse_int)


    # @tf.function(jit_compile=True)  # Enable graph execution to speed things up
    def call(self, ry_noisy=tf.Tensor, h_freq_ns3=None):#, dmimochans:dMIMOChannels=None):

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

            # # Now we need to sum over the respective transmit antennas
            # total_tx_antennas = h_freq_ns3.shape[4]
            # h_freq_ns3_averaged = tf.add_n([h_freq_ns3_averaged[..., i * 2:i * 2 + 2] for i in range(total_tx_antennas // 2)])
            # # new shape is [num_subframes, num_subcarriers, len(data_syms)/2, total_rx_antennas, 2]

        # Extract data OFDM symbols
        # (num_subframes, num_subcarriers, len(data_syms), total_rx_antennas, 1)
        ry_stbc = tf.gather(ry_noisy, indices=self.data_syms, axis=2)

        # TODO: accurate noise variance estimation
        nvar = tf.cast(4e-1, tf.float32)

        # Channel estimation
        # if self.cfg.perfect_csi is False:
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
        #         # interpolation function
        #         num_pt = 16  # fixed constant for now
        #         sfrm = tf.signal.frame(h_est, num_pt, 1)  # (num_batch, num_frame, 16)
        #         y_pre = h_est[..., :num_pt] @ self.Wf[:, :num_pt]
        #         y_main = sfrm @ self.Wf[:, num_pt:(num_pt + 2)]
        #         y_main = flatten_last_dims(y_main)
        #         y_post = h_est[..., -num_pt:] @ self.Wf[:, (num_pt + 2):]
        #         y_hat = tf.concat((y_pre, y_main, y_post), axis=-1)
        #         h_hat.append(y_hat)

        #     h_hat = tf.concat(h_hat, axis=0)  # [num_batch, num_rx, num_rx_ant, num_tx, num_tx_stream, nfft]
        #     h_hat = tf.transpose(h_hat[:, 0], (0, 4, 2, 1, 3))  # [num_batch, nfft, 1, num_rx_ant, num_tx_stream]
        #     h_hat_averaged = tf.repeat(h_hat, len(self.data_syms)//2, axis=2)

        # Channel estimation
        # if self.cfg.perfect_csi:
        #     h_freq, rx_snr_db, rx_pwr_dbm = dmimochans._load_channel(dmimochans._channel_type, slot_idx=self.cfg.start_slot_idx, batch_size=self.batch_size)
        ry_noisy = tf.transpose(ry_noisy, (0, 4, 3, 2, 1))
        # ry_noise shape [batch_size, num_rx_ant, num_tx_ant, num_ofdm_sym, nfft]
        if self.cfg.perfect_csi:
            h_hat = h_freq_ns3_estimated
            h_hat_averaged = h_freq_ns3_averaged
        else:
            h_hat = []
            # for k in range(ry_noisy.shape[0]):
            for k in range(self.batch_size):
                h_est, err_var = self.lmmse_est([ry_noisy[k:k+1], 5e-3])
                h_hat.append(h_est[:, 0, :,  0, :, :, :])  # [1, num_rx_ant, num_tx_ant, num_ofdm_sym, nfft]
            h_hat = tf.concat(h_hat, axis=0) # [batch_size, num_rx_ant, num_tx_ant, num_ofdm_sym, nfft]
            h_hat = tf.transpose(h_hat, (0, 4, 3, 1, 2)) # [batch_size , nfft , num_ofdm_sym , num_rx_ant , num_tx_ant]
            data_syms = [i for i in range(self.cfg.symbols_per_slot) if i not in self.cfg.pilot_indices]
            h_hat = tf.gather(h_hat, indices=data_syms, axis=2) # [batch_size , nfft , num_data_sym , num_rx_ant , num_tx_ant]
            h_hat_averaged = (h_hat[:, :, ::2] + h_hat[:, :, 1::2]) / 2.0 # [batch_size , nfft , num_data_sym/2 , num_rx_ant , num_tx_ant]

        
        
        # Reshape ry_stbc of shape [num_subframes, num_subcarriers, num_ofdm_symbols/2, 2, total_rx_antennas]
        num_ofdm_symbols = ry_stbc.shape[-3]
        total_rx_antennas = ry_stbc.shape[-2]
        ry_stbc = tf.reshape(ry_stbc, (*ry_stbc.shape[:-3], num_ofdm_symbols // 2, 2, total_rx_antennas))

        if self.cfg.perfect_csi:
            if self.num_clusters == 1:
                y, gains = alamouti_decode(ry_stbc[..., :], h_freq_ns3_averaged[..., :, :])
                y = y / tf.cast(gains, y.dtype)
                out_y = [self.demapper_list[0]([y,nvar])]
                out_gain = [gains]
                # y.shape = gains.shape = (num_subframes, num_subcarriers, num_ofdm_symbols)
            elif self.num_clusters == 2:
                y, gains = alamouti_decode_zf_double(ry_stbc, h_freq_ns3_averaged)
                out_y = [self.demapper_list[i]([x,nvar]) for i,x in enumerate([y[...,0],y[...,1]])]
                out_gain = [gains[...,0], gains[...,1]]
                # y.shape = gains.shape = (num_subframes, num_subcarriers, num_ofdm_symbols, 2) where 2 represents the stream/cluster index
            else:
                raise ValueError(f'The only number of clusters supported are 1 and 2. {self.num_clusters} is not supported.')
        else:
            if self.num_clusters == 1:
                y, gains = alamouti_decode(ry_stbc[..., :], h_hat_averaged[..., :, :])
                y = y / tf.cast(gains, y.dtype)
                out_y = [self.demapper_list[0]([y,nvar])]
                out_gain = [gains]
                # y.shape = gains.shape = (num_subframes, num_subcarriers, num_ofdm_symbols)
            elif self.num_clusters == 2:
                y, gains = alamouti_decode_zf_double(ry_stbc, h_hat_averaged)
                # y.shape = gains.shape = (num_subframes, num_subcarriers, num_ofdm_symbols, 2) where 2 represents the stream/cluster index
                out_y = [self.demapper_list[i]([x,nvar]) for i,x in enumerate([y[...,0],y[...,1]])]
                out_gain = [gains[...,0], gains[...,1]]
            else:
                raise ValueError(f'The only number of clusters supported are 1 and 2. {self.num_clusters} is not supported.')

        

        ## Start the SIC process
        if self.num_clusters == 2:
            cluster0isbetter = tf.cast((gains[...,0] >= gains[...,1]),y.dtype) # (num_subframes, num_subcarriers, num_ofdm_symbols)
            cluster1isbetter = 1 - cluster0isbetter
            better_y = cluster0isbetter * y[...,0] + cluster1isbetter * y[...,1] # (num_subframes, num_subcarriers, num_ofdm_symbols)
            worse_y = cluster1isbetter * y[...,0] + cluster0isbetter * y[...,1] # (num_subframes, num_subcarriers, num_ofdm_symbols)
            
            # The detected symbols:
            x0 = self.mapper_list[0](self.demapper_list[0]([y[...,0],nvar]))
            x1 = self.mapper_list[1](self.demapper_list[1]([y[...,1],nvar]))
            
            better_x = cluster0isbetter * x0 + cluster1isbetter * x1 # (num_subframes, num_subcarriers, num_ofdm_symbols)
            worse_x =  cluster1isbetter * x0 + cluster0isbetter * x1 # (num_subframes, num_subcarriers, num_ofdm_symbols)
            # Note that we can also get back from better_x and worse_x to x0 and x1
            # All we need to do is to get the cluster0isbetter indices of better_x
            # and cluster1isbtter (=cluster0isworse) indices of worse_x and add them together to form x0
            # We do the reverse to form x1. 
            
            # h_hat is of shape [batch_size , nfft , num_data_sym , num_rx_ant , num_tx_ant] where num_tx_ant=4.
            # Now h_hat[...,0:2] and h_hat[...,2:4] represents the channel from the two clusters to the Rx nodes
            channel0 = h_hat[...,0:2] # [batch_size , nfft , num_data_sym , num_rx_ant , 2]
            channel1 = h_hat[...,2:4] # [batch_size , nfft , num_data_sym , num_rx_ant , 2]
            better_x_alamouti = alamouti_encode(better_x) # (num_subframes, num_subcarriers, num_ofdm_symbols, 2)
            channel_shape = channel0.shape
            channel_better_cluster = (channel0*cluster0isbetter[...,tf.newaxis,tf.newaxis]+
                                    channel1*cluster1isbetter[...,tf.newaxis,tf.newaxis])
            # (num_subframes, num_subcarriers, num_ofdm_symbols, node_rx_antennas, 2)
            channel_worse_cluster = (channel0*cluster1isbetter[...,tf.newaxis,tf.newaxis]+
                                    channel1*cluster0isbetter[...,tf.newaxis,tf.newaxis])
            # (num_subframes, num_subcarriers, num_ofdm_symbols, node_rx_antennas, 2)
            better_cluster_effect = tf.matmul(channel_better_cluster, better_x_alamouti[...,tf.newaxis]) # (num_subframes, num_subcarriers, num_ofdm_symbols, node_rx_antennas, 1)
            better_cluster_effect = tf.reshape(better_cluster_effect, (*better_cluster_effect.shape[:-3],num_ofdm_symbols//2, 2, better_cluster_effect.shape[-2])) 
            # (num_subframes, num_subcarriers, num_ofdm_symbols//2 , 2, node_rx_antennas)
            # Remove the effect from the received signal
            ry_all_new = ry_stbc - better_cluster_effect # (num_subframes, num_subcarriers, num_ofdm_symbols//2 , 2, node_rx_antennas)
            channel_worse_cluster_reshaped = (channel_worse_cluster[...,::2,:,:] + channel_worse_cluster[...,1::2,:,:])/2 # (num_subframes, num_subcarriers, num_ofdm_symbols//2, node_rx_antennas, 2)
            new_y , new_SNR = alamouti_decode(ry_all_new, channel_worse_cluster_reshaped)
            # both are of (num_subframes, num_subcarriers, num_ofdm_symbols)
            new_y = new_y/ tf.cast(new_SNR,new_y.dtype) # (num_subframes, num_subcarriers, num_ofdm_symbols)
            new_x = (self.mapper_list[0](self.demapper_list[0]([new_y,nvar])) * cluster1isbetter + 
                    self.mapper_list[1](self.demapper_list[1]([new_y,nvar])) * cluster0isbetter)
            x0 = cluster0isbetter * better_x + cluster1isbetter * new_x # (num_subframes, num_subcarriers, num_ofdm_symbols)
            x1 = cluster1isbetter * better_x + cluster0isbetter * new_x # (num_subframes, num_subcarriers, num_ofdm_symbols)
            # Turn into bits
            out_y = [self.demapper_list[i]([x,nvar]) for i,x in enumerate([x0,x1])]
            out_gain = [gains[...,0], gains[...,1]]
        elif self.num_clusters == 1:
            # Turn into bits
            out_y = [self.demapper_list[0]([y,nvar])]
            out_gain = [gains]
            

        return out_y, out_gain, nvar

