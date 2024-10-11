import os
import tensorflow as tf
import numpy as np

from dmimo.config import Ns3Config


class LoadNs3Channel:

    def __init__(self, config: Ns3Config, dtype=tf.complex64):
        self._cfg = config
        self._dtype = dtype
        self._current_slot = -1
        self._Hts = None  # TxSquad channels, shape is [num_txue * num_ue_ant,num_bs_ant,num_ofdm_sym,num_subcarrier]
        self._Hrs = None  # RxSquad channels, shape is [num_bs_ant,num_rxue * num_ue_ant,num_ofdm_sym,num_subcarrier]
        self._Hdm = None  # dMIMO channels, shape is [num_rxs_ant,num_txs_ant,num_ofdm_sym,num_subcarrier]
        self._Lts = None  # TxSquad pathloss in dB, shape is [num_txue,num_ofdm_sym]
        self._Lrs = None  # RxSquad pathloss in dB, shape is [num_rxue,num_ofdm_sym]
        self._Ldm = None  # dMIMO pathloss in dB, shape is [num_rxue+1,num_txue+1,num_ofdm_sym]

    def __call__(self, channel_type, slot_idx=None, batch_size=1, ue_selection=True):
        if slot_idx is None:
            slot_idx = self._current_slot + 1  # advance to next slot by default
        slot_idx = slot_idx % self._cfg.total_slots  # for test purpose only

        for batch_idx in range(batch_size):
            # Load channel data for current slot when needed
            if slot_idx != self._current_slot:
                self._current_slot = slot_idx
                chan_filename = os.path.join(self._cfg.data_folder,
                                             "{}_{}.npz".format(self._cfg.file_prefix, slot_idx))
                with np.load(chan_filename) as data:
                    self._Hts = data['Hts']
                    self._Hrs = data['Hrs']
                    self._Hdm = data['Hdm']
                    self._Lts = data['Lts']
                    self._Lrs = data['Lrs']
                    self._Ldm = data['Ldm']

                # Apply UE selection masks
                # Note that the TxBS and RxBS are always selected
                if ue_selection and (self._cfg.txue_mask is not None):
                    assert self._cfg.num_txue == np.count_nonzero(self._cfg.txue_mask)
                    tx_ue_mask = self._cfg.txue_mask
                    tx_ant_mask = np.repeat(self._cfg.txue_mask, self._cfg.num_ue_ant)
                    txs_mask = np.concatenate(([True], tx_ue_mask), axis=0)
                    txs_ant_mask = np.concatenate((np.repeat([True], self._cfg.num_bs_ant),
                                                   np.repeat(tx_ue_mask, self._cfg.num_ue_ant)), axis=0)

                    self._Hts = self._Hts[tx_ant_mask]
                    self._Hdm = self._Hdm[:, txs_ant_mask]
                    self._Lts = self._Lts[tx_ue_mask]
                    self._Ldm = self._Ldm[:, txs_mask]

                if ue_selection and (self._cfg.rxue_mask is not None):
                    assert self._cfg.num_rxue == np.count_nonzero(self._cfg.rxue_mask)
                    rx_ue_mask = self._cfg.rxue_mask
                    rx_ant_mask = np.repeat(self._cfg.rxue_mask, self._cfg.num_ue_ant)
                    rxs_mask = np.concatenate(([True], rx_ue_mask), axis=0)
                    rxs_ant_mask = np.concatenate((np.repeat([True], self._cfg.num_bs_ant),
                                                   np.repeat(rx_ue_mask, self._cfg.num_ue_ant)), axis=0)
                    self._Hrs = self._Hrs[:, rx_ant_mask]
                    self._Hdm = self._Hdm[rxs_ant_mask]
                    self._Lrs = self._Lrs[rx_ue_mask]
                    self._Ldm = self._Ldm[rxs_mask]

            slot_idx = (slot_idx + 1) % self._cfg.total_slots

            h_freq, rx_snr = self.convert_channel(channel_type)

            if batch_idx == 0 and batch_size > 1:
                h_shape = [batch_size, *h_freq.shape]
                h_freq_all = np.zeros(h_shape, np.cdouble)
                s_shape = [batch_size, *rx_snr.shape]
                rx_snr_all = np.zeros(s_shape)

            if batch_size > 1:
                h_freq_all[batch_idx] = h_freq
                rx_snr_all[batch_idx] = rx_snr
            else:
                h_freq_all = np.expand_dims(h_freq, 0)
                rx_snr_all = np.expand_dims(rx_snr, 0)

        h_freq_all = np.expand_dims(h_freq_all, 1)  # (batch_size, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size)
        h_freq_all = np.expand_dims(h_freq_all, 3)  # (batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size)
        rx_snr_all = np.expand_dims(rx_snr_all, 1)  # (batch_size, num_rx, num_rx/num_tx, num_rx_ant/num_tx_ant, num_ofdm_sym)

        return tf.cast(h_freq_all, self._dtype), rx_snr_all

    def convert_channel(self, channel_type):

        # ----------------------------------------------------------------------
        #   Calibrate Rx power and estimate maximum SNR at receiving nodes
        # ----------------------------------------------------------------------

        # calibrated transmission power per node (dBm)
        tx_pwr_bs = self._cfg.bs_txpwr_dbm + self._cfg.bs_ant_gain
        tx_pwr_ue = self._cfg.ue_txpwr_dbm + self._cfg.ue_ant_gain

        if channel_type == "Baseline":
            # received power per antenna from all transmit antennas
            rx_power_dbm = tx_pwr_bs + self._cfg.bs_ant_gain - self._Ldm[0, :1, :]  # [1, num_ofdm_sym]
            rx_snr_db = rx_power_dbm - (self._cfg.thermal_noise + self._cfg.noise_figure)
            rx_snr_db = np.repeat(rx_snr_db, self._cfg.num_bs_ant, axis=0)  # [num_bs_ant, num_ofdm_sym]

        elif channel_type == "TxSquad":
            # assuming perfect AGC on each UE nodes, pathloss can be compensated
            # by individual UEs as long as received signal has high enough SNR

            rx_pwr_dbm = tx_pwr_bs + self._cfg.ue_ant_gain - self._Lts  # [num_txue, num_ofdm_sym]
            rx_snr_db = rx_pwr_dbm - (self._cfg.thermal_noise + self._cfg.noise_figure)
            rx_snr_db = np.repeat(rx_snr_db, self._cfg.num_ue_ant, axis=0)  # [num_txue*num_ue_ant, num_ofdm_sym]

        elif channel_type == "RxSquad":

            if self._cfg.ue_txpwr_ctrl:
                # assuming transmit power control on each UE and perfect power AGC on BS
                # pathloss from different UEs are normalized according to the maximum-loss (weakest) path as reference

                # total power is num_rxue * weakest_path_power
                rx_pwr_dbm = tx_pwr_ue + 10.0 * np.log10(self._cfg.num_rxue) + self._cfg.bs_ant_gain \
                    - np.max(self._Lrs, axis=0, keepdims=True)  # [1, num_ofdm_sym]
                rx_snr_db = rx_pwr_dbm - (self._cfg.thermal_noise + self._cfg.noise_figure)  # [1, num_ofdm_sym]
                rx_snr_db = np.repeat(rx_snr_db, self._cfg.num_bs_ant, axis=0)  # [num_bs_ant, num_ofdm_sym]

                # no need for agc gain compensation
                rs_rx_agc = np.zeros((self._cfg.num_rxue * self._cfg.num_ue_ant, 1))

            else:
                # assuming perfect AGC on BS and no power control on each UE, pathloss from different UEs
                # needs to be normalized using the minimum-loss (strongest) path as reference

                rx_pwr_path = tx_pwr_ue + self._cfg.bs_ant_gain - self._Lrs  # [num_rxue, num_ofdm_sym]
                # received power is sum of all from transmitter antennas
                rx_pwr_dbm = np.log10(np.sum(np.power(10.0, rx_pwr_path), axis=0, keepdims=True))
                # normalized received power assuming perfect AGC (per path)
                rs_rx_agc = rx_pwr_path - rx_pwr_dbm  # [num_rxue,num_ofdm_sym]
                rs_rx_agc = np.repeat(rs_rx_agc, self._cfg.num_ue_ant, axis=0)  # [num_rxue * num_ue_ant, num_ofdm_sym]
                # maximum snr per rx antenna
                rx_snr_db = rx_pwr_dbm - (self._cfg.thermal_noise + self._cfg.noise_figure)  # [1, num_ofdm_sym]
                rx_snr_db = np.repeat(rx_snr_db, self._cfg.num_bs_ant, axis=0)  # [num_bs_ant, num_ofdm_sym]

        elif channel_type == "dMIMO" or channel_type == "dMIMO-Forward":
            # 1) assuming perfect Rx AGC and no power control on Tx BS/UEs, pathloss from different BS/UEs
            #    needs to be normalized using the minimum-loss path as reference
            # 2) assuming perfect power control for TxSquad BS to RxSquad BB pathloss and
            #    RxSquad UE to RxSquad BS, no normalization is required for these channel coefficients
            # 3) pathloss for forwarding links is ignored, because the RxSquad links have low pathloss and
            #    received signal is always well above sensitivity level

            # expand according to number of Tx/Rx nodes
            tx_pwr_dbm = np.concatenate((np.repeat(tx_pwr_bs, self._cfg.num_bs),
                                         np.repeat(tx_pwr_ue, self._cfg.num_txue)))
            rx_ant_gain = np.concatenate((np.repeat(self._cfg.bs_ant_gain, self._cfg.num_bs),
                                         np.repeat(self._cfg.ue_ant_gain, self._cfg.num_rxue)))
            rx_pwr_path = np.reshape(tx_pwr_dbm, (1, -1, 1)) + np.reshape(rx_ant_gain, (-1, 1, 1)) - self._Ldm  # [num_rxue+1,num_txue+1,num_ofdm_sym]

            # received power is sum of all transmitter antennas
            rx_pwr_dbm = np.log10(np.sum(np.power(10.0, rx_pwr_path), axis=1, keepdims=True))  # [num_rxue+1,1,num_ofdm_sym]
            # normalized received power assuming perfect AGC (per path)
            dm_rx_agc = rx_pwr_path - rx_pwr_dbm  # [num_rxue+1,num_txue+1,num_ofdm_sym]
            rx_snr_db = np.squeeze(rx_pwr_dbm, axis=1) - (self._cfg.thermal_noise + self._cfg.noise_figure)  # [num_rxue+1,num_ofdm+sym]

            # expand according to the number of tx/rx antennas
            dm_rx_agc = np.concatenate((np.repeat(dm_rx_agc[:1], self._cfg.num_bs_ant, axis=0),
                                        np.repeat(dm_rx_agc[1:], self._cfg.num_ue_ant, axis=0)), axis=0)
            dm_rx_agc = np.concatenate((np.repeat(dm_rx_agc[:, :1], self._cfg.num_bs_ant, axis=1),
                                        np.repeat(dm_rx_agc[:, 1:], self._cfg.num_ue_ant, axis=1)), axis=1)

            if channel_type == "dMIMO":
                rx_snr_db = np.concatenate((np.repeat(rx_snr_db[:1], self._cfg.num_bs_ant, axis=0),
                                            np.repeat(rx_snr_db[1:], self._cfg.num_ue_ant, axis=0)), axis=0)  # [num_rxs_ant,num_ofdm_sym]
            else:  # TODO: dMIMO-Forward
                best_snr = np.max(rx_snr_db[1:], axis=0, keepdims=True)
                rx_snr_db = np.concatenate((np.repeat(rx_snr_db[:1], self._cfg.num_bs_ant, axis=0),
                                            np.repeat(best_snr, self._cfg.num_bs_ant, axis=0)), axis=0)  # [num_rxs_ant,num_ofdm_sym]

        elif channel_type == "dMIMO-Raw":
            pass

        else:
            raise ValueError("unsupported channel type")

        # ----------------------------------------------------------------------
        #   Combine pathloss and channel coefficients
        # ----------------------------------------------------------------------

        if channel_type == "Baseline":
            # direct TxSquad BS to RxSquad BS channel as baseline
            # shape: [num_bs_ant,num_bs_ant,num_ofdm_sym,num_subcarrier]
            h_freq = self._Hdm[:self._cfg.num_bs_ant, :self._cfg.num_bs_ant]

        elif channel_type == "TxSquad":
            h_freq = self._Hts  # [num_txue * num_ue_ant, num_bs_ant, num_ofdm_sym, num_subcarrier)

        elif channel_type == "RxSquad":
            # channel power scaling according to perfect AGC
            path_gains = np.power(10.0, rs_rx_agc/10.0)

            # adjust channel coefficient according to pathloss
            path_gains = np.expand_dims(np.expand_dims(np.sqrt(path_gains), 0), -1)  # [1,num_rxue * num_ue_ant,num_ofdm_sym,1]
            h_freq = path_gains * self._Hrs  # [num_bs_ant,num_rxue*num_ue_ant,num_ofdm_sym,fft_size]

        elif channel_type == "dMIMO":
            # channel power scaling according to perfect AGC
            path_gains = np.power(10.0, dm_rx_agc/10.0)  # [num_rxs_ant,num_txs_ant,num_ofdm_symbol]

            # adjust channel coefficient according to path loss
            path_gains = np.expand_dims(np.sqrt(path_gains), 3)
            h_freq = path_gains * self._Hdm

        elif channel_type == "dMIMO-Forward":
            # channel power scaling according to perfect AGC
            dm_path_gains = np.expand_dims(np.power(10.0, dm_rx_agc/10.0), -1)  # [num_rxs_ant,num_txs_ant,num_ofdm_sym,1]

            # adjust channel coefficient according to pathloss
            h_dm = dm_path_gains * self._Hdm

            # Obtain effective channel of dMIMO and forwarding channels
            h_rs = np.transpose(self._Hrs, (2, 3, 0, 1))  # (num_ofdm_sym,num_subcarrier,num_bs_ant,total_ue_ant)
            h_dm = np.transpose(h_dm, (2, 3, 0, 1))  # (num_ofdm_sym,num_subcarrier,num_rxs_ant,num_txs_ant)

            # forward channel via RxSq UEs, normalize according to total_ue_ant
            # TODO: determine power scaling method
            # h_fw = np.sqrt(1.0/self._cfg.num_txue) * np.matmul(h_rs, h_dm[:, :, self._cfg.num_bs_ant:, :])
            h_fw = np.sqrt(1.0/self._cfg.num_bs_ant) * np.matmul(h_rs, h_dm[:, :, self._cfg.num_bs_ant:, :])
            h_freq = np.concatenate((h_dm[:, :, :self._cfg.num_bs_ant, :], h_fw), 2)  # (num_ofdm_symbol,fft_size,2*num_bs_ant,total_squad_ant)
            h_freq = np.transpose(h_freq, (2, 3, 0, 1))

        elif channel_type == "dMIMO-Raw":
            h_freq = self._Hdm   # original dMIMO channel from ns-3
            rx_snr_db = self._Ldm  # origin pathloss in dB from ns-3

        else:
            raise ValueError("unsupported channel type")

        return h_freq, rx_snr_db
