import numpy as np

from dmimo.config import Ns3Config, SimConfig
from dmimo.channel import dMIMOChannels


def update_node_selection(cfg: SimConfig, ns3cfg: Ns3Config):
    """
    Select TxSquad and RxSquad UEs using ns-3 channel statistics

    :param cfg: simulation configuration
    :parm ns3cfg: ns-3 configuration
    :return: Tx/Rx squad UE selection masks
    """

    # Instantiate dMIMO channel
    # ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=False)

    # load statistics for previous ns-3 channels
    # shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
    h_freq, snr_db = dmimo_chans.load_channel(slot_idx=cfg.start_slot_idx - cfg.csi_delay,
                                              batch_size=cfg.num_slots_p1 + cfg.num_slots_p2,
                                              ue_selection=False)

    # average over symbols and subcarrier in all subframes
    h_gain = np.mean(np.abs(h_freq), axis=(0, 1, 3, 5, 6), keepdims=False)
    # do not count BS antennas
    num_bs_ant = dmimo_chans.ns3_config.num_bs_ant

    # average over antennas on the same UE
    tx_gain = np.sum(h_gain[:, num_bs_ant:], axis=0)
    tx_gain = np.reshape(tx_gain, (2, -1))
    tx_gain = np.mean(tx_gain, axis=0, keepdims=False)

    # average over antennas on the same UE
    rx_gain = np.sum(h_gain[num_bs_ant:, :], axis=1)
    rx_gain = np.reshape(rx_gain, (2, -1))
    rx_gain = np.mean(rx_gain, axis=0, keepdims=False)

    # select the UEs with best link quality
    tx_ue_sort_idx = np.argsort(tx_gain)[-cfg.num_tx_ue_sel:]
    rx_ue_sort_idx = np.argsort(rx_gain)[-cfg.num_rx_ue_sel:]

    # update selection masks
    tx_ue_mask = np.zeros(len(tx_gain))
    tx_ue_mask[tx_ue_sort_idx] = np.ones(len(tx_ue_sort_idx))
    rx_ue_mask = np.zeros(len(rx_gain))
    rx_ue_mask[rx_ue_sort_idx] = np.ones(len(rx_ue_sort_idx))

    return tx_ue_mask, rx_ue_mask

