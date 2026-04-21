# Configuration for the ns3 simulation
import numpy as np

from .sysconfig import NetworkConfig


class Ns3Config(NetworkConfig):

    def __init__(self, **kwargs):
        self._name = "ns3 Configuration"
        self._total_slots = 10      # total number of subframes/slots in the channel file
        self._data_folder = "../ns3/channels"
        self._file_prefix = "dmimochans"
        self._bs_txpwr_dbm = 35     # maximum transmission power per BS (dBm)
        self._ue_txpwr_dbm = 26     # maximum transmission power per UE (dBm)
        self._bs_ant_gain = 5       # BS antenna gain using 5dBi dipoles
        self._ue_ant_gain = 5       # UE antenna gain using 5dBi monopole
        self._noise_figure = 4      # RF front-end LNA noise figure
        self._thermal_noise = -105  # thermal noise power (dBm) for specified RF bandwidth (default 7.68MHz)
        self._ue_txpwr_ctrl = True  # enable RxSquad UE transmit power control
        super().__init__(**kwargs)

    def reset_ue_selection(self):
        self.num_txue_sel = self.num_txue
        self.num_rxue_sel = self.num_rxue
        self.txue_mask, self.rxue_mask = None, None

    def update_ue_selection(self, tx_ue_mask, rx_ue_mask):
        self.num_txue_sel = np.count_nonzero(tx_ue_mask)
        self.num_rxue_sel = np.count_nonzero(rx_ue_mask)
        self.txue_mask = tx_ue_mask
        self.rxue_mask = rx_ue_mask

    @property
    def total_slots(self):
        return self._total_slots

    @total_slots.setter
    def total_slots(self, val):
        assert val > 0, "Invalid total number of slots value"
        self._total_slots = val

    @property
    def data_folder(self):
        return self._data_folder

    @data_folder.setter
    def data_folder(self, val):
        self._data_folder = val

    @property
    def file_prefix(self):
        return self._file_prefix

    @file_prefix.setter
    def file_prefix(self, val):
        self._file_prefix = val

    @property
    def bs_txpwr_dbm(self):
        return self._bs_txpwr_dbm

    @bs_txpwr_dbm.setter
    def bs_txpwr_dbm(self, val):
        assert 0 < val <= 50, "Invalid BS Tx power value"
        self._bs_txpwr_dbm = val

    @property
    def ue_txpwr_dbm(self):
        return self._ue_txpwr_dbm

    @ue_txpwr_dbm.setter
    def ue_txpwr_dbm(self, val):
        assert 0 < val <= 50, "Invalid UE Tx power value"
        self._ue_txpwr_dbm = val

    @property
    def bs_ant_gain(self):
        return self._bs_ant_gain

    @bs_ant_gain.setter
    def bs_ant_gain(self, val):
        self._bs_ant_gain = val

    @property
    def ue_ant_gain(self):
        return self._ue_ant_gain

    @ue_ant_gain.setter
    def ue_ant_gain(self, val):
        self._ue_ant_gain = val

    @property
    def noise_figure(self):
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self, val):
        assert val >= 0, "Invalid noise figure value"
        self._noise_figure = val

    @property
    def thermal_noise(self):
        return self._thermal_noise

    @thermal_noise.setter
    def thermal_noise(self, val):
        assert val < 0, "Invalid thermal noise power value"
        self._thermal_noise = val

    @property
    def ue_txpwr_ctrl(self):
        return self._ue_txpwr_ctrl

    @ue_txpwr_ctrl.setter
    def ue_txpwr_ctrl(self, val):
        self._ue_txpwr_ctrl = val
