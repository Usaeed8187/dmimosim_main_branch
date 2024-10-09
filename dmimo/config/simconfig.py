# Configuration for system simulation

from .sysconfig import CarrierConfig, MCSConfig


class SimConfig(CarrierConfig, MCSConfig):

    def __init__(self, **kwargs):
        self._name = "Simulation Configuration"
        self._enable_ue_selection = True        # Enable Tx/Rx UE selection
        self._num_tx_ue_sel = 8                 # number of Tx UE selected
        self._num_rx_ue_sel = 8                 # number of Rx UE selected
        self._start_slot_idx = 15               # start slot index for simulation
        self._csi_delay = 2                     # CSI estimation delay
        self._first_slot_idx = 0                # first slot index for phase 2 in simulation
        self._num_slots_p1 = 1                  # number of slots in phase 1/3
        self._num_slots_p2 = 3                  # number of slots in phase 2
        self._total_slots = 50                  # total slots of ns-3 channels
        self._ns3_folder = "../ns3/channels"    # data folder for ns-3 channels
        self._precoding_method = "ZF"           # precoding method
        self._ue_indices = None                 # UE antennas indices for MU-MIMO precoding
        self._ue_ranks = None                   # UE ranks for MU-MIMO precoding
        self._perfect_csi = False               # Use perfect CSI for debugging
        self._csi_prediction = False            # Use CSI prediction
        self._rank_adapt = True                 # turn on rank adaptation
        self._link_adapt = True                 # turn on link adaptation
        self._enable_txsquad = False            # enable simulation of TxSquad transmission
        self._enable_rxsquad = False            # enable simulation of TxSquad transmission
        self._cfo_sigma = [0.0]                 # standard deviation of CFO in Hz (vector for multiple nodes)
        self._sto_sigma = [0.0]                 # standard deviation of STO in nanoseconds (vector for multiple nodes)
        self._gen_sync_errors = False           # auto-generate random CFO/STO values for each phase/cycle
        self._random_cfo_vals = [0.0]           # random STO values in nanoseconds (temporary values for simulation)
        self._random_sto_vals = [0.0]           # random CFO values in Hz (temporary values for simulation)
        super().__init__(**kwargs)

    @property
    def enable_ue_selection(self):
        return self._enable_ue_selection

    @enable_ue_selection.setter
    def enable_ue_selection(self, val):
        self._enable_ue_selection = val

    @property
    def num_tx_ue_sel(self):
        return self._num_tx_ue_sel

    @num_tx_ue_sel.setter
    def num_tx_ue_sel(self, val):
        self._num_tx_ue_sel = val

    @property
    def num_rx_ue_sel(self):
        return self._num_rx_ue_sel

    @num_rx_ue_sel.setter
    def num_rx_ue_sel(self, val):
        self._num_rx_ue_sel = val

    @property
    def start_slot_idx(self):
        return self._start_slot_idx

    @start_slot_idx.setter
    def start_slot_idx(self, val):
        self._start_slot_idx = val

    @property
    def csi_delay(self):
        return self._csi_delay

    @csi_delay.setter
    def csi_delay(self, val):
        self._csi_delay = val

    @property
    def first_slot_idx(self):
        return self._first_slot_idx

    @first_slot_idx.setter
    def first_slot_idx(self, val):
        self._first_slot_idx = val

    @property
    def num_slots_p1(self):
        return self._num_slots_p1

    @num_slots_p1.setter
    def num_slots_p1(self, val):
        self._num_slots_p1 = val

    @property
    def num_slots_p2(self):
        return self._num_slots_p2

    @num_slots_p2.setter
    def num_slots_p2(self, val):
        self._num_slots_p2 = val

    @property
    def total_slots(self):
        return self._total_slots

    @total_slots.setter
    def total_slots(self, val):
        self._total_slots = val

    @property
    def ns3_folder(self):
        return self._ns3_folder

    @ns3_folder.setter
    def ns3_folder(self, val):
        self._ns3_folder = val

    @property
    def precoding_method(self):
        return self._precoding_method

    @precoding_method.setter
    def precoding_method(self, val):
        self._precoding_method = val

    @property
    def ue_indices(self):
        return self._ue_indices

    @ue_indices.setter
    def ue_indices(self, val):
        self._ue_indices = val

    @property
    def ue_ranks(self):
        return self._ue_ranks

    @ue_ranks.setter
    def ue_ranks(self, val):
        self._ue_ranks = val

    @property
    def perfect_csi(self):
        return self._perfect_csi

    @perfect_csi.setter
    def perfect_csi(self, val):
        self._perfect_csi = val

    @property
    def csi_prediction(self):
        return self._csi_prediction

    @csi_prediction.setter
    def csi_prediction(self, val):
        self._csi_prediction = val

    @property
    def rank_adapt(self):
        return self._rank_adapt

    @rank_adapt.setter
    def rank_adapt(self, val):
        self._rank_adapt = val
    
    @property
    def link_adapt(self):
        return self._link_adapt

    @link_adapt.setter
    def link_adapt(self, val):
        self._link_adapt = val

    @property
    def enable_txsquad(self):
        return self._enable_txsquad

    @enable_txsquad.setter
    def enable_txsquad(self, val):
        self._enable_txsquad = val

    @property
    def enable_rxsquad(self):
        return self._enable_rxsquad

    @enable_rxsquad.setter
    def enable_rxsquad(self, val):
        self._enable_rxsquad = val

    @property
    def cfo_sigma(self):
        return self._cfo_sigma

    @cfo_sigma.setter
    def cfo_sigma(self, val):
        self._cfo_sigma = val

    @property
    def sto_sigma(self):
        return self._sto_sigma

    @sto_sigma.setter
    def sto_sigma(self, val):
        self._sto_sigma = val

    @property
    def gen_sync_errors(self):
        return self._gen_sync_errors

    @gen_sync_errors.setter
    def gen_sync_errors(self, val):
        self._gen_sync_errors = val

    @property
    def random_sto_vals(self):
        return self._random_sto_vals

    @random_sto_vals.setter
    def random_sto_vals(self, val):
        self._random_sto_vals = val

    @property
    def random_cfo_vals(self):
        return self._random_cfo_vals

    @random_cfo_vals.setter
    def random_cfo_vals(self, val):
        self._random_cfo_vals = val

