# Configuration for system simulation

from .sysconfig import CarrierConfig, MCSConfig


class SimConfig(CarrierConfig, MCSConfig):

    def __init__(self, **kwargs):
        self._name = "Simulation Configuration"
        self._enable_ue_selection = True                            # Enable Tx/Rx UE selection
        self._start_slot_idx = 15                                   # start slot index for simulation
        self._csi_delay = 2                                         # CSI estimation delay
        self._first_slot_idx = 0                                    # first slot index for phase 2 in simulation
        self._num_slots_p1 = 1                                      # number of slots in phase 1/3
        self._num_slots_p2 = 3                                      # number of slots in phase 2
        self._total_slots = 50                                      # total slots of ns-3 channels
        self._ns3_folder = "../ns3/channels"                        # data folder for ns-3 channels
        self._precoding_method = "ZF"                               # precoding method for phase 2 and baseline: 'ZF', 'SVD', '5G_ZF_no_channel_reconstruction', '5G_ZF', '5G_max_min_demo'
        self._csi_quantization_on = True                                # Enable PMI, CQI feedback instead of perfect resolution feedback
        self._PMI_feedback_architecture = "dMIMO_phase2_CB2"        # 'dMIMO_phase2_rel_15_type_II', 'dMIMO_phase2_type_II_CB1', 'dMIMO_phase2_type_II_CB2'
        self._ue_indices = None                                     # UE antennas indices for MU-MIMO precoding
        self._ue_ranks = None                                       # UE ranks for MU-MIMO precoding
        self._perfect_csi = False                                   # Use perfect CSI for debugging
        self._csi_prediction = False                                # Use CSI prediction
        self._use_perfect_csi_history_for_prediction = False        # Use perfect CSI history for prediction (otherwise use imperfect estimated CSI history)
        self._rank_adapt = True                                     # turn on rank adaptation
        self._link_adapt = True                                     # turn on link adaptation
        self._enable_txsquad = False                                # enable simulation of TxSquad transmission
        self._enable_rxsquad = False                                # enable simulation of TxSquad transmission
        self._cfo_sigma = [0.0]                                     # standard deviation of CFO in Hz (vector for multiple nodes)
        self._sto_sigma = [0.0]                                     # standard deviation of STO in nanoseconds (vector for multiple nodes)
        self._gen_sync_errors = False                               # auto-generate random CFO/STO values for each phase/cycle
        self._random_cfo_vals = [0.0]                               # random STO values in nanoseconds (temporary values for simulation)
        self._random_sto_vals = [0.0]                               # random CFO values in Hz (temporary values for simulation)
        self._CSI_feedback_method = '5G'                            # which CSI feedback method to use. choices: '5G', 'RVQ'
        self._phase_1_precoding_method = "5G_max_min_demo"          # precoding method for phase 1: 'ZF', '5G_max_min_demo'
        self._scheduling = False                                    # Turn scheduling on or off for phase 2 of MU-MIMO architecture
        self._scheduled_rx_ue_indices = None                        # Scheduled UE antennas indices for MU-MIMO precoding
        self._ncjt_ldpc_decode_and_forward = True                   # Enable LDPC decode-and-forward at the Rx UEs for NCJT simulations
        self._channel_prediction_method = "two_mode_wesn"           # Channel prediction method: "old", "two_mode", "two_mode_tf"
        self._rl_user_count = 2                                     # Number of worst users for transmitter-side DEQN
        super().__init__(**kwargs)

    @property
    def enable_ue_selection(self):
        return self._enable_ue_selection

    @enable_ue_selection.setter
    def enable_ue_selection(self, val):
        self._enable_ue_selection = val

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
    def use_perfect_csi_history_for_prediction(self):
        return self._use_perfect_csi_history_for_prediction

    @use_perfect_csi_history_for_prediction.setter
    def use_perfect_csi_history_for_prediction(self, val):
        self._use_perfect_csi_history_for_prediction = val

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

    @property
    def CSI_feedback_method(self):
        return self._CSI_feedback_method

    @CSI_feedback_method.setter
    def CSI_feedback_method(self, val):
        self._CSI_feedback_method = val

    @property
    def phase_1_precoding_method(self):
        return self._phase_1_precoding_method

    @phase_1_precoding_method.setter
    def phase_1_precoding_method(self, val):
        self._phase_1_precoding_method = val

    @property
    def scheduling(self):
        return self._scheduling

    @scheduling.setter
    def scheduling(self, val):
        self._scheduling = val

    @property
    def scheduled_rx_ue_indices(self):
        return self._scheduled_rx_ue_indices

    @scheduled_rx_ue_indices.setter
    def scheduled_rx_ue_indices(self, val):
        self._scheduled_rx_ue_indices = val

    @property
    def ncjt_ldpc_decode_and_forward(self):
        return self._ncjt_ldpc_decode_and_forward
    
    @ncjt_ldpc_decode_and_forward.setter
    def ncjt_ldpc_decode_and_forward(self, val):
        assert val in [True, False], "Invalid value for ncjt_ldpc_decode_and_forward"
        if val is False:
            if self.enable_rxsquad is True:
                self.enable_rxsquad = False
                print("Warning: RxSquad disabled since ncjt_ldpc_decode_and_forward is set to False.")
        self._ncjt_ldpc_decode_and_forward = val

    @property
    def PMI_feedback_architecture(self):
        return self._PMI_feedback_architecture

    @PMI_feedback_architecture.setter
    def PMI_feedback_architecture(self, val):
        self._PMI_feedback_architecture = val

    @property
    def channel_prediction_method(self):
        return self._channel_prediction_method
    
    @channel_prediction_method.setter
    def channel_prediction_method(self, val):
        self._channel_prediction_method = val

    @property 
    def csi_quantization_on(self):
        return self._csi_quantization_on
    
    @csi_quantization_on.setter
    def csi_quantization_on(self, val):
        self._csi_quantization_on = val

    @property
    def rl_user_count(self):
        return self._rl_user_count

    @rl_user_count.setter
    def rl_user_count(self, val):
        self._rl_user_count = int(val)