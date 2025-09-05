import numpy as np
import itertools
import tensorflow as tf
from tensorflow.python.keras.layers import Layer


from dmimo.config import Ns3Config, SimConfig
from dmimo.channel import dMIMOChannels
from dmimo.mimo import rankAdaptation


class MUMIMOScheduler(Layer):

    def __init__(self,
                snrdb,
                precoder='ZF',
                method='exhaustive',
                num_streams_per_UE=1,
                rank=1,
                qam_order=2,
                always_schedule_gNB=True,
                max_rx_UEs_scheduled=8,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)


        self.method = method
        self.num_streams_per_UE = num_streams_per_UE
        self.rank = rank
        self.qam_order = qam_order

        self.num_BS_Ant = 4
        self.num_UE_Ant = 2

        self.always_schedule_gNB = always_schedule_gNB
        self.max_rx_UEs_scheduled = max_rx_UEs_scheduled

        self.rank_adaptation = rankAdaptation(num_bs_ant=4, num_ue_ant=2, architecture='MU-MIMO', snrdb=snrdb, fft_size=512, precoder=precoder)


    def call(self, h_eff):
        
        """
        Select RxSquad UEs using channel estimates

        Args:
            h_eff (tf.complex128): Tensor of shape `[1, 1, num_rx_ant, 1, num_tx_streams, num_ofdm_syms, num_subcarriers]` containing the
                reconstructed channel estimate.

        Returns:
            rx_UE_mask `dtype('int64')`: Array of shape `[num_scheduled_nodes, ]`.

        Raises:
            Exception: if using unsupported scheduling method. Supported methods are: 'exhaustive', 
        """

        rx_UE_mask = self.generate_rx_UE_mask(h_eff)

        return rx_UE_mask


    def generate_rx_UE_mask(self, h_eff):

        if self.method == 'exhaustive':
            
            # Get the maximum number of RxSquad nodes to schedule at a time out of all the existing RxSquad nodes
            max_num_rx_nodes = (h_eff.shape[2] - self.rank*2) // self.rank + self.always_schedule_gNB
            if self.max_rx_UEs_scheduled is None:
                num_rx_nodes_scheduled = max_num_rx_nodes
            else:
                num_rx_nodes_scheduled = self.max_rx_UEs_scheduled + self.always_schedule_gNB # equal to self.max_rx_UEs_scheduled is self.always_schedule_gNB is False, equal to self.max_rx_UEs_scheduled + 1 otherwise
            assert num_rx_nodes_scheduled <= max_num_rx_nodes, "Incorrect number of rx nodes scheduled"

            # Loop over all choices of number of RxSquad nodes to schedule and pick the best choice
            max_rate = float('-inf')
            if self.always_schedule_gNB:

                rx_UE_indices = list(np.arange(1, max_num_rx_nodes)) # Assume we always schedule from the maximum number of UEs

                for curr_num_rx_UEs in range(1, num_rx_nodes_scheduled):

                    curr_rx_node_combinations = list(itertools.combinations(rx_UE_indices, curr_num_rx_UEs))

                    for combo_id, combo in enumerate(curr_rx_node_combinations):
                        
                        combo = np.concatenate(([0], combo)) # Always scheduling gNB
                        rx_ants = self.find_rx_ants(combo)
                        
                        curr_h_eff = tf.gather(h_eff, rx_ants, axis=2)
                        _, _, curr_sum_rate = self.rank_adaptation.generate_rank_MU_MIMO(curr_h_eff, channel_type='dMIMO', prefixed_ranks=self.rank, num_rx_nodes=len(combo), pmi_input=True)

                        if curr_sum_rate >= max_rate:
                            best_combo = combo
                            max_rate = curr_sum_rate
                            print("\n \n current best combo: ", best_combo)
                            print("number of receive streams: ", len(rx_ants))
                
                return best_combo

            else:
                # rx_node_indices = list(np.arange(num_rx_nodes_scheduled))

                # for curr_num_rx_nodes in range(2, num_rx_nodes_scheduled+1): # assume we use 2 rx nodes at the minimum
                        
                #     curr_rx_node_combinations = list(itertools.combinations(rx_node_indices, curr_num_rx_nodes))

                #     for combo_id, combo in enumerate(curr_rx_node_combinations):
                        
                #         rx_ants = self.find_rx_ants(combo)
                        
                #         curr_h_eff = tf.gather(h_eff, rx_ants, axis=2)
                #         _, _, curr_sum_rate = self.rank_adaptation.generate_rank_MU_MIMO(curr_h_eff, channel_type='dMIMO', max_rank=self.rank)

                #         if curr_sum_rate >= max_rate:
                #             best_combo = combo
                #             max_rate = curr_sum_rate
                #             print("current best combo: ", best_combo)
                #             print("number of receive antennas used: ", len(rx_ants))
                
                # return best_combo
                raise Exception(f"Not scheduling the Rx Squad gNB is not fully supported yet.")
        else:
            raise Exception(f"The scheduling method specified has not been implemented.")

    def find_rx_ants(self, combo):

        num_BS_streams = self.rank*2 # Assume BS always has double the streams of the UEs
        num_UE_streams = self.rank

        if 0 in combo:    
            rx_ants_BS = np.arange(0, num_BS_streams)
            rx_UEs = combo[1:]
            rx_ants_UEs = []
            for rx_UE_idx in rx_UEs:
                rx_ants_UEs.append(np.arange(num_BS_streams + (rx_UE_idx-1)*num_UE_streams, num_BS_streams + rx_UE_idx*num_UE_streams))
            rx_ants_UEs = np.array(rx_ants_UEs)
            rx_ants_UEs = np.concatenate((rx_ants_UEs))
            rx_ants = np.concatenate((rx_ants_BS, rx_ants_UEs))
        else:
            rx_UEs = combo
            rx_ants_UEs = []
            for rx_UE_idx in rx_UEs:
                rx_ants_UEs.append(np.arange(num_BS_streams + (rx_UE_idx-1)*num_UE_streams, num_BS_streams + rx_UE_idx*num_UE_streams))
            rx_ants_UEs = np.array(rx_ants_UEs)
            rx_ants = np.concatenate((rx_ants_UEs))

        return rx_ants
