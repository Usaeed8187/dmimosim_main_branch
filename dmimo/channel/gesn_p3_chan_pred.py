import copy
import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
import itertools
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class gesn_p3_chan_pred:

    def __init__(self, 
                architecture,
                rc_config,
                len_features=None, 
                num_rx_ant=8, 
                num_tx_ant=8, 
                max_adjacency='all', 
                method='per_antenna_pair', 
                num_neurons=None,
                cp_len=64,
                num_subcarriers=512,
                subcarrier_spacing=15e3,
                batch_size = 1,
                edge_weighting_method='grad_descent'):
        
        ns3_config = Ns3Config()
        self.rc_config = rc_config

        self.syms_per_subframe = 14
        self.nfft = 512  # TODO: remove hardcoded param value
        self.subcarriers_per_RB = 12
        self.N_RB = int(np.ceil(self.nfft / self.subcarriers_per_RB))
        self.num_rx_ant = num_rx_ant

        self.num_bs_ant = ns3_config.num_bs_ant
        self.num_ue_ant = ns3_config.num_ue_ant
        
        if architecture == 'baseline':
            self.N_t = ns3_config.num_bs_ant
            self.N_r = ns3_config.num_bs_ant
        elif architecture == 'SU_MIMO':
            self.N_t = num_tx_ant
            self.N_r = ns3_config.num_bs_ant * 2
        elif architecture == 'MU_MIMO':
            self.N_t = num_tx_ant
            self.N_r = num_rx_ant
        else:
            raise ValueError("\n The architecture specified is not defined")

        self.sparsity = self.rc_config.W_tran_sparsity
        self.spectral_radius = self.rc_config.W_tran_radius
        self.max_forget_length = self.rc_config.max_forget_length
        self.initial_forget_length = self.rc_config.initial_forget_length
        self.forget_length = self.rc_config.initial_forget_length
        self.forget_length_search_step = self.rc_config.forget_length_search_step
        self.input_scale = self.rc_config.input_scale
        self.window_length = self.rc_config.window_length
        self.learning_delay = self.rc_config.learning_delay
        self.reg = self.rc_config.regularization
        self.enable_window = self.rc_config.enable_window
        self.history_len = self.rc_config.history_len
        self.edge_weight_update_method = edge_weighting_method # "none", "grad_descent"
        self.cp_len = cp_len
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.num_epochs = self.rc_config.num_epochs
        self.learning_rate = self.rc_config.lr
        self.edge_weight_initialization = rc_config.weight_initialization # "none", "model_based_freq_corr", "model_based_delays", "uniform", "ones"
        self.batch_size = batch_size
        self.method = method
        self.window_weight_application = 'none'
        self.vector_inputs = rc_config.vector_inputs

        if self.vector_inputs == 'all':
            self.edge_weight_update_method = 'none'
            self.edge_weight_initialization = 'none'

        seed = 10
        self.RS = np.random.RandomState(seed)
        self.type = self.rc_config.type # 'real', 'complex'
        self.dtype = tf.complex64

        # Calculate weight matrix dimensions
        if method == 'per_antenna_pair':
            # one antenna pair is one vertex
            self.num_tx_nodes = int((self.N_t - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.num_rx_nodes = int((self.N_r - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.N_v = ns3_config.num_bs_ant * ns3_config.num_bs_ant                                        # number of vertices in the graph. Will be updated in *predict_per_antenna_pair()
            self.N_e = int((self.N_v*(self.N_v-1))/2)                                                       # number of edges in the graph (assumes fully connected). Will be updated in *predict_per_antenna_pair()
            if self.rc_config.treatment == 'SISO':
                self.N_f = self.N_RB                                                                        # length of feature vector for each vertex
            else:
                raise ValueError("\n The GESN treatment specified is not defined")                          # length of feature vector for each vertex

        elif method == 'per_node_pair':
            # one tx-rx node pair is one vertex
            self.num_tx_nodes = int((self.N_t - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.num_rx_nodes = int((self.N_r - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.N_v = self.num_tx_nodes * self.num_rx_nodes                                            # number of vertices in the graph
            self.N_e = int((self.N_v*(self.N_v-1))/2)                                                   # number of edges in the graph (assumes fully connected)
            if self.rc_config.treatment == 'SISO':
                self.N_f = self.N_RB                                                                        # length of feature vector for each vertex
            else:
                if len_features == None:
                    self.N_f = self.N_RB * ns3_config.num_bs_ant * ns3_config.num_bs_ant                    # length of feature vector for each vertex
                else:
                    self.N_f = len_features                                                                 # length of feature vector for each vertex

        else:
            raise ValueError("\n The GESN method specified is not defined")
        if num_neurons is None:
            self.N_n = self.rc_config.num_neurons * self.N_v
            self.N_n_per_vertex = self.rc_config.num_neurons
        else:
            self.N_n = num_neurons * self.N_v
            self.N_n_per_vertex = num_neurons

        if self.enable_window:
            self.N_in = self.N_f * self.window_length
        else:
            self.N_in = self.N_f
        self.N_out = self.N_f * self.N_v
        self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)

        # Initialize adjacency matrix (currently static for all time steps)
        if max_adjacency == 'all':
            self.max_adjacency = self.N_v
        elif max_adjacency == 'k_nearest_neighbours':
            raise ValueError("\n The knn clustering method has not yet been implemented")
        else:
            self.max_adjacency = max_adjacency

        # Initialize weight matrices
        self.init_weights()

        self.train_rls = False
        self.DF_rls = self.rc_config.DF_rls

        if self.train_rls or self.DF_rls:
            # for RLS algorithm
            self.psi = np.identity(self.N_in + self.N_n)
            if self.type == 'complex':
                self.psi = self.psi.astype(complex)
            self.psi_inv = np.linalg.inv(self.psi)

            # self.RLS_lambda = 0.9995, 0.9998
            # self.RLS_w = 1 / (1 + np.exp(-(self.EbNo - 11)))

            # self.RLS_lambda = 0.99999
            self.RLS_lambda = self.rc_config.RLS_lambda
            self.RLS_w = 1

    def get_csi_history(self, first_slot_idx, csi_delay, rg_csi, dmimo_chans):
        """
        Returns a tf tensor of shape:
        [history_length, batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym, fft_size]
        containing channel estimates (complex).
        """
        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)
        channel_history_slots = tf.range(first_csi_history_idx, first_slot_idx, csi_delay)

        h_freq_csi_history = tf.zeros((tf.size(channel_history_slots), self.batch_size, 1, (self.num_rx_nodes-1)*self.num_ue_ant+self.num_bs_ant,
                                       1, (self.num_tx_nodes-1)*self.num_ue_ant+self.num_bs_ant, self.syms_per_subframe, self.num_subcarriers), dtype=tf.complex64)
        for loop_idx, slot_idx in enumerate(channel_history_slots):
            # h_freq_csi has shape [batch_size, num_rx, num_rx_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
            folder_path = "ns3/channel_estimates_{}_{}_rx_{}_tx_{}".format(self.rc_config.mobility, self.rc_config.drop_idx,
                                                                                                self.N_r, self.N_t)
            file_path = "{}/dmimochans_{}".format(folder_path, slot_idx)
            try:
                data = np.load("{}.npz".format(file_path))
                h_freq_csi = data['h_freq_csi']
            except:
                h_freq_csi, _ = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=slot_idx)
                os.makedirs(folder_path, exist_ok=True)
                np.savez(file_path, h_freq_csi=h_freq_csi)
            indices = tf.constant([[loop_idx]])
            updates = tf.expand_dims(h_freq_csi, axis=0)
            h_freq_csi_history = tf.tensor_scatter_nd_update(h_freq_csi_history, indices, updates)

        return h_freq_csi_history
