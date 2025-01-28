"""
Simulation of single-cluster NCJT scenario

"""

import os
import numpy as np
from typing import List 
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver
from sionna.utils import BinarySource
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import SimConfig, Ns3Config
from dmimo.channel import dMIMOChannels
from dmimo.mimo import update_node_selection
from dmimo.ncjt import MC_NCJT_TxUE, MC_NCJT_RxUE, NCJT_PostCombination


class MC_NCJT(Model):

    def __init__(self, cfg: SimConfig, ns3cfg: Ns3Config, cluster_formations:List[List[int]], modulation_order_list:List[int], dmimo_chans:dMIMOChannels , **kwargs):
        '''
        Multi cluster NCJT model. This model will handle transmission and reception of 
        NCJT signal in the multi-cluster setup. num_clusters represents the number of clusters. 
        The following config parameters are 
        required in this model:
        cfg.num_slots_p2
        cfg.fft_size
        cfg.symbols_per_slot
        cfg.pilot_indices
        cfg.code_rate
        cfg.modulation_order

        :param SimConfig cfg: configuration object
        :param cluster_formations: A list of size number of clusters, which each of its elements is a list of indices corresponding to the antennas of that cluster
        :param list modulation_order_list: list of modulation order of each cluster
        :param dMIMOChannels dmimo_chans: dMIMOChannels object
        '''
        super().__init__(kwargs)
        
        ## Assertions
        # Flatten the list of list of indices into a single list
        flattened_ant_idx = [item for sublist in cluster_formations for item in sublist]
        # Check if the length of the flattened list of indices is the same as the length of the set of unique elements
        assert len(flattened_ant_idx) == len(set(flattened_ant_idx)), "Some indices appear in multiple clusters."
        assert all(isinstance(item, int) and item % 2 == 0 for item in modulation_order_list), "Not all elements of modulation_order_list are even numbers"
        assert len(cluster_formations) == len(modulation_order_list) , 'The number of clusters assumed in modulation_order_list is not the same as that of cluster_formations.'
        
        ## Main body
        self.num_clusters = len(modulation_order_list)
        self.modulation_order_list = np.array(modulation_order_list)
        self.cluster_formations = cluster_formations
        self.cfg = cfg
        self.ns3cfg = ns3cfg
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2
        A: List[LDPC5GEncoder] = []
        # Update number of QAM symbols for data and LDPC params
        self.num_data_symbols = cfg.fft_size * (cfg.symbols_per_slot - len(cfg.pilot_indices))
        self.ldpc_n = int(2 * self.num_data_symbols)  # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * cfg.code_rate)  # Number of information bits

        self.num_codewords_list = self.modulation_order_list // 2  # number of codewords per frame
        self.num_bits_per_frame = self.ldpc_k * sum(self.num_codewords_list)
        self.num_uncoded_bits_per_frame = self.ldpc_n * sum(self.num_codewords_list)

        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True, num_iter=8)

        # Fixed interleaver design for current RG setting
        self.intlvr = RowColumnInterleaver(self.num_data_symbols // 2, axis=-1)
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)
        
        ## Create antenna_index to transmitting_index vector
        # So here is the issue we want to solve. We know that the index ranges from 0 to 23.
        # Each of those antennas corresponds to one of the transmitting clusters. And
        # within each cluster, the antenna is either the first Alamouti antenna or the second
        # Alamouti antenna. Here is what the mapper should map the antenna index to:
        # If it is corresponding to the first Alamouti antenna of the n-th cluster --> n*2
        # If it is corresponding to the second Alamouti antenna of the n-th cluster --> n*2+1 
        self.ant_to_stream_mapper = [1 for _ in range(24)] # Can we replace 24 with some dynamic thing?
        alamouti_idx = [0 for _ in range(self.num_clusters)]
        for ant_idx in range(24):
            if ant_idx in flattened_ant_idx:
                for i_cluster in range(self.num_clusters):
                    if ant_idx in cluster_formations[i_cluster]:
                        self.ant_to_stream_mapper[ant_idx] += 2*i_cluster + alamouti_idx[i_cluster]
                        alamouti_idx[i_cluster] = 1 - alamouti_idx[i_cluster]
                        break
            else:
                self.ant_to_stream_mapper[ant_idx] = 0 # This antenna should send nulls 
        

        if self.cfg.perfect_csi is False:
            # TODO: Auto generate LMMSE weights
            # filename = os.path.abspath(os.path.join(self.cfg.ns3_folder + "/../lmmse_weight.npy"))
            # self.Wf = np.load(filename) 
            # self.Wf = tf.constant(tf.convert_to_tensor(self.Wf, dtype=tf.complex64))
            from sionna.ofdm import ResourceGrid
            from dmimo.channel import estimate_freq_cov
            # Dummy resource grid for inputting to the estimate_freq_cov function:
            rg = ResourceGrid(num_ofdm_symbols=cfg.symbols_per_slot,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=2*self.num_clusters,
                               cyclic_prefix_length=cfg.cyclic_prefix_len,
                               num_guard_carriers=[0, 0],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=cfg.pilot_indices)
            self.Wf = estimate_freq_cov(dmimo_chans, rg, start_slot=cfg.start_slot_idx, total_slots=cfg.total_slots)

        self.ncjt_tx = MC_NCJT_TxUE(cfg , self.num_clusters , modulation_order_list)
        self.ncjt_rx = MC_NCJT_RxUE(cfg, lmmse_weights=self.Wf, batch_size = self.batch_size, modulation_order_list=modulation_order_list)
        self.ncjt_combination_list:List[NCJT_PostCombination] = [NCJT_PostCombination(modulation_order_list[i], return_LLRs=True) for i in range(self.num_clusters)]

    def call(self, dmimo_chans: dMIMOChannels):

        # Tx gNB processing


        # The binary source will create batches of information bits
        info_bits_list = [self.binary_source([self.batch_size, self.num_codewords_list[i], self.encoder.k]) for i in range(self.num_clusters)]

        # LDPC encoder processing
        c_list = [self.encoder(info_bits_list[i]) for i in range(self.num_clusters)] # [batch_size, num_codewords, ldpc_n]

        # Interleaving for coded bits
        c_intrlv_list = [self.intlvr(c_list[i]) for i in range(self.num_clusters)] # [batch_size, num_codewords, ldpc_n]

        # Phase 1 downlink transmission
        # TODO: assuming ideal transmission for now
        tx_bit_streams = [tf.reshape(c_intrlv_list[i], [self.batch_size, self.cfg.fft_size, -1]) for i in range(self.num_clusters)]
        # tx_bit_streams[i].shape: [batch_size , fft_size, num_data_syms * modulation_order_list[i]]

        # Phase 2 transmission from all gNB/UEs
        tx_signals_list = []
        for i_cluster in range(self.num_clusters):
            ue_tx_signal = self.ncjt_tx(tx_bit_streams[i_cluster], is_txbs=False, cluster_idx = i_cluster)
            tx_signals_list.append(ue_tx_signal)
        tx_signals = tf.concat(tx_signals_list, axis=-1) # shape: [batch_size, num_subcarriers, num_ofdm_symbols, num_tx_ant=4]
        # add a null antenna as the first antenna. (Used for when you want to turn off some of the antennas)
        padding = [([0,0] if i!=tx_signals.ndim-1 else [1,0]) for i in range(tx_signals.ndim)] # On the last axis, pad with zeros before the first antenna
        tx_signals = tf.pad(tx_signals, padding) # shape: [batch_size, num_subcarriers, num_ofdm_symbols, 5]
        tx_signals = tf.gather(tx_signals, self.ant_to_stream_mapper, axis=-1) # # shape: [batch_size, num_subcarriers, num_ofdm_symbols, 24]

        # new shape [batch_size, num_tx_ant, num_ofdm_sym, fft_size)
        tx_signals = tf.transpose(tx_signals, [0, 3, 2, 1])
        tx_signals = tf.expand_dims(tx_signals, axis=1)

        # apply dMIMO channels to the resource grid in the frequency domain
        ry, _ = dmimo_chans([tx_signals, self.cfg.first_slot_idx])
        if self.cfg.perfect_csi:
            h_freq, rx_snr_db, rx_pwr_dbm = dmimo_chans._load_channel(dmimo_chans._channel_type, slot_idx=self.cfg.start_slot_idx, batch_size=self.batch_size)
            # h_freq shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym, fft_size]
            h_freq = h_freq[:,0,:,0,:,:,:] # [batch_size, num_rx_ant, num_tx_ant, num_ofdm_sym, fft_size]
            h_freq_summed = tf.concat([tf.reduce_sum(tf.gather(h_freq, self.cluster_formations[i][j::2], 
                                                axis=2),
                                                axis=2, keepdims=True) for i in range(self.num_clusters) for j in range(2)],
                                                axis=2)
            h_freq_summed = tf.transpose(h_freq_summed, (0,4,3,1,2)) # (num_subframes, num_subcarriers, num_ofdm_symbols, total_rx_antennas, total_tx_antennas)
        ry = tf.transpose(ry, [0, 4, 3, 2, 1]) # [batch_size, fft_size, num_ofdm_syms, num_rx, 1]

        # Rx Squad processing 
        y_list = [[] for i in range(self.num_clusters)]
        gains_list = [[] for i in range(self.num_clusters)]
        nvar_list = []
        for ue_idx in range(0, self.ns3cfg.num_rxue_sel + 1):
            if ue_idx == 0:
                if self.cfg.perfect_csi:
                    y, gains, nvar = self.ncjt_rx(ry[:, :, :, 0:4, :], h_freq_ns3 = h_freq_summed[:, :, :, 0:4, :])
                else:
                    y, gains, nvar = self.ncjt_rx(ry[:, :, :, 0:4, :])
            else:
                if self.cfg.perfect_csi:
                    y, gains, nvar = self.ncjt_rx(ry[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :], h_freq_ns3 = h_freq_summed[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :])
                else:
                    y, gains, nvar = self.ncjt_rx(ry[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :])
            for i in range(self.num_clusters):
                y_list[i].append(y[i])
                gains_list[i].append(gains[i])
            nvar_list.append(nvar)

        # Phase 3 uplink transmission
        # TODO: assuming ideal transmission for now

        # Post-detection combining
        detected_bits_list = []
        detected_info_bits_list = []
        for i in range(self.num_clusters):
            combination_output = self.ncjt_combination_list[i](y_list[i], gains_list[i], nvar_list)
            detected_bits = tf.cast(combination_output > 0, tf.float32)
            # [batch_size, num_subcarriers, num_data_syms*num_bits_per_symbol]
            detected_bits_list.append(tf.reshape(detected_bits, tx_bit_streams[i].shape))

            # LDPC Decoding
            combination_llr = tf.reshape(combination_output, [self.batch_size, self.num_codewords_list[i], self.ldpc_n])
            combination_llr = self.dintlvr(combination_llr)
            detected_info_bits_list.append(self.decoder(combination_llr))  # [cfg.num_subframes, num_codewords, ldpc_k]

        # Error statistics
        uncoded_ber = compute_ber(tf.concat(detected_bits_list , axis=-1), tf.concat(tx_bit_streams , axis=-1))
        coded_ber = compute_ber(tf.concat(info_bits_list, axis=1) , tf.concat(detected_info_bits_list, axis=1) ).numpy()
        coded_bler = compute_bler(tf.concat(info_bits_list, axis=1) , tf.concat(detected_info_bits_list, axis=1)).numpy()

        # Goodput and throughput estimation
        goodbits = (1.0 - coded_ber) * self.num_bits_per_frame
        userbits = (1.0 - coded_bler) * self.num_bits_per_frame

        return [uncoded_ber, coded_ber, coded_bler], [goodbits, userbits]


def sim_mc_ncjt(cfg: SimConfig, ns3cfg: Ns3Config,):
    """
    Simulation of single-cluster NCJT scenario
    """

    # dMIMO channels from ns-3 simulator
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True)

    # UE selection
    ns3cfg.reset_ue_selection()
    if cfg.enable_ue_selection is True:
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
        ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

    # Create single-cluster NCJT simulation
    cluster_ant_list = [list(range(8)), list(range(8,24))]
    mod_order_list = [6,6] # [4,4]
    mc_ncjt = MC_NCJT(cfg, ns3cfg, cluster_ant_list, mod_order_list, dmimo_chans)

    # Loop over channels for all transmission cycles 
    total_cycles = 0
    uncoded_ber, coded_ber, coded_bler, goodput, throughput = 0, 0, 0, 0, 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        # Run simulation for one cycle
        bers, bits = mc_ncjt(dmimo_chans)
        # Update statistics (per slot)
        uncoded_ber += bers[0]
        coded_ber += bers[1]
        coded_bler += bers[2]
        goodput += bits[0]
        throughput += bits[1]

    uncoded_ber /= total_cycles
    coded_ber /= total_cycles
    coded_bler /= total_cycles

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = cfg.num_slots_p2 / (cfg.num_slots_p1 + cfg.num_slots_p2)
    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    return [uncoded_ber, coded_ber, coded_bler, goodput, throughput]
