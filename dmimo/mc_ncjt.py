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

from .rxs_mimo import RxSquad

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
        # Check if each cluster has an even number of antennas
        assert all(len(cluster_formations[i]) % 2 == 0 for i in range(len(cluster_formations))), "Some clusters have odd number of antennas."
        assert all(isinstance(item, int) and item % 2 == 0 for item in modulation_order_list), "Not all elements of modulation_order_list are even numbers"
        assert len(cluster_formations) == len(modulation_order_list) , 'The number of clusters assumed in modulation_order_list is not the same as that of cluster_formations.'
        
        ## Main body
        self.num_clusters = len(modulation_order_list)
        self.modulation_order_list = np.array(modulation_order_list)
        self.cluster_formations = cluster_formations
        self.cfg = cfg
        self.ns3cfg = ns3cfg
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2
        # Check if RB based 2 UE selection is enabled in kwargs
        self.RB_based_ue_selection = kwargs.get('RB_based_ue_selection', False)
        if self.RB_based_ue_selection:
            per_sc_SNR = True
            self.num_selected_ues = kwargs.get('num_selected_ues', ns3cfg.num_rxue_sel)
            self.num_selected_ues = min(self.num_selected_ues, ns3cfg.num_rxue_sel)
        else:
            per_sc_SNR = kwargs.get('perSC_SNR', False)
        
        # Total number of antennas in the TxSquad, always use all gNB antennas
        num_txs_ant = 2 * ns3cfg.num_txue_sel + ns3cfg.num_bs_ant

        # Adjust guard subcarriers for channel estimation grid
        self.effective_subcarriers = (cfg.fft_size // num_txs_ant) * num_txs_ant
        self.csi_guard_carriers_1 = (cfg.fft_size - self.effective_subcarriers) // 2
        self.csi_guard_carriers_2 = (cfg.fft_size - self.effective_subcarriers) - self.csi_guard_carriers_1

        A: List[LDPC5GEncoder] = []
        # Update number of QAM symbols for data and LDPC params
        self.num_data_symbols = (self.effective_subcarriers) * (cfg.symbols_per_slot - len(cfg.pilot_indices))
        self.ldpc_n = int(2 * self.num_data_symbols)  # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * cfg.code_rate)  # Number of information bits

        self.num_codewords_list = self.modulation_order_list // 2  # number of codewords per frame
        self.num_bits_per_frame = self.ldpc_k * sum(self.num_codewords_list)
        self.num_uncoded_bits_per_frame = self.ldpc_n * sum(self.num_codewords_list)

        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True, num_iter=8)
        self.decoder_soft = LDPC5GDecoder(self.encoder, hard_out=False, num_iter=8)

        # Fixed interleaver design for current RG setting
        self.intlvr = RowColumnInterleaver(self.num_data_symbols // 2, axis=-1)
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)
        
        # Create antenna_index to transmitting_index vector (self.ant_to_stream_mapper)
        # This maps each of the 24 physical antennas to a stream index in the padded tx_signals array.
        # - Stream 0 is reserved for the "null" (silent) antenna (all zeros, for antennas not in any cluster).
        # - Streams 1, 2, ... are for real signals, assigned per cluster and per Alamouti antenna.
        # For each antenna:
        #   - If it is in a cluster, assign it a stream index based on its cluster and Alamouti position:
        #       stream_index = 1 + 2*i_cluster + alamouti_idx[i_cluster]
        #       (i_cluster: cluster number, alamouti_idx alternates 0/1 for each antenna in the cluster)
        #   - If it is not in any cluster, assign it to stream 0 (null/silent).
        self.ant_to_stream_mapper = [1 for _ in range(24)]  # Start with 1 (real streams start at 1)
        alamouti_idx = [0 for _ in range(self.num_clusters)]  # Tracks which Alamouti antenna (0 or 1) for each cluster
        for ant_idx in range(24):
            if ant_idx in flattened_ant_idx:
                for i_cluster in range(self.num_clusters):
                    if ant_idx in cluster_formations[i_cluster]:
                        # Assign stream index: 1 + 2*i_cluster + alamouti_idx[i_cluster]
                        self.ant_to_stream_mapper[ant_idx] += 2*i_cluster + alamouti_idx[i_cluster]
                        # Alternate Alamouti index for next antenna in this cluster
                        alamouti_idx[i_cluster] = 1 - alamouti_idx[i_cluster]
                        break
            else:
                # Not in any cluster: assign to null stream (silent)
                self.ant_to_stream_mapper[ant_idx] = 0
        

        if self.cfg.perfect_csi is False or self.cfg.perfect_csi is True:
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
                               num_guard_carriers=[self.csi_guard_carriers_1, self.csi_guard_carriers_2],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=cfg.pilot_indices)
            self.Wf = estimate_freq_cov(dmimo_chans, rg, start_slot=cfg.start_slot_idx, total_slots=cfg.total_slots)

        self.ncjt_tx = MC_NCJT_TxUE(cfg , ns3cfg, self.num_clusters , modulation_order_list)
        self.ncjt_rx = MC_NCJT_RxUE(cfg , ns3cfg, lmmse_weights=self.Wf, batch_size = self.batch_size, modulation_order_list=modulation_order_list)
        self.ncjt_combination_list:List[NCJT_PostCombination] = [NCJT_PostCombination(modulation_order_list[i], return_LLRs=True, perSC_SNR = per_sc_SNR) for i in range(self.num_clusters)]

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
        tx_bit_streams = [tf.reshape(c_intrlv_list[i], [self.batch_size, self.effective_subcarriers, -1]) for i in range(self.num_clusters)]
        # tx_bit_streams[i].shape: [batch_size , fft_size, num_data_syms * modulation_order_list[i]]

        # --- Phase 2: Multi-cluster transmission signal construction ---
        # 1. For each cluster, generate the transmit signal (per-cluster, per-Ant pair)
        tx_signals_list = []
        for i_cluster in range(self.num_clusters):
            ue_tx_signal = self.ncjt_tx(tx_bit_streams[i_cluster], is_txbs=False, cluster_idx=i_cluster)
            tx_signals_list.append(ue_tx_signal)
        # 2. Concatenate all clusters' signals along the antenna axis
        #    Result: [batch_size, num_subcarriers, num_ofdm_symbols, num_tx_ant=4] (2 clusters × 2 Alamouti antennas)
        tx_signals = tf.concat(tx_signals_list, axis=-1)
        # 3. Add a "null" (all-zeros) antenna as the first antenna.
        #    This allows us to later map unused physical antennas to a silent stream.
        #    Padding shape: [batch_size, num_subcarriers, num_ofdm_symbols, 5]
        padding = [([0,0] if i != tx_signals.ndim-1 else [1,0]) for i in range(tx_signals.ndim)]
        tx_signals = tf.pad(tx_signals, padding)
        # 4. Map the 5 streams (null + 4 complex) to the full 24-antenna array using self.ant_to_stream_mapper.
        #    This expands the signal to [batch_size, num_subcarriers, num_ofdm_symbols, 24],
        #    with only the selected antennas active and all others silent.
        tx_signals = tf.gather(tx_signals, self.ant_to_stream_mapper, axis=-1)

        # new shape [batch_size, num_tx_ant, num_ofdm_sym, fft_size)
        tx_signals = tf.transpose(tx_signals, [0, 3, 2, 1])
        tx_signals = tf.expand_dims(tx_signals, axis=1)

        # apply dMIMO channels to the resource grid in the frequency domain
        ry, _ = dmimo_chans([tx_signals, self.cfg.first_slot_idx])
        if self.cfg.perfect_csi:
            h_freq, rx_snr_db, rx_pwr_dbm = dmimo_chans._load_channel(dmimo_chans._channel_type, slot_idx=self.cfg.first_slot_idx, batch_size=self.batch_size)
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
        LLR_list = [[] for i in range(self.num_clusters)]
        nvar_list = []
        for ue_idx in range(0, self.ns3cfg.num_rxue_sel + 1):
            if ue_idx == 0:
                if self.cfg.perfect_csi:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, 0:4, :], h_freq_ns3 = h_freq_summed[:, :, :, 0:4, :])
                else:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, 0:4, :])
            else:
                if self.cfg.perfect_csi:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :], h_freq_ns3 = h_freq_summed[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :])
                else:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :])
            for i_cluster in range(self.num_clusters):
                y_list[i_cluster].append(y[i_cluster])
                gains_list[i_cluster].append(gains[i_cluster])
                LLR_list[i_cluster].append(LLRs[i_cluster])
            nvar_list.append(nvar)

        detected_bits_list = []
        detected_info_bits_list = []
        detected_bits_rxnode_wise_list = []
        detected_info_bits_rxnode_wise_list = []
        decoded_llrs_list = []
        for i_rxnode in range(self.ns3cfg.num_rxue_sel + 1):
            detected_bits_cluster_wise = []
            detected_info_bits_cluster_wise = []
            decoded_llrs_cluster_wise = []
            for i_cluster in range(self.num_clusters):
                detected_bits_for_this_rxnode = tf.cast(LLR_list[i_cluster][i_rxnode] > 0, tf.float32)

                # LDPC Decoding
                llr = tf.reshape(LLR_list[i_cluster][i_rxnode], [self.batch_size, self.num_codewords_list[i_cluster], self.ldpc_n])
                llr = self.dintlvr(llr)
                decoded_llrs = self.decoder_soft(llr)
                decoded_llrs_cluster_wise.append(decoded_llrs)
                detected_info_bits_for_this_rxnode = tf.cast(decoded_llrs > 0, tf.float32)  # [batch_size, num_subcarriers, num_data_syms * num_bits_per_symbol]
                detected_bits_cluster_wise.append(detected_bits_for_this_rxnode)
                detected_info_bits_cluster_wise.append(detected_info_bits_for_this_rxnode)
            detected_bits_rxnode_wise_list.append(detected_bits_cluster_wise)
            detected_info_bits_rxnode_wise_list.append(detected_info_bits_cluster_wise)
            decoded_llrs_list.append(decoded_llrs_cluster_wise)
        uncoded_ber_node_wise = [compute_ber(tf.concat(detected_bits_rxnode_wise_list[i_node] , axis=-1), 
                                            tf.concat(tx_bit_streams , axis=-1)).numpy()
                                            for i_node in range(self.ns3cfg.num_rxue_sel + 1)]
        coded_ber_node_wise = [compute_ber(tf.concat(detected_info_bits_rxnode_wise_list[i_node], axis=1) , 
                                          tf.concat(info_bits_list, axis=1) ).numpy()
                                          for i_node in range(self.ns3cfg.num_rxue_sel + 1)]
        coded_bler_node_wise = [compute_bler(tf.concat(detected_info_bits_rxnode_wise_list[i_node], axis=1) , 
                                  tf.concat(info_bits_list, axis=1)).numpy()
                                    for i_node in range(self.ns3cfg.num_rxue_sel + 1)]

        detected_bits_rxnode_wise_list = tf.convert_to_tensor(detected_bits_rxnode_wise_list) # shape: [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_n]
        detected_info_bits_rxnode_wise_list = tf.convert_to_tensor(detected_info_bits_rxnode_wise_list) # shape: [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
        decoded_llrs_list = tf.convert_to_tensor(decoded_llrs_list) # shape: [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
        # Here we multiply the median of the decoded LLRs with the sign of each decoded LLR to get the final decoded LLRs sent to the Rx base station
        # The new shape will still be the same as decoded_llrs_list which is [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
        decoded_llrs_list_forwarded = tf.convert_to_tensor(np.median(tf.math.abs(decoded_llrs_list).numpy(), axis= -1, keepdims=True))*tf.sign(decoded_llrs_list)
        llr_based_info_bits = tf.cast(tf.reduce_sum(decoded_llrs_list_forwarded, axis=0) > 0, tf.float32) # Sum LLRs from all Rx nodes shape : [num_clusters, batch_size, num_codewords, ldpc_k]
        # Phase 3 uplink transmission
        # TODO: assuming ideal transmission for now
        # y_list[i][j] shape: [batch_size, num_subcarriers, num_ofdm_symbols * modulation_order_list[i]]
        # y_list[i][j] shows the received signal from stream cluster i at Rx node j
        # gains_list[i][j] shape: [batch_size, num_subcarriers, num_ofdm_symbols]
        # gains_list[i][j] shows the channel gain from stream cluster i at Rx node j
        # nvar_list[j] is a scalar.
        # nvar_list[j] shows the noise variance at Rx node j

        if self.cfg.enable_rxsquad:
            raise NotImplementedError("RxSquad is not yet implemented in MC_NCJT.")


        if self.RB_based_ue_selection:
            # We now create y_concat_list and gains_concat_list for each cluster by concatenating the signals from all Rx nodes
            y_concat_list = [tf.stack(y_list[i], axis=-1) for i in range(self.num_clusters)] # list of [batch_size, num_subcarriers, num_ofdm_symbols * modulation_order_list[i], num_rx_nodes]
            gains_concat_list = [tf.stack(gains_list[i], axis=-1) for i in range(self.num_clusters)] # list of [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_nodes]
            # on the gains_concat_list, we replace every 12 subcarriers with the average gain over those 12 subcarriers
            for i in range(self.num_clusters):
                num_RB = tf.cast(tf.math.ceil(self.effective_subcarriers / 12), tf.int32)
                # Handle the first num_RB-1 RBs first
                gains = gains_concat_list[i][:, :num_RB*12, ...]
                gains_reshaped = tf.reshape(gains, [self.batch_size, num_RB, 12, self.cfg.symbols_per_slot - len(self.cfg.pilot_indices), self.ns3cfg.num_rxue_sel + 1])
                gains_avg = tf.reduce_mean(gains_reshaped, axis=[2,3], keepdims=True) # average over 12 subcarriers shape: [batch_size, num_RB, 1, 1, num_rx_nodes]
                # repeat to match original shape shape: [batch_size, num_RB, 12, num_ofdm_symbols, num_rx_nodes]
                gains_avg_repeated = tf.repeat(tf.repeat(gains_avg, repeats=12, axis=2), repeats=self.cfg.symbols_per_slot - len(self.cfg.pilot_indices), axis=3) 
                # reshape to match original shape
                gains_avg_repeated = tf.reshape(gains_avg_repeated, [self.batch_size, num_RB*12, self.cfg.symbols_per_slot - len(self.cfg.pilot_indices), self.ns3cfg.num_rxue_sel + 1])
                # handle the last partial RB if exists
                gains_tail = gains_concat_list[i][:, num_RB*12:, ...] # shape: [batch_size, remaining_subcarriers, num_ofdm_symbols, num_rx_nodes]
                gains_tail_avg = tf.reduce_mean(gains_tail, axis=(1,2), keepdims=True) # shape: [batch_size, 1, 1, num_rx_nodes]
                # repeat to match original shape: [batch_size, remaining_subcarriers, num_ofdm_symbols, num_rx_nodes]
                gains_tail_avg_repeated = tf.repeat(tf.repeat(gains_tail_avg, repeats=tf.shape(gains_tail)[1], axis=1), repeats=self.cfg.symbols_per_slot - len(self.cfg.pilot_indices), axis=2)
                gains_concat_list[i] = tf.concat([gains_avg_repeated, gains_tail_avg_repeated], axis=1) # concatenate back

            # Among the Rx nodes except the gNB (first node), we see which two nodes have the highest SNR for each cluster at each subccarrier
            # We use y_concat_list and gains_concat_list to find the two nodes with highest SNR for each cluster
            # Since we have already averaged the gains over subcarriers of each RB, it is as if we have per-subcarrier SNR information
            # We select the two nodes with highest average SNR over all subcarriers for each cluster
            # We then only use those two nodes for post-detection combining
            selected_y_list = []
            selected_gains_list = []
            selected_nvar_list = []
            for i in range(self.num_clusters):
                # Get indices of the top 2 nodes with highest SNR (excluding the first node which is gNB)
                gain_allrxUEs = gains_concat_list[i][:, :, :, 1:]  # Exclude gNB
                y_allrxUEs = y_concat_list[i][:, :, :, 1:]  # Exclude gNB
                y_allrxUEs = tf.stack([self.ncjt_combination_list[i].mapper(y_allrxUEs[...,j]) for j in range(self.ns3cfg.num_rxue_sel)], axis=-1) # map to symbols, shape: [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_nodes-1]
                # topk_gains shape: [batch_size, num_subcarriers, num_ofdm_symbols, self.num_selected_ues]
                # topkindices shape: [batch_size, num_subcarriers, num_ofdm_symbols, self.num_selected_ues]
                topk_gains, topkindices = tf.math.top_k(gain_allrxUEs, k=self.num_selected_ues, sorted=False)
                
                topk_y = tf.gather(y_allrxUEs, topkindices, batch_dims=-1)
                # nvar_list is a little bit more tricky since it has scalar values for each Rx node
                nvar_allrxUEs = tf.stack(nvar_list[1:], axis=0)  # shape: [num_rx_nodes-1]
                # we need to loop over the batch dimension to gather the correct nvar for each selected node
                topk_nvar = tf.gather(nvar_allrxUEs, topkindices, axis=0) # shape: [batch_size, num_subcarriers, num_ofdm_symbols, self.num_selected_ues]
                # selected_nvar_list.append(topk_nvar) # topk_nvar shape: [batch_size, num_subcarriers, num_ofdm_symbols, self.num_selected_ues]
                selected_y_list.append([y_concat_list[i][...,0]] + [self.ncjt_combination_list[i].demapper([topk_y[...,j] , topk_nvar[...,j]]) for j in range(self.num_selected_ues)])
                selected_gains_list.append([gains_concat_list[i][...,0]] + [topk_gains[...,j] for j in range(self.num_selected_ues)])
                selected_nvar_list.append(tf.stack([nvar_list[0]*np.ones_like(topk_nvar[...,0])] + [topk_nvar[...,j] for j in range(self.num_selected_ues)], axis=-1))
        
        
        # Post-detection combining
        detected_bits_list = []
        detected_info_bits_list = []
        for i in range(self.num_clusters):
            if self.RB_based_ue_selection:
                combination_output = self.ncjt_combination_list[i](selected_y_list[i], selected_gains_list[i], selected_nvar_list[i])
            else:
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

class MC_NCJT_LLR_Combining(MC_NCJT):
    
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
        tx_bit_streams = [tf.reshape(c_intrlv_list[i], [self.batch_size, self.effective_subcarriers, -1]) for i in range(self.num_clusters)]
        # tx_bit_streams[i].shape: [batch_size , fft_size, num_data_syms * modulation_order_list[i]]

        # --- Phase 2: Multi-cluster transmission signal construction ---
        # 1. For each cluster, generate the transmit signal (per-cluster, per-Ant pair)
        tx_signals_list = []
        for i_cluster in range(self.num_clusters):
            ue_tx_signal = self.ncjt_tx(tx_bit_streams[i_cluster], is_txbs=False, cluster_idx=i_cluster)
            tx_signals_list.append(ue_tx_signal)
        # 2. Concatenate all clusters' signals along the antenna axis
        #    Result: [batch_size, num_subcarriers, num_ofdm_symbols, num_tx_ant=4] (2 clusters × 2 Alamouti antennas)
        tx_signals = tf.concat(tx_signals_list, axis=-1)
        # 3. Add a "null" (all-zeros) antenna as the first antenna.
        #    This allows us to later map unused physical antennas to a silent stream.
        #    Padding shape: [batch_size, num_subcarriers, num_ofdm_symbols, 5]
        padding = [([0,0] if i != tx_signals.ndim-1 else [1,0]) for i in range(tx_signals.ndim)]
        tx_signals = tf.pad(tx_signals, padding)
        # 4. Map the 5 streams (null + 4 complex) to the full 24-antenna array using self.ant_to_stream_mapper.
        #    This expands the signal to [batch_size, num_subcarriers, num_ofdm_symbols, 24],
        #    with only the selected antennas active and all others silent.
        tx_signals = tf.gather(tx_signals, self.ant_to_stream_mapper, axis=-1)

        # new shape [batch_size, num_tx_ant, num_ofdm_sym, fft_size)
        tx_signals = tf.transpose(tx_signals, [0, 3, 2, 1])
        tx_signals = tf.expand_dims(tx_signals, axis=1)

        # apply dMIMO channels to the resource grid in the frequency domain
        ry, _ = dmimo_chans([tx_signals, self.cfg.first_slot_idx])
        if self.cfg.perfect_csi:
            h_freq, rx_snr_db, rx_pwr_dbm = dmimo_chans._load_channel(dmimo_chans._channel_type, slot_idx=self.cfg.first_slot_idx, batch_size=self.batch_size)
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
        LLR_list = [[] for i in range(self.num_clusters)]
        nvar_list = []
        for ue_idx in range(0, self.ns3cfg.num_rxue_sel + 1):
            if ue_idx == 0:
                if self.cfg.perfect_csi:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, 0:4, :], h_freq_ns3 = h_freq_summed[:, :, :, 0:4, :])
                else:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, 0:4, :])
            else:
                if self.cfg.perfect_csi:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :], h_freq_ns3 = h_freq_summed[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :])
                else:
                    y, gains, nvar, LLRs = self.ncjt_rx(ry[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :])
            for i_cluster in range(self.num_clusters):
                y_list[i_cluster].append(y[i_cluster])
                gains_list[i_cluster].append(gains[i_cluster])
                LLR_list[i_cluster].append(LLRs[i_cluster])
            nvar_list.append(nvar)

        detected_bits_list = []
        detected_info_bits_list = []
        detected_bits_rxnode_wise_list = []
        detected_info_bits_rxnode_wise_list = []
        decoded_llrs_list = []
        for i_rxnode in range(self.ns3cfg.num_rxue_sel + 1):
            detected_bits_cluster_wise = []
            detected_info_bits_cluster_wise = []
            decoded_llrs_cluster_wise = []
            for i_cluster in range(self.num_clusters):
                detected_bits_for_this_rxnode = tf.cast(LLR_list[i_cluster][i_rxnode] > 0, tf.float32)

                # LDPC Decoding
                llr = tf.reshape(LLR_list[i_cluster][i_rxnode], [self.batch_size, self.num_codewords_list[i_cluster], self.ldpc_n])
                llr = self.dintlvr(llr)
                decoded_llrs = self.decoder_soft(llr)
                decoded_llrs_cluster_wise.append(decoded_llrs)
                detected_info_bits_for_this_rxnode = tf.cast(decoded_llrs > 0, tf.float32)  # [batch_size, num_subcarriers, num_data_syms * num_bits_per_symbol]
                detected_bits_cluster_wise.append(detected_bits_for_this_rxnode)
                detected_info_bits_cluster_wise.append(detected_info_bits_for_this_rxnode)
            detected_bits_rxnode_wise_list.append(detected_bits_cluster_wise)
            detected_info_bits_rxnode_wise_list.append(detected_info_bits_cluster_wise)
            decoded_llrs_list.append(decoded_llrs_cluster_wise)
        uncoded_ber_node_wise = [compute_ber(tf.concat(detected_bits_rxnode_wise_list[i_node] , axis=-1), 
                                            tf.concat(tx_bit_streams , axis=-1)).numpy()
                                            for i_node in range(self.ns3cfg.num_rxue_sel + 1)]
        coded_ber_node_wise = [compute_ber(tf.concat(detected_info_bits_rxnode_wise_list[i_node], axis=1) , 
                                          tf.concat(info_bits_list, axis=1) ).numpy()
                                          for i_node in range(self.ns3cfg.num_rxue_sel + 1)]
        coded_bler_node_wise = [compute_bler(tf.concat(detected_info_bits_rxnode_wise_list[i_node], axis=1) , 
                                  tf.concat(info_bits_list, axis=1)).numpy()
                                    for i_node in range(self.ns3cfg.num_rxue_sel + 1)]

        detected_bits_rxnode_wise_list = tf.convert_to_tensor(detected_bits_rxnode_wise_list) # shape: [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_n]
        detected_info_bits_rxnode_wise_list = tf.convert_to_tensor(detected_info_bits_rxnode_wise_list) # shape: [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
        decoded_llrs_list = tf.convert_to_tensor(decoded_llrs_list) # shape: [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
        
        decoded_llrs_list_forwarded = tf.convert_to_tensor(np.median(tf.math.abs(decoded_llrs_list).numpy(), axis= -1, keepdims=True)) # shape: [num_rx_nodes, num_clusters, batch_size, num_codewords, 1]
        # Phase 3 uplink transmission
        if self.cfg.enable_rxsquad is True:
            rxcfg = self.cfg.clone()
            rxcfg.csi_delay = 4
            rxcfg.decoder = "lmmse"
            rxcfg.perfect_csi = False
            rxcfg.first_slot_idx = self.cfg.first_slot_idx + self.cfg.num_slots_p2
            num_ue_bits_per_frame = self.num_bits_per_frame * self.ns3cfg.num_rxue_sel

            rx_ue_mask = np.zeros(10)
            rx_ue_mask[0:self.ns3cfg.num_rxue_sel] = 1

            rx_ns3cfg = Ns3Config(data_folder=self.cfg.ns3_folder, total_slots=self.cfg.total_slots)
            rx_ns3cfg.update_ue_selection(None, rx_ue_mask)
            rxs_chans = dMIMOChannels(rx_ns3cfg, "RxSquad", add_noise=False)
            rx_squad = RxSquad(rxcfg, self.ns3cfg, num_ue_bits_per_frame, rxs_chans, coderate=5/6)
            print("Each RxSquad UE transmitting {} streams, each with modulation order {}".format(rx_squad.num_streams_per_tx, rx_squad.num_bits_per_symbol_per_UE))

            # Recall that detected_info_bits_rxnode_wise_list is of shape [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
            # Transpose to shape [batch_size, num_clusters, num_rx_nodes, num_codewords, ldpc_k]
            dec_bits = tf.transpose(detected_info_bits_rxnode_wise_list, perm=[2, 1, 0, 3, 4]) # shape: [batch_size, num_clusters, num_rx_nodes, num_codewords, ldpc_k]
            # Now we need to extract the bits to be forwarded by each Rx node
            forwarding_bits = dec_bits[:,:,-(self.ns3cfg.num_rxue_sel):, : , :]
            dec_bits_phase_3, \
            node_wise_uncoded_ber_phase_3, \
            uncoded_ber_phase_3, \
            node_wise_coded_ber_phase_3, \
            coded_ber_phase_3, \
            node_wise_coded_bler_phase_3, \
            coded_bler_phase_3 = rx_squad(rxs_chans, forwarding_bits, min_codewords=32)
            # print("PHASE 3 STATS\nUNCODED BER: {}\nCODED BER: {}\nBLER: {}".format(uncoded_ber_phase_3 , coded_ber_phase_3, coded_bler_phase_3))
            # if uncoded_ber_phase_3 >= 1e-2 or coded_ber_phase_3 >= 1e-2:
            #     print("Warning: High RxSquad transmission BER")
            
            dec_bits_phase_3 = tf.reshape(dec_bits_phase_3, [dec_bits_phase_3.shape[0], forwarding_bits.shape[0], forwarding_bits.shape[1], forwarding_bits.shape[3], forwarding_bits.shape[4]])
            dec_bits_phase_3 = tf.transpose(dec_bits_phase_3, perm=[1, 2, 0, 3, 4])
            gNB_bits_phase_2 = dec_bits[:,:,:-(self.ns3cfg.num_rxue_sel), : , :]
            end_to_end_dec_bits = tf.concat([gNB_bits_phase_2, dec_bits_phase_3], axis=2)

            # transpose back to shape [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
            end_to_end_dec_bits = tf.transpose(end_to_end_dec_bits, perm=[2, 1, 0, 3, 4])
            
            # Here we multiply the median of the decoded LLRs with the sign of each decoded LLR to get the final decoded LLRs sent to the Rx base station
            # The new shape will still be the same as decoded_llrs_list which is [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
            received_llrs = decoded_llrs_list_forwarded*tf.sign(end_to_end_dec_bits - 0.5)

        else:
            # Here we multiply the median of the decoded LLRs with the sign of each decoded LLR to get the final decoded LLRs sent to the Rx base station
            # The new shape will still be the same as decoded_llrs_list which is [num_rx_nodes, num_clusters, batch_size, num_codewords, ldpc_k]
            received_llrs = decoded_llrs_list_forwarded*tf.sign(decoded_llrs_list)
        llr_based_info_bits = tf.cast(tf.reduce_sum(received_llrs, axis=0) > 0, tf.float32) # Sum LLRs from all Rx nodes shape : [num_clusters, batch_size, num_codewords, ldpc_k]
        # Recall that llr_based_info_bits is of shape [num_clusters, batch_size, num_codewords, ldpc_k]
        llr_based_coded_ber = compute_ber(tf.concat([llr_based_info_bits[i_cluster] for i_cluster in range(self.num_clusters)], axis=1) , tf.concat(info_bits_list, axis=1))
        llr_based_coded_bler = compute_bler(tf.concat([llr_based_info_bits[i_cluster] for i_cluster in range(self.num_clusters)], axis=1) , tf.concat(info_bits_list, axis=1))

        coded_ber = llr_based_coded_ber.numpy()
        coded_bler = llr_based_coded_bler.numpy()

        goodbits = (1.0 - coded_ber) * self.num_bits_per_frame
        userbits = (1.0 - coded_bler) * self.num_bits_per_frame
        
        uncoded_ber = 1.0 # There is no uncoded BER in this LLR combining scheme, so we set it to an invalid value of 1.0
        return [uncoded_ber , coded_ber, coded_bler], [goodbits, userbits]



def sim_mc_ncjt(cfg: SimConfig, ns3cfg: Ns3Config, cluster_ant_list:List[List[int]], modulation_order_list:List[int], **kwargs):
    """
    Simulation of single-cluster NCJT scenario
    """

    # dMIMO channels from ns-3 simulator
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True)

    # UE selection
    if cfg.enable_ue_selection is True:
        ns3cfg.reset_ue_selection()
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
        ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

    # Create multi-cluster NCJT simulation
    if cfg.ncjt_ldpc_decode_and_forward is True:
        mc_ncjt = MC_NCJT_LLR_Combining(cfg, ns3cfg, cluster_ant_list, modulation_order_list, dmimo_chans, **kwargs)
    else:
        mc_ncjt = MC_NCJT(cfg, ns3cfg, cluster_ant_list, modulation_order_list, dmimo_chans, **kwargs)

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
