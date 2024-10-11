"""
Simulation of single-cluster NCJT scenario

"""

import os
import numpy as np
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
from dmimo.ncjt import NCJT_TxUE, NCJT_RxUE, NCJT_PostCombination


class SC_NCJT(Model):

    def __init__(self, cfg: SimConfig, **kwargs):
        super().__init__(kwargs)

        self.cfg = cfg
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2

        # Update number of QAM symbols for data and LDPC params
        self.num_data_symbols = cfg.fft_size * (cfg.symbols_per_slot - len(cfg.pilot_indices))
        self.ldpc_n = int(2 * self.num_data_symbols)  # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * cfg.code_rate)  # Number of information bits
        self.num_codewords = cfg.modulation_order // 2  # number of codewords per frame
        self.num_bits_per_frame = self.ldpc_k * self.num_codewords
        self.num_uncoded_bits_per_frame = self.ldpc_n * self.num_codewords

        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True, num_iter=6)

        # Fixed interleaver design for current RG setting
        self.intlvr = RowColumnInterleaver(self.num_data_symbols // 2, axis=-1)
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)

        if self.cfg.perfect_csi is False:
            # TODO: Auto generate LMMSE weights
            filename = os.path.abspath(os.path.join(self.cfg.ns3_folder + "/../lmmse_weight.npy"))
            self.Wf = np.load(filename)
            self.Wf = tf.constant(tf.convert_to_tensor(self.Wf, dtype=tf.complex64))

        self.ncjt_tx = NCJT_TxUE(cfg)
        self.ncjt_rx = NCJT_RxUE(cfg, lmmse_weights=self.Wf)
        self.ncjt_combination = NCJT_PostCombination(cfg)

    def call(self, dmimo_chans: dMIMOChannels):

        # Tx gNB processing

        # The binary source will create batches of information bits
        info_bits = self.binary_source([self.batch_size, self.num_codewords, self.encoder.k])

        # LDPC encoder processing
        c = self.encoder(info_bits)

        # Interleaving for coded bits
        c = self.intlvr(c)

        # Phase 1 downlink transmission
        # TODO: assuming ideal transmission for now
        tx_bit_stream = tf.reshape(c, [self.batch_size, self.cfg.fft_size, -1])

        # Phase 2 transmission from all gNB/UEs
        tx_signals_list = []
        for ue_idx in range(0, self.cfg.num_tx_ue_sel + 1):
            ue_tx_signal = self.ncjt_tx(tx_bit_stream, is_txbs=(ue_idx == 0))
            tx_signals_list.append(ue_tx_signal)
        tx_signals = tf.concat(tx_signals_list, axis=-1)
        # new shape [batch_size, num_tx_ant, num_ofdm_sym, fft_size)
        tx_signals = tf.transpose(tx_signals, [0, 3, 2, 1])
        tx_signals = tf.expand_dims(tx_signals, axis=1)

        # apply dMIMO channels to the resource grid in the frequency domain
        ry = dmimo_chans([tx_signals, self.cfg.first_slot_idx])
        ry = tf.transpose(ry, [0, 4, 3, 2, 1])

        # Rx Squad processing
        y_list = []
        gains_list = []
        nvar_list = []
        for ue_idx in range(0, self.cfg.num_rx_ue_sel + 1):
            if ue_idx == 0:
                y, gains, nvar = self.ncjt_rx(ry[:, :, :, 0:4, :])
            else:
                y, gains, nvar = self.ncjt_rx(ry[:, :, :, ((ue_idx * 2) + 2):((ue_idx * 2) + 4), :])
            y_list.append(y)
            gains_list.append(gains)
            nvar_list.append(nvar)

        nvar = np.mean(nvar_list)  # FIXME

        # Phase 3 uplink transmission
        # TODO: assuming ideal transmission for now

        # Post-detection combining
        combination_output = self.ncjt_combination(y_list, gains_list, nvar)
        detected_bits = tf.cast(combination_output > 0, tf.float32)
        # [batch_size, num_subcarriers, num_data_syms*num_bits_per_symbol]
        detected_bits = tf.reshape(detected_bits, tx_bit_stream.shape)

        # LDPC Decoding
        combination_llr = tf.reshape(combination_output, [self.batch_size, self.num_codewords, self.ldpc_n])
        combination_llr = self.dintlvr(combination_llr)
        detected_info_bits = self.decoder(combination_llr)  # [cfg.num_subframes, num_codewords, ldpc_n]

        # Error statistics
        uncoded_ber = compute_ber(detected_bits, tx_bit_stream)
        coded_ber = compute_ber(info_bits, detected_info_bits).numpy()
        coded_bler = compute_bler(info_bits, detected_info_bits).numpy()

        # Goodput and throughput estimation
        goodbits = (1.0 - coded_ber) * self.num_bits_per_frame
        userbits = (1.0 - coded_bler) * self.num_bits_per_frame

        return [uncoded_ber, coded_ber, coded_bler], [goodbits, userbits]


def sim_sc_ncjt(cfg: SimConfig):
    """
    Simulation of single-cluster NCJT scenario
    """

    # dMIMO channels from ns-3 simulator
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True)

    # UE selection
    if cfg.enable_ue_selection is True:
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
        # Update will be applied to dMIMOChannels object
        ns3cfg.update_ue_mask(tx_ue_mask, rx_ue_mask)

    # Create single-cluster NCJT simulation
    sc_ncjt = SC_NCJT(cfg)

    # Loop over channels for all transmission cycles
    total_cycles = 0
    uncoded_ber, coded_ber, coded_bler, goodput, throughput = 0, 0, 0, 0, 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        # Run simulation for one cycle
        bers, bits = sc_ncjt(dmimo_chans)
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
