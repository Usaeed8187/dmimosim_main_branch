import tensorflow as tf
import numpy as np

from sionna.utils import BinarySource
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN
from sionna.ofdm import ResourceGrid, ResourceGridMapper
from sionna.utils.metrics import compute_ber

from dmimo.config import Ns3Config, NcjtSimConfig
from dmimo.channel import LoadNs3Channel
from dmimo.stbc import alamouti_decode, alamouti_encode
from dmimo.stbc import hard_log_likelihood


def adjust_channel(h: tf.Tensor, pathloss_dB: tf.Tensor, nAntTxList, nAntRxList, cfg: NcjtSimConfig):
    """
    A function that adjusts the channel gains based on the transmit power of the transmitters and
    the pathlosses.

    :param tf.Tensor h: the small scale channel gains
    :param tf.Tensor pathloss_dB: Path loss tensor (in dB)
    :param list nAntTxList: A list containing the number of antennas of each transmit node
    :param list nAntRxList: A list containing the number of antennas of each receive node
    :param NcjtSimConfig cfg: a ``NcjtSimConfig`` configuration file.
    :return: Channel gains with transmit power and pathloss applied
    """

    num_tx = pathloss_dB.shape[3]
    num_rx = pathloss_dB.shape[2]
    assert len(nAntTxList) == num_tx and len(nAntRxList) == num_rx

    transmit_pwr = tf.reshape(tf.convert_to_tensor([cfg.BsTxPwrdB] + [cfg.UeTxPwrdB for _ in range(num_tx - 1)]),
                              [1, 1, 1, num_tx, 1])
    # rx_pwr_raw has shape [num_subframes, 1, num_rx, num_tx, num_ofdm_symbols]
    rx_pwr_raw = transmit_pwr + cfg.ANTENNA_GAIN - pathloss_dB + cfg.EQUALIZER
    # Reshape rx_pwr_raw tensor into (num_subframes, 1, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, 1)
    tensor_list = []
    for i in range(num_rx):
        tensor_list.extend([rx_pwr_raw[:, :, i:i + 1, :, :] for _ in range(nAntRxList[i])])
    rx_pwr_raw = tf.concat(tensor_list, axis=2)

    tensor_list = []
    for i in range(num_tx):
        tensor_list.extend([rx_pwr_raw[:, :, :, i:i + 1, :] - 10 * np.log10(nAntTxList[i]) for _ in
                            range(nAntTxList[i])])  # Adjusting power according to number of transmit antennas
    rx_pwr = tf.concat(tensor_list, axis=3)

    # reshape to [num_subframes, 1, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, 1]
    rx_pwr = tf.reshape(rx_pwr, (*h.shape[:-1], 1))
    h_adjusted = h * tf.cast(tf.sqrt(10 ** (rx_pwr / 10)), h.dtype)
    # h_adjusted shape [num_subframes, num_subcarriers, num_ofdm_symbols, total_rx_antennas, total_tx_antennas]
    h_adjusted = tf.transpose(h_adjusted[:, 0, :, 0, :, :, :], (0, 4, 3, 1, 2))

    return h_adjusted


def ncjt_first_phase(bit_stream: tf.Tensor, h_freq: tf.Tensor, mapper: Mapper, demapper: Demapper):
    """
    This function simulates the broadcast channel of the first phase
    
    :param tf.Tensor bit_stream:
        A tensor representing the bit_stream to be sent.
    :param tf.Tensor h_freq:
        The channel matrix from the Tx base station to the Tx UEs
    :param Mapper mapper: Sionna-based modulator
    :param Demapper demapper: Sionna-based demodulator
    """


def ncjt_alamouti_dmimo_phase(bit_streams: tf.Tensor, channel: tf.Tensor, cfg: NcjtSimConfig,
                              mapper: Mapper, demapper: Demapper, awgn: AWGN, no: float,
                              do_demapping=False):
    """
    This function simulates the non-coherent transmission of symbols using the Alamouti scheme
    in the dMIMO phase. This function assumes perfect channel estimate.

    :param tf.Tensor bit_streams: 
        A `(...,M)`-shaped ``Tensor`` representing the bit stream to transmit on
        each of the `M` transmitters. If there are no errors in the first phase, then all the
        M bit streams should be the same.
    :param tf.Tensor channel: 
        A `(...,Mr,Nt)`-shaped channel that represents the dMIMO channel
    :param NcjtSimConfig cfg: a ``NcjtSimConfig`` configuration file.
    :param Mapper mapper: Sionna-based modulator
    :param Demapper demapper: Sionna-based demodulator
    :param AWGN awgn: Sionna-based AWGN layer for adding noise to the received signal
    :param float no: Noise power in linear scale
    :param ResourceGridMapper rg_mapper: A Sionna-based resource grid mapper to map the symbols to the resource grid that includes pilots
    :param do_demapping: Enable demapping procedure
    :return y_list: List of Alamouti-decoded received signals, ready to be fed to a demapper

    :return gains_list: List of SNR gains achieved by Alamouti coding and decoding (i.e. |h1|**2 + |h2|**2).
    """

    rg_phase2 = ResourceGrid(num_ofdm_symbols=cfg.num_ofdm_symbols,
                      fft_size=cfg.num_subcarriers,
                      subcarrier_spacing=15e3,
                      num_tx=2,
                      num_streams_per_tx=1,
                      cyclic_prefix_length=64,
                      num_guard_carriers=[0, 0],
                      dc_null=False,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=cfg.pilot_syms)
    rg_mapper_phase2 = ResourceGridMapper(rg_phase2)
    # Alamouti encoding on each transmit node
    x_list = []
    for i_tx in range(cfg.num_TxBs + cfg.num_TxUe):
        nAnt = cfg.nAntTxBs if (i_tx == cfg.num_TxBs - 1) else cfg.nAntTxUe
        x = mapper(bit_streams[..., i_tx])  # [num_subframes, num_subcarriers, len(data_syms)]
        x = alamouti_encode(x)  # [num_subframes, num_subcarriers, len(data_syms), 2]
        # Transpose to make the signal compatible with rg_mapper
        x = tf.transpose(x, [0,3,2,1]) # [num_subframes, 2, len(data_syms), num_subcarriers]
        x = rg_mapper_phase2(tf.reshape(x,[cfg.num_subframes_phase2,2,1,len(cfg.data_syms)* cfg.num_subcarriers])) # Becuase input to rg_mapper must be of shape [batch_size,num_tx, num_streams_per_tx, num_data_symbols]
        # x.shape is [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] which would be [num_subframes, 2, 1, num_ofdm_symbols, num_subcarriers]
        x = tf.reshape(x , [cfg.num_subframes_phase2,2,cfg.num_ofdm_symbols, cfg.num_subcarriers])
        x = tf.transpose(x, [0,3,2,1]) # [num_subframes, num_subcarriers, num_ofdm_symbols, 2]
        x = tf.concat([x for _ in range(nAnt // 2)], axis=-1) # In case there are more than 2 antennas on the transmitter
        # new x shape [num_subframes, num_subcarriers, num_ofdm_symbols, total_tx_antennas]
        x_list.append(x)
    x = tf.concat(x_list, axis=-1)  # [num_subframes, num_subcarriers, num_ofdm_symbols, total_tx_antennas]
    x = x[..., tf.newaxis]  # [num_subframes, num_subcarriers, num_ofdm_symbols, total_tx_antennas , 1]

    # Figure out the channel based on the set of transmit and receive nodes participating
    if cfg.num_TxBs == 0:
        tx_antenna_slice = slice(cfg.nAntTxBs, cfg.num_TxUe * cfg.nAntTxUe)
    elif cfg.num_TxBs == 1:
        tx_antenna_slice = slice(cfg.nAntTxBs + cfg.num_TxUe * cfg.nAntTxUe)
    else:
        raise ValueError('Number of TX base stations has to be either 0 or 1.')
    if cfg.num_RxBs == 0:
        rx_antenna_slice = slice(cfg.nAntRxBs, cfg.num_RxUe * cfg.nAntRxUe)
    elif cfg.num_RxBs == 1:
        rx_antenna_slice = slice(cfg.nAntRxBs + cfg.num_RxUe * cfg.nAntRxUe)
    else:
        raise ValueError('Number of RX base stations has to be either 0 or 1.')

    h_freq_ns3 = channel[..., rx_antenna_slice, tx_antenna_slice]
    total_tx_antennas = h_freq_ns3.shape[-1]
    total_rx_antennas = h_freq_ns3.shape[-2]

    # Pass through the channel
    ry = tf.linalg.matmul(h_freq_ns3, x)  # (num_subframes, num_subcarriers, num_ofdm_symbols, total_rx_antennas, 1)
    ry = tf.gather(ry, indices=cfg.data_syms,axis=2) # (num_subframes, num_subcarriers, len(data_syms), total_rx_antennas, 1)
    num_ofdm_symbols = ry.shape[-3]
    total_rx_antennas = ry.shape[-2]
    # Reshaping to be compatible with the alamouti_encode function
    # ry shape [num_subframes, num_subcarriers, num_ofdm_symbols/2 , 2, total_rx_antennas]
    ry = tf.reshape(ry, (*ry.shape[:-3], num_ofdm_symbols // 2, 2, total_rx_antennas))

    # Add noise
    no = tf.cast(no, (tf.float32 if ry.dtype == tf.complex64 else tf.float64))
    ry_noisy = awgn([ry, no])  # Add AWGN to the signal

    # Alamouti decoding

    ## Donald, Here is the place to do channel estimation for each Rx node separately. 
    ## Donald, The indices for the pilots are cfg.pilot_syms 
    ## Donald, I am currently using h_freq_ns3 as the perfect channel estimate. Please replace this with 
    # h_freq_ns3_estimated with shape (num_subframes, num_subcarriers, num_ofdm_symbols, total_rx_antennas, total_tx_antennas)
    h_freq_ns3_estimated = h_freq_ns3
    h_freq_ns3_estimated = tf.gather(h_freq_ns3_estimated, indices=cfg.data_syms, axis=2) # (num_subframes, num_subcarriers, len(data_syms), total_rx_antennas, total_tx_antennas)

    # Here we have an issue. Alamouti assumes that in two consecutive OFDM symbols the channel stays the same. 
    # but that isn't generally true. In any case, we are going to feed the average of two consecutive OFDM symbol
    # channel to the STBC decoder.
    # (num_subframes, num_subcarriers, num_ofdm_symbols/2, total_rx_antennas, total_tx_antennas)
    h_freq_ns3_averaged = (h_freq_ns3_estimated[..., ::2, :, :] + h_freq_ns3_estimated[..., 1::2, :, :]) / 2
    # Now we need to sum over the respective transmit antennas
    h_freq_ns3_averaged = tf.add_n([h_freq_ns3_averaged[..., i * 2:i * 2 + 2] for i in range(total_tx_antennas // 2)])
    # new shape is [num_subframes, num_subcarriers, len(data_syms)/2, total_rx_antennas, 2]

    y_list = []
    gains_list = []
    for i in range(cfg.num_RxBs + cfg.num_RxUe):
        if i == cfg.num_RxBs - 1:
            rx_antenna_slice = slice(i * cfg.nAntRxBs, (i + 1) * cfg.nAntRxBs)
        else:
            rx_antenna_slice = slice(cfg.num_RxBs * cfg.nAntRxBs + (i - cfg.num_RxBs) * cfg.nAntRxUe,
                                     cfg.num_RxBs * cfg.nAntRxBs + (i - cfg.num_RxBs + 1) * cfg.nAntRxUe)
        # assuming perfect CSI # (num_subframes, num_subcarriers, num_ofdm_symbols)
        y, gains = alamouti_decode(ry_noisy[..., rx_antenna_slice], h_freq_ns3_averaged[..., rx_antenna_slice, :])
        # y.shape = gains.shape = (num_subframes, num_subcarriers, num_ofdm_symbols)
        y = y / tf.cast(gains, y.dtype)
        # Turn into bits. Shape = (num_subframes, num_subcarriers, num_ofdm_symbols*cfg.num_bits_per_symbol_phase2)
        if do_demapping:
            y = demapper([y, no])
        y_list.append(y)
        gains_list.append(gains)

    return y_list, gains_list


def ncjt_third_phase():
    """
    This function simulates the transmission of the bit streams available at each 
    Rx user back to the base station.
    """


def ncjt_sim_all_phases(cfg: NcjtSimConfig, perfect_phase1=True, perfect_phase3=True):
    """
    This function simulates all the 3 phases of the dMIMO setup and returns the average BER.

    :param NcjtSimConfig cfg: 
        NcjtSimConfig configuration settings for the simulation
    :param bool perfect_phase1: 
        Whether the transmission of bits from Tx BS to Tx UEs is perfectly done without any errors
    :param bool perfect_phase3: 
        Whether the transmission of detected bits from Rx UEs to Rx BS is perfectly done without any errors
    :return avg_ber: 
        Average BER over the course of this simulation
    :return h_dmimo: 
        dMIMO channel extracted from the NS3. This is returned to later be fed back to this function again
        so that the relatively slow process of NS3 channel extraction is not repeated.
    """

    # Channel loading

    ns3_config = Ns3Config(total_slots=cfg.starting_subframe + cfg.num_subframes, _data_folder=cfg.ns3_folder)
    ns3_channel = LoadNs3Channel(ns3_config)
    h_dmimo, pathlossdb_dmimo_raw = ns3_channel("dMIMO-Raw", slot_idx=cfg.starting_subframe + cfg.num_subframes_phase1,
                                                batch_size=cfg.num_subframes_phase2)
    # batch_size is the same as num_subframes
    # h_dmimo.shape = (num_subframes, 1, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, fft_size)
    # pathlossdb_dmimo_raw.shape = (num_subframes, 1, num_rx, num_tx, num_ofdm_symbols)
    nAntRx_list = [4] + [2 for _ in range(10)]
    nAntTx_list = [4] + [2 for _ in range(10)]
    h_dmimo = adjust_channel(h_dmimo, pathlossdb_dmimo_raw, nAntRx_list, nAntTx_list, cfg)
    # (num_subframes, num_subcarriers, num_ofdm_symbols, total_rx_antennas, total_tx_antennas)

    binary_source = BinarySource()
    add_noise = AWGN()
    # Donald: I don't know what parameters to choose here for rg for phase 1
    rg_phase1 = ResourceGrid(num_ofdm_symbols=cfg.num_ofdm_symbols,
                      fft_size=cfg.num_subcarriers,
                      subcarrier_spacing=15e3,
                      num_tx=1,
                      num_streams_per_tx=1,
                      cyclic_prefix_length=64,
                      num_guard_carriers=[0, 0],
                      dc_null=False,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=cfg.pilot_syms)
    rg_mapper_phase1 = ResourceGridMapper(rg_phase1)
    bit_stream = binary_source(
        [cfg.num_subframes_phase1, cfg.num_subcarriers, len(cfg.data_syms) * cfg.num_bits_per_symbol_phase1])

    # Tx Squad Phase
    if perfect_phase1:
        bit_stream_dmimo = tf.stack([tf.reshape(bit_stream,
                                                (cfg.num_subframes_phase2,
                                                 cfg.num_subcarriers,
                                                 len(cfg.data_syms) * cfg.num_bits_per_symbol_phase2))
                                     for _ in range(cfg.num_TxBs + cfg.num_TxUe)], axis=-1)
    else:
        mapper_phase1 = Mapper('qam',cfg.num_bits_per_symbol_phase1)
        x = mapper_phase1(bit_stream)
        x = rg_mapper_phase1(tf.reshape(x,[cfg.num_subframes_phase1,1,1,cfg.num_subcarriers* len(cfg.data_syms)])) # Becuase input to rg_mapper must be of shape [batch_size,num_tx, num_streams_per_tx, num_data_symbols]
        # x.shape is [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] which would be [num_subframes, 1, 1, num_ofdm_symbols, num_subcarriers]
        x = tf.reshape(x , [cfg.num_subframes_phase1,cfg.num_ofdm_symbols, cfg.num_subcarriers])
        x = tf.transpose(x, [0,2,1]) # [num_subframes, num_subcarriers, num_ofdm_symbols]
        # Donald: x needs to be broadcast to the Tx UEs.
        raise ValueError('Only perfect Tx Squad phase is supported for now.')

    # dMIMO Phase
    mapper_dmimo = Mapper("qam", cfg.num_bits_per_symbol_phase2)
    demapper_dmimo = Demapper("maxlog", "qam", cfg.num_bits_per_symbol_phase2, hard_out=True)

    no = (10 ** ((cfg.NOISE_FLOOR + cfg.NOISE_FIGURE + cfg.EQUALIZER) / 10))
    rx_bit_stream_dmimo_list, gains_list = ncjt_alamouti_dmimo_phase(bit_stream_dmimo, h_dmimo, cfg, mapper_dmimo,
                                                                     demapper_dmimo, add_noise, no, do_demapping=True)

    # Rx Squad phase
    if perfect_phase3:
        bit_streams_at_RxBs = rx_bit_stream_dmimo_list
    else:
        # Donald: Remember to use cfg.num_bits_per_symbol_phase3 for creating a mapper/demapper here.
        raise ValueError('Only perfect Tx Squad phase is supported for now.')

    if cfg.perSC_SNR:  # In case we have the exact gain of each subcarrier
        snrs = tf.stack(gains_list, axis=-1) / no
    else:  # In case only per subframe SNR gain is available at the Rx BS
        snrs = tf.stack(gains_list, axis=-1) / no
        snrs = tf.ones_like(snrs) * tf.reduce_mean(snrs, axis=(-2, -3),
                                                   keepdims=True)  # Averaging over OFDM symbols and subcarriers

    # Post detection combining
    symbol_streams_at_RxBs = [mapper_dmimo(rx_node_bit_stream) for rx_node_bit_stream in bit_streams_at_RxBs]
    # y_combined shape [cfg.num_subframes_phase2, num_ofdm_symbols]
    y_combined = hard_log_likelihood(tf.stack(symbol_streams_at_RxBs, axis=-1), snrs,
                                     k_constellation=cfg.num_bits_per_symbol_phase2)
    # detected_bits shape [cfg.num_subframes_phase2,num_subcarriers, num_ofdm_symbols * cfg.num_bits_per_symbol_phase2]
    detected_bits = demapper_dmimo([y_combined, no])
    detected_bits = tf.reshape(detected_bits, (cfg.num_subframes_phase1, cfg.num_subcarriers,
                                               (len(cfg.data_syms)) * cfg.num_bits_per_symbol_phase1))
    avg_ber = compute_ber(detected_bits, bit_stream)

    return avg_ber
