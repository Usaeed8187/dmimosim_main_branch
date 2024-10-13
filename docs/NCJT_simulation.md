## Non-Coherent Joint Transmission (NCJT) System Simulation

## Overview
This document provides details and guidance on simulation of the non-coherent joint transmission (NCJT) scheme.
In the NCJT scheme, the transmitters need to be synchronized only on the symbol level. On top of that, the transmitters do not require channel feedback from the receivers.
In this project we make use of space-time block codes (STBC) for the NCJT scheme transmission. The Alamouti STBC is the STBC used in this project. 

The "**sims**" folder contains the top-level simulation script the NCJT scheme in the "**sim_sc_ncjt.py**" file. Running this file generates BER results that will be printed as output.
<!-- that will be stored in the "**results/NCJT_Alamouti_QTR3/**" folder. -->

The file "**dmimo/sc_ncjt.py**" contains the necessary building blocks for simulating the NCJT scheme.
These building blocks include:
1. Loading the channels from NS3
1. Encdoding and decoding of the Alamouti STBC
1. Post detection combining of the received symbols of each receiver node.
1. Calculation of BER.

The function ``sim_sc_ncjt`` in "**dmimo/sc_ncjt.py**" simulates the dMIMO transmission phase in the for the NCJT scheme.  Currently, phase 1 and phase 3 of the transmission is not simulated for this scheme and it is assumed that they are done perfectly. 
For this end, an instance of ``SC_NCJT`` is created which represents the single cluster NCJT scheme simulator.
Within ``SC_NCJT``, three main components are used: ``NCJT_TxUE``, ``NCJT_RxUE`` and ``NCJT_PostCombination``. 
The first two handle modulation/demodulation of bits and Alamouti encdoing and decoding of symbols.
THe ``NCJT_PostCombination`` component is responsible for combining the data symbols received by each of the receiver nodes.
Outside of these three components, the ``SC_NCJT`` layer also handles the LDPC coding/decoding of information data, as well as BER, BLER and throughput calculation.

We will go over the three components in what follows.

### Class ``NCJT_TxUE``
Implements the Alamouti transmission for each transmitter.
Configuration parameters used when instantiating an object of this class:

* ``cfg.nAntTxBs``: Number of antennas in the Tx base station, default 4.
* ``cfg.nAntTxUe``: Number of antennas in the Tx user equipment, default 2.
* ``cfg.modulation_order``: Number of bits per symbol, aka the modulation order, default 2.
* ``cfg.symbols_per_slot``: Number of OFDM symbols in every slot, default 14.
* ``cfg.fft_size``: Number of subcarriers, default 512.
* ``cfg.subcarrier_spacing``: Number of subcarriers, default 512.
* ``cfg.cyclic_prefix_len``: The number of samples for the cyclic prefix, default 64.
* ``cfg.pilot_indices``: Indices of OFDM symbols used for pilots, default `[2,11]`.

Arguments used for calling an object of this class:
* ``bit_stream_dmimo``: A `Tensor` of shape `[batch_size, num_subcarriers, num_data_ofdm_syms * modulation_order]`. This will be mapped to symbols according to ``cfg.modulation_order``.
* `is_txbs`: Whether or not this particular transmitter is a base station. If not, it will be a UE.

This call will return the Alamouti-encoded signal ready to be transmitted over the air.

### Class ``NCJT_RxUE``
Implements the reception of the signal, channel estimation and Alamouti decoding of the signal at each receiver node. Instantiating an object of this class requires a confuguration variable `cfg` and a matrix `lmmse_weights`. The `lmmse_weights` matrix represents the covariance matrix between time-frequency bins in the OFDM grid and is used for LMMSE channel estimation.

Configuration parameters used when instantiating an object of this class include:

* ``cfg.symbols_per_slot``: Number of OFDM symbols in every slot, default 14.
* ``cfg.pilot_indices``: Indices of OFDM symbols used for pilots, default `[2,11]`.
* ``cfg.modulation_order``: Number of bits per symbol, aka the modulation order, default 2.
* ``cfg.fft_size``: Number of subcarriers, default 512.
* ``cfg.subcarrier_spacing``: Number of subcarriers, default 512.
* ``cfg.cyclic_prefix_len``: The number of samples for the cyclic prefix, default 64.
* ``cfg.perfect_csi``: whether or not the channel estimation is perfect, default `False`.

Arguments used for calling an object of this class:
* ``ry_noisy``: Received signal `Tensor` of shape `[batch_size, num_subcarriers, symbols_per_slot, total_rx_antennas, 1]`. 
* `h_freq_ns3`: The perfect channel. Used only when `cfg.perfect_csi` is `True`. Shape should be `[batch_size, num_subcarriers, symbols_per_slot, total_rx_antennas, total_tx_antennas]`

This call will do channel estimation, Alamouti decoding and bit detection. It returns the following:
* `y`: The detected symbols, with shape `[num_subframes, num_subcarriers, num_ofdm_symbols]`
* `gains`: The SNR gain offered by Alamouti decoding, with same shape as `y`.
* `nvar`: Estimated noise variance.

### Class ``NCJT_PostCombination``
Used for post detection combination of the detected symbols of all receiver nodes. For instantiation, the following arguments are required:
* `cfg`: `SimConfig` configuration object.
* `return_LLRs`: Boolean variable determining whether to return log likelihood ratios of the bits or the detected bits. The LLRs are usually returned when LDPC coding is used. Default `False`.
* `perSC_SNR`: Whether or not the post detection combiner has access to per-subcarrier SNR. 

Calling an object of this class requires the following arguments:
* `rx_bit_streams`: The list of detected bit streams of each of the receivers.
* `gains_list`: The SNR gains provided by Alamouti decoding.
* `no`: Noise variance.
