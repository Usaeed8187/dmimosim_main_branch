import os
import sys
import numpy as np


def convert_ipc_channels(ipc_data_folder, ns3_chans_folder):

    # Load scenario configuration
    config = np.load(os.path.join(ipc_data_folder, '00_config.npz'), allow_pickle=True)['config'].item()
    
    # TxSquad channels, shape is [num_txue * num_ue_ant, num_bs_ant, num_ofdm_sym, num_subcarrier]
    Hts = np.zeros((
        config['numSquad1UEs'] * config['numUEAnt'],
        config['numBSAnt'],
        config['numSymsPerSubframe'],
        config['numSCs']),
        dtype=np.complex64)
    Hts[:, :, :, :] = np.nan
    # RxSquad channels, shape is [num_bs_ant, num_rxue * num_ue_ant, num_ofdm_sym, num_subcarrier]
    Hrs = np.zeros((
        config['numBSAnt'],
        config['numSquad2UEs'] * config['numUEAnt'],
        config['numSymsPerSubframe'],
        config['numSCs']),
        dtype=np.complex64)
    Hrs[:, :, :, :] = np.nan
    # dMIMO channels, shape is [num_rxs_ant, num_txs_ant, num_ofdm_sym, num_subcarrier]
    Hdm = np.zeros((
        config['numSquad2UEs'] * config['numUEAnt'] + config['numBSAnt'],
        config['numSquad1UEs'] * config['numUEAnt'] + config['numBSAnt'],
        config['numSymsPerSubframe'],
        config['numSCs']),
        dtype=np.complex64)
    Hdm[:, :, :, :] = np.nan
    # TxSquad pathloss in dB, shape is [num_txue, num_ofdm_sym]
    Lts = np.zeros((
        config['numSquad1UEs'],
        config['numSymsPerSubframe']),
        dtype=np.double)
    Lts[:, :] = np.nan
    # RxSquad pathloss in dB, shape is [num_rxue, num_ofdm_sym]
    Lrs = np.zeros((
        config['numSquad2UEs'],
        config['numSymsPerSubframe']),
        dtype=np.double)
    Lrs[:, :] = np.nan
    # dMIMO pathloss in dB, shape is [num_rxue+1, num_txue+1, num_ofdm_sym]
    Ldm = np.zeros((
        config['numSquad2UEs'] + 1,
        config['numSquad1UEs'] + 1,
        config['numSymsPerSubframe']),
        dtype=np.double)
    # External interference to RxSquad nodes
    num_extnodes = len(config['node_ids']['External_ids'])
    if num_extnodes > 0:
        # External channels, shape is [num_rxs_ant, num_ext_node, num_ofdm_sym, num_subcarrier]
        Hex = np.zeros((
            config['numSquad2UEs'] * config['numUEAnt'] + config['numBSAnt'],
            num_extnodes,
            config['numSymsPerSubframe'],
            config['numSCs']),
            dtype=np.complex64)
        # External pathloss in dB, shape is [num_rxue+1, num_ext_node, num_ofdm_sym]
        Lex = np.zeros((
            config['numSquad2UEs'] + 1,
            num_extnodes,
            config['numSymsPerSubframe']),
            dtype=np.double)

    for slot_idx in range(config['numSubframes']):
        # Load channel data for current slot
        tBS_id = config['node_ids']['BS1_id']
        tUE_ids = config['node_ids']['Squad1UE_ids']
        rBS_id = config['node_ids']['BS2_id']
        rUE_ids = config['node_ids']['Squad2UE_ids']
        ueAnts = config['numUEAnt']
        bsAnts = config['numBSAnt']
        for sym_in_sf in range(config['numSymsPerSubframe']):
            file_t = slot_idx * config['numSymsPerSubframe'] + sym_in_sf
            npz_filename = os.path.join(ipc_data_folder, f"ch_t_{file_t}.npz")
            with np.load(npz_filename, allow_pickle=True) as chfile:
                try:
                    propLosses = chfile['propagationLossesDb'].item()
                    Hmats = chfile['Hmats'].item()
                    # Load Hts and Lts
                    for i, tUEid in enumerate(tUE_ids):
                        start = i * ueAnts
                        end = (i + 1) * ueAnts
                        Hts[start:end, :, sym_in_sf, :] = Hmats[(tBS_id, tUEid)]
                        Lts[i, sym_in_sf] = propLosses[(tBS_id, tUEid)]
                    # Load Hrs and Lrs
                    for i, rUEid in enumerate(rUE_ids):
                        start = i * ueAnts
                        end = (i + 1) * ueAnts
                        Hrs[:, start:end, sym_in_sf, :] = Hmats[(rUEid, rBS_id)]
                        Lrs[i, sym_in_sf] = propLosses[(rUEid, rBS_id)]
                    # Load Hdm and Ldm
                    for i, tNodeId in enumerate([tBS_id] + tUE_ids):
                        if i == 0:
                            start = 0
                            end = bsAnts
                        else:
                            start = bsAnts + (i - 1) * ueAnts
                            end = bsAnts + i * ueAnts
                        for j, rNodeId in enumerate([rBS_id] + rUE_ids):
                            if j == 0:
                                start2 = 0
                                end2 = bsAnts
                            else:
                                start2 = bsAnts + (j - 1) * ueAnts
                                end2 = bsAnts + j * ueAnts
                            Hdm[start2:end2, start:end, sym_in_sf, :] = Hmats[(tNodeId, rNodeId)]
                            Ldm[j, i, sym_in_sf] = propLosses[(tNodeId, rNodeId)]
                    # External nodes
                    if num_extnodes > 0:
                        tEx_ids = config['node_ids']['External_ids']
                        for i, tExid in enumerate(tEx_ids):
                            for j, rNodeId in enumerate([rBS_id] + rUE_ids):
                                if j == 0:
                                    start = 0
                                    end = bsAnts
                                else:
                                    start = bsAnts + (j - 1) * ueAnts
                                    end = bsAnts + j * ueAnts

                                Hex[start:end, i:i+1, sym_in_sf, :] = Hmats[(tExid, rNodeId)]
                                Lex[j, i:i+1, sym_in_sf] = propLosses[(tExid, rNodeId)]

                except Exception as e:
                    print(f"Error reading channel data from {npz_filename}")
                    print(e)

        # save channel for current subframe/slot
        output_file = os.path.join(ns3_chans_folder, "dmimochans_{}.npz".format(slot_idx))
        if num_extnodes > 0:
            np.savez_compressed(output_file, Hdm=Hdm, Hrs=Hrs, Hts=Hts, Hex=Hex, Ldm=Ldm, Lrs=Lrs, Lts=Lts, Lex=Lex)
        else:
            np.savez_compressed(output_file, Hdm=Hdm, Hrs=Hrs, Hts=Hts, Ldm=Ldm, Lrs=Lrs, Lts=Lts)

    return config['numSubframes'] * config['numSymsPerSubframe']


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: convert_ipc_channels <ipc_data_folder> <ns3_chans_folder>")
        quit()

    input_data_folder = sys.argv[1]
    output_data_folder = sys.argv[2]
    if not os.path.exists(os.path.join(input_data_folder, "00_config.npz")):
        print("Error: The directory {} does not contain the channel files!".format(input_data_folder))
        quit()

    if os.path.exists(output_data_folder):
        print("Error: Output folder already exist! Will not overwrite previous ns-3 channel data.")
        quit()
    else:
        os.makedirs(output_data_folder)

    # Convert ns-3 channel for each subframe/slot
    num_ofdm_syms = convert_ipc_channels(input_data_folder, output_data_folder)
    print("\rFinish converting channels ({} snapshots)\n".format(num_ofdm_syms))

