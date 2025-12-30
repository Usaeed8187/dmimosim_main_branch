import sys
from pathlib import Path
from typing import List, Optional, Tuple
import itertools

import numpy as np

# Make the ICML_DEQN_clean folder importable
REPO_ROOT = Path(__file__).resolve().parents[2]
DEQN_PATH = REPO_ROOT / "ICML_DEQN_clean"
if str(DEQN_PATH) not in sys.path:
    sys.path.append(str(DEQN_PATH))

from ICML_DEQN_clean.DQN_RC_new_WESN import DeepWESNQNetwork  # noqa: E402


def _flatten_w1_indices(w1_entry) -> np.ndarray:
    """Flatten the w1_beam_indices structure into a 1-D numpy array."""

    if w1_entry is None:
        return np.array([], dtype=np.float32)

    if isinstance(w1_entry, (list, tuple)):
        parts = []
        for item in w1_entry:
            flattened = _flatten_w1_indices(item)
            if flattened.size > 0:
                parts.append(flattened)
        if len(parts) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts)

    return np.atleast_1d(np.array(w1_entry, dtype=np.float32)).flatten()


def _extract_w1_from_feedback(pmi_feedback_bits) -> List:
    """Extract raw w1_beam_indices for each receiver from PMI feedback bits."""

    if pmi_feedback_bits is None:
        return []

    w1_list = []
    entries = pmi_feedback_bits if isinstance(pmi_feedback_bits, list) else [pmi_feedback_bits]
    for rx_entry in entries:
        if isinstance(rx_entry, dict) and "w1_beam_indices" in rx_entry:
            w1_list.append(rx_entry.get("w1_beam_indices"))
        else:
            w1_list.append(None)
    return w1_list


def _structure_to_tuple(struct):
    if struct is None:
        return tuple()
    if isinstance(struct, (list, tuple)):
        return tuple(_structure_to_tuple(s) for s in struct)
    return int(np.array(struct).flatten()[0])


def _tuple_to_list(struct):
    if not isinstance(struct, tuple):
        return struct
    out = []
    for item in struct:
        if isinstance(item, tuple):
            out.append(_tuple_to_list(item))
        else:
            out.append(int(item))
    return out

def _enumerate_beam_sets(N1, O1, N2, O2, L):
    """
    Enumerate all possible beam index sets allowed by Algorithm 1 (structure only).

    Returns:
        list of sorted tuples of beam indices
    """

    beam_sets = set()

    # possible offsets
    q1_vals = range(O1)
    q2_vals = range(O2)

    for q1 in q1_vals:
        for q2 in q2_vals:

            # possible coarse indices
            n1_vals = range(N1)
            n2_vals = range(N2)

            # choose distinct n1's
            for n1_sel in itertools.combinations(n1_vals, L):

                if N2 == 1:
                    # n2 is fixed
                    n2_sel = [0] * L
                    beams = [
                        (O2 * 0 + q2) * (O1 * N1) + (O1 * n1 + q1)
                        for n1 in n1_sel
                    ]
                    beam_sets.add(tuple(sorted(beams)))
                else:
                    # choose distinct n2's
                    for n2_sel in itertools.permutations(n2_vals, L):
                        if len(set(n2_sel)) < L:
                            continue

                        beams = [
                            (O2 * n2 + q2) * (O1 * N1) + (O1 * n1 + q1)
                            for n1, n2 in zip(n1_sel, n2_sel)
                        ]
                        beam_sets.add(tuple(sorted(beams)))

    return sorted(beam_sets)

def _ensure_1d_array(arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32).flatten()


class RLBeamSelector:
    """Lightweight manager to run DEQN agents for PMI beam predictions."""

    def __init__(
        self,
        max_actions: int = 128,
        memory_size: int = 200,
        input_window_size: int = 3,
        output_window_size: int = 3,
        use_enumerated_actions: bool = True,
    ):
        
        self.O2 = 1
        self.N2 = 1
        self.O1 = 4
        
        self.max_actions = max_actions
        self.memory_size = memory_size
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.use_enumerated_actions = use_enumerated_actions

        self.agents: List[List[Optional[DeepWESNQNetwork]]] = []
        self.action_maps: List[List[List[Tuple[int, ...]]]] = []
        self.prev_states: List[List[Optional[np.ndarray]]] = []
        self.prev_actions: List[List[Optional[int]]] = []
        self.state_dims: List[List[Optional[int]]] = []


    def _ensure_pair_capacity(self, rx_idx: int, tx_idx: int):
        while len(self.agents) <= rx_idx:
            self.agents.append([])
            self.action_maps.append([])
            self.prev_states.append([])
            self.prev_actions.append([])
            self.state_dims.append([])

        while len(self.agents[rx_idx]) <= tx_idx:
            self.agents[rx_idx].append(None)
            self.action_maps[rx_idx].append([])
            self.prev_states[rx_idx].append(None)
            self.prev_actions[rx_idx].append(None)
            self.state_dims[rx_idx].append(None)

    def _maybe_init_agent(self, rx_idx: int, tx_idx: int, state_dim: int):
        self._ensure_pair_capacity(rx_idx, tx_idx)

        if self.agents[rx_idx][tx_idx] is None:
            
            self.agents[rx_idx][tx_idx] = DeepWESNQNetwork(
                self.max_actions,
                state_dim,
                self.input_window_size,
                self.output_window_size,
                self.memory_size,
                n_layers=1,
                nInternalUnits=64,
                spectral_radius=0.3,
            )
            self.state_dims[rx_idx][tx_idx] = state_dim

    def _canonical_action(self, beam_struct, L: int) -> Optional[Tuple[int, ...]]:
        beams = _flatten_w1_indices(beam_struct).astype(int)

        unique_beams: List[int] = []
        seen = set()

        for beam in beams.flatten():
            if len(unique_beams) >= L:
                break
            if not 0 <= int(beam) < self.num_beams:
                continue
            if int(beam) in seen:
                continue
            unique_beams.append(int(beam))
            seen.add(int(beam))

        candidate = 0
        while len(unique_beams) < L and self.num_beams > 0:
            if candidate not in seen and 0 <= candidate < self.num_beams:
                unique_beams.append(candidate)
                seen.add(candidate)
            candidate += 1

        if len(unique_beams) < L:
            return None

        return tuple(unique_beams[:L])

    def _register_action(self, rx_idx: int, tx_idx: int, beam_struct, L: int) -> Optional[int]:
        beam_tuple = self._canonical_action(beam_struct, L)
        if beam_tuple is None:
            return None
        
        existing = self.action_maps[rx_idx][tx_idx]

        if self.use_enumerated_actions:
            if not existing or len(existing[0]) != L:
                self.action_maps[rx_idx][tx_idx] = _enumerate_beam_sets(
                    self.N1, self.O1, self.N2, self.O2, L
                )
                existing = self.action_maps[rx_idx][tx_idx]

            sorted_beam = tuple(sorted(beam_tuple))
            for idx, saved in enumerate(existing):
                if sorted_beam == saved:
                    return idx

            return 0 if existing else None


        for idx, saved in enumerate(existing):
            if saved == beam_tuple:
                return idx

        if len(existing) < self.max_actions:
            self.action_maps[rx_idx][tx_idx].append(beam_tuple)
            return len(existing)

        return 0

    def _decode_action(self, rx_idx: int, tx_idx: int, action_idx: int) -> Optional[Tuple[int, ...]]:
        if action_idx is None:
            return None
        if rx_idx >= len(self.action_maps) or tx_idx >= len(self.action_maps[rx_idx]):
            return None
        if 0 <= action_idx < len(self.action_maps[rx_idx][tx_idx]):
            return self.action_maps[rx_idx][tx_idx][action_idx]

        return None

    def _build_state(self, prev_action_idx: int, sinr_vec: np.ndarray) -> np.ndarray:
        prev_action_arr = np.array([prev_action_idx], dtype=np.float32)
        return np.concatenate([prev_action_arr, sinr_vec.astype(np.float32)])


    def prepare_next_actions(
        self,
        pmi_feedback_bits,
        sinr_dB,
        node_wise_bler,
    ) -> Optional[List[List[Optional[np.ndarray]]]]:
        """Update all agents with the newest feedback and return predicted beams per Rx–Tx pair."""

        w1_structures = _extract_w1_from_feedback(pmi_feedback_bits)
        if len(w1_structures) == 0:
            return None

        sinr_array = []
        if sinr_dB is None:
            sinr_array = [None] * len(w1_structures)
        else:
            sinr_array = [np.array(s, dtype=np.float32) for s in sinr_dB]

        node_bler = np.asarray(node_wise_bler) if node_wise_bler is not None else None

        overrides: List[List[Optional[np.ndarray]]] = []

        for rx_idx, raw_w1 in enumerate(w1_structures):
            rx_overrides: List[Optional[np.ndarray]] = []
            tx_entries = raw_w1 if isinstance(raw_w1, (list, tuple)) else [raw_w1]

            for tx_idx, tx_w1 in enumerate(tx_entries):

                if tx_idx == 0:
                    self.N1 = 4
                else:
                    self.N1 = 2
                self.num_beams = (self.O1 * self.N1) * (self.O2 * self.N2)

                w1_vec = _flatten_w1_indices(tx_w1)
                L = len(w1_vec) if len(w1_vec) > 0 else 1
                sinr_vec = sinr_array[rx_idx] if rx_idx < len(sinr_array) else None
                sinr_vec = _ensure_1d_array(sinr_vec)

                # Ensure we have storage for this Rx/Tx pair before accessing any state
                self._ensure_pair_capacity(rx_idx, tx_idx)

                curr_w1_idx = self._register_action(rx_idx, tx_idx, normalized_w1, L)
                state = self._build_state(
                    int(curr_w1_idx) if curr_w1_idx is not None else 0, sinr_vec
                )

                if self.use_enumerated_actions:
                    beam_sets = _enumerate_beam_sets(self.N1, self.O1, self.N2, self.O2, L)
                    self.max_actions = len(beam_sets)

                self._maybe_init_agent(rx_idx, tx_idx, state.shape[0])

                agent = self.agents[rx_idx][tx_idx]
                assert agent is not None

                normalized_w1 = self._canonical_action(tx_w1, L)
                if normalized_w1 is None:
                    normalized_w1 = tuple(range(min(L, self.num_beams)))

                prev_state = self.prev_states[rx_idx][tx_idx]
                prev_action = self.prev_actions[rx_idx][tx_idx]
                if prev_state is not None and prev_action is not None:
                    
                    if self.use_enumerated_actions:
                        match_bonus = int(
                            target_w1_idx is not None
                            and target_w1_idx == self.prev_actions[rx_idx][tx_idx]
                        )
                    else:
                        match_bonus = int(target_w1_idx is not None and target_w1_idx == normalized_w1)

                    if node_bler is not None and node_bler.size > rx_idx:
                        bler_contrib = 1.0 - float(np.mean(node_bler[rx_idx]))
                    else:
                        bler_contrib = 1.0

                    reward = bler_contrib + match_bonus

                    agent.store_transition(prev_state, prev_action, reward, state)
                    agent.activate_target_net(state)

                    episode_len = getattr(agent, "memory_counter", 0)
                    if episode_len >= agent.nForgetPoints:
                        agent.learn_new(episode_len, max(episode_len - 1, 0), method="double")
                
                predicted_idx = agent.choose_action(state)
                predicted_w1 = self._decode_action(rx_idx, tx_idx, predicted_idx)
                if predicted_w1 is None:
                    predicted_w1 = normalized_w1

                rx_overrides.append(_tuple_to_list(predicted_w1) if predicted_w1 is not None else None)

                self.prev_states[rx_idx][tx_idx] = state
                self.prev_actions[rx_idx][tx_idx] = predicted_idx

            overrides.append(rx_overrides)

        return overrides