import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Make the ICML_DEQN_clean folder importable
REPO_ROOT = Path(__file__).resolve().parents[2]
DEQN_PATH = REPO_ROOT / "ICML_DEQN_clean"
if str(DEQN_PATH) not in sys.path:
    sys.path.append(str(DEQN_PATH))

from ICML_DEQN_clean.DQN_RC_new import DeepQNetwork  # noqa: E402


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


def _ensure_1d_array(arr: Optional[np.ndarray], target_len: int) -> np.ndarray:
    if arr is None:
        return np.zeros(target_len, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).flatten()
    if arr.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if arr.size < target_len:
        padding = np.zeros(target_len - arr.size, dtype=np.float32)
        arr = np.concatenate([arr, padding])
    elif arr.size > target_len:
        arr = arr[:target_len]
    return arr


class RLBeamSelector:
    """Lightweight manager to run DEQN agents for PMI beam predictions."""

    def __init__(self, max_actions: int = 128, memory_size: int = 200):
        self.max_actions = max_actions
        self.memory_size = memory_size

        self.agents: List[Optional[DeepQNetwork]] = []
        self.action_maps: List[List[np.ndarray]] = []
        self.prev_states: List[Optional[np.ndarray]] = []
        self.prev_actions: List[Optional[int]] = []
        self.state_dims: List[Optional[int]] = []

    def _maybe_init_agent(self, rx_idx: int, state_dim: int):
        while len(self.agents) <= rx_idx:
            self.agents.append(None)
            self.action_maps.append([])
            self.prev_states.append(None)
            self.prev_actions.append(None)
            self.state_dims.append(None)

        if self.agents[rx_idx] is None:
            self.agents[rx_idx] = DeepQNetwork(
                self.max_actions,
                state_dim,
                self.memory_size,
                n_layers=1,
                nInternalUnits=64,
                spectral_radius=0.3,
            )
            self.state_dims[rx_idx] = state_dim

    def _register_action(self, rx_idx: int, beam_struct) -> int:
        beam_tuple = _structure_to_tuple(beam_struct)
        existing = self.action_maps[rx_idx]
        for idx, saved in enumerate(existing):
            if saved == beam_tuple:
                return idx

        if len(existing) < self.max_actions:
            self.action_maps[rx_idx].append(beam_tuple)
            return len(existing)

        return 0

    def _decode_action(self, rx_idx: int, action_idx: int) -> Optional[np.ndarray]:
        if action_idx is None:
            return None
        if rx_idx >= len(self.action_maps):
            return None
        if 0 <= action_idx < len(self.action_maps[rx_idx]):
            return self.action_maps[rx_idx][action_idx]
        return None

    def _build_state(self, w1_vec: np.ndarray, sinr_vec: np.ndarray) -> np.ndarray:
        return np.concatenate([w1_vec.astype(np.float32), sinr_vec.astype(np.float32)])

    def prepare_next_actions(
        self,
        pmi_feedback_bits,
        sinr_dB,
        node_wise_bler,
    ) -> Optional[List[Optional[np.ndarray]]]:
        """Update all agents with the newest feedback and return predicted beams."""

        w1_structures = _extract_w1_from_feedback(pmi_feedback_bits)
        if len(w1_structures) == 0:
            return None

        sinr_array = []
        if sinr_dB is None:
            sinr_array = [None] * len(w1_structures)
        else:
            sinr_array = [np.array(s, dtype=np.float32) for s in sinr_dB]

        node_bler = np.asarray(node_wise_bler) if node_wise_bler is not None else None

        overrides: List[Optional[np.ndarray]] = []

        for rx_idx, raw_w1 in enumerate(w1_structures):
            w1_vec = _flatten_w1_indices(raw_w1)
            sinr_vec = sinr_array[rx_idx] if rx_idx < len(sinr_array) else None
            sinr_vec = _ensure_1d_array(sinr_vec, max(len(w1_vec), 1))

            state = self._build_state(_ensure_1d_array(w1_vec, len(w1_vec) if len(w1_vec) > 0 else 1), sinr_vec)
            self._maybe_init_agent(rx_idx, state.shape[0])

            agent = self.agents[rx_idx]
            assert agent is not None

            normalized_w1 = _structure_to_tuple(raw_w1)
            current_action_idx = self._register_action(rx_idx, normalized_w1 if normalized_w1 is not None else (0,))

            if self.prev_states[rx_idx] is not None and self.prev_actions[rx_idx] is not None:
                target_w1 = self._decode_action(rx_idx, self.prev_actions[rx_idx])
                match_bonus = int(target_w1 is not None and _flatten_w1_indices(target_w1).shape == w1_vec.shape and np.array_equal(_flatten_w1_indices(target_w1), w1_vec))

                if node_bler is not None and node_bler.size > rx_idx:
                    bler_val = float(np.mean(node_bler[rx_idx]))
                else:
                    bler_val = 0.0

                reward = bler_val + match_bonus

                agent.store_transition(self.prev_states[rx_idx], self.prev_actions[rx_idx], reward, state)
                agent.activate_target_net(state)

                episode_len = getattr(agent, "memory_counter", 0)
                if episode_len >= agent.nForgetPoints:
                    agent.learn_new(episode_len, max(episode_len - 1, 0), method="double")

            predicted_idx = agent.choose_action(state)
            predicted_w1 = self._decode_action(rx_idx, predicted_idx)
            if predicted_w1 is None:
                predicted_w1 = normalized_w1

            overrides.append(_tuple_to_list(predicted_w1) if predicted_w1 is not None else None)

            self.prev_states[rx_idx] = state
            self.prev_actions[rx_idx] = current_action_idx if current_action_idx is not None else predicted_idx

        return overrides