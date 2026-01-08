import sys
from pathlib import Path
from typing import List, Optional, Tuple
import itertools
import pickle
from dmimo.channel import weiner_filter_pred
from dmimo.channel.twomode_wesn_pred import predict_all_links

import numpy as np

from dmimo.mimo.quantized_CSI_feedback import quantized_CSI_feedback

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
        epsilon_total_steps: Optional[int] = None,
        random_seed: Optional[int] = None,
        imitation_method: Optional[str] = "none",
    ):
        
        self.O2 = 1
        self.N2 = 1
        self.O1 = 4

        # Default values to ensure checkpoint loading works before any actions are prepared
        self.N1 = 1
        self.num_beams = (self.O1 * self.N1) * (self.O2 * self.N2)
        
        self.max_actions = max_actions
        self.memory_size = memory_size
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.epsilon_total_steps = epsilon_total_steps
        
        self.imitation_method = imitation_method

        self.seed_sequence = np.random.SeedSequence(random_seed) if random_seed is not None else None
        self.agent_seeds: List[List[Optional[int]]] = []

        self.agents: List[List[Optional[DeepWESNQNetwork]]] = []
        self.action_maps: List[List[List[Tuple[int, ...]]]] = []
        self.prev_states: List[List[Optional[np.ndarray]]] = []
        self.prev_actions: List[List[Optional[int]]] = []
        self.state_dims: List[List[Optional[int]]] = []
        self.reward_log: List[Tuple[int, int, float]] = []
        self.action_log: List[Tuple[int, int, int, int]] = []
        self.step_counter: int = 0
        self.evaluation_only: bool = False

    def set_epsilon_total_steps(self, total_steps: Optional[int]) -> None:
        """Update the epsilon decay horizon for all agents."""

        self.epsilon_total_steps = total_steps

    def set_evaluation_mode(self, evaluation_only: bool) -> None:
        """Enable or disable training during beam selection."""

        self.evaluation_only = evaluation_only

    def reset_episode(self):
        """Clear per-episode state without discarding learned experience."""

        for rx_idx in range(len(self.prev_states)):
            for tx_idx in range(len(self.prev_states[rx_idx])):
                self.prev_states[rx_idx][tx_idx] = None
                self.prev_actions[rx_idx][tx_idx] = None
        
        self.reward_log.clear()
        self.action_log.clear()
        self.step_counter = 0

    def log_reward(self, rx_idx: int, tx_idx: int, reward: float) -> None:
        """Record a reward emitted by the DEQN agent.

        Args:
            rx_idx: Receiver index associated with the reward.
            tx_idx: Transmitter index associated with the reward.
            reward: Reward value produced for the state/action pair.
        """

        self.reward_log.append((rx_idx, tx_idx, float(reward)))

    def get_reward_log(self) -> List[Tuple[int, int, float]]:
        """Return a copy of the reward log accumulated so far."""

        return list(self.reward_log)
    
    def log_action(self, step: int, rx_idx: int, tx_idx: int, action_idx: Optional[int]) -> None:
        """Record the chosen action index for a given step and agent pair."""

        self.action_log.append((int(step), int(rx_idx), int(tx_idx), int(action_idx) if action_idx is not None else -1))

    def get_action_log(self) -> List[Tuple[int, int, int, int]]:
        """Return a copy of the action log accumulated so far."""

        return list(self.action_log)

    def _ensure_pair_capacity(self, rx_idx: int, tx_idx: int):
        while len(self.agents) <= rx_idx:
            self.agents.append([])
            self.action_maps.append([])
            self.prev_states.append([])
            self.prev_actions.append([])
            self.state_dims.append([])
            self.agent_seeds.append([])

        while len(self.agents[rx_idx]) <= tx_idx:
            self.agents[rx_idx].append(None)
            self.action_maps[rx_idx].append([])
            self.prev_states[rx_idx].append(None)
            self.prev_actions[rx_idx].append(None)
            self.state_dims[rx_idx].append(None)
            self.agent_seeds[rx_idx].append(None)

    def _next_seed(self) -> int:
        if self.seed_sequence is not None:
            return int(self.seed_sequence.spawn(1)[0].entropy % 10000)

        return int(np.random.SeedSequence().entropy % 10000)

    def _maybe_init_agent(self, rx_idx: int, tx_idx: int, state_dim: int):
        self._ensure_pair_capacity(rx_idx, tx_idx)

        if self.agents[rx_idx][tx_idx] is None:

            if self.agent_seeds[rx_idx][tx_idx] is None:
                self.agent_seeds[rx_idx][tx_idx] = self._next_seed()
            
            self.agents[rx_idx][tx_idx] = DeepWESNQNetwork(
                self.max_actions,
                state_dim,
                self.input_window_size,
                self.output_window_size,
                self.memory_size,
                n_layers=1,
                nInternalUnits=64,
                spectral_radius=0.3,
                random_seed=self.agent_seeds[rx_idx][tx_idx],
            )
            self.state_dims[rx_idx][tx_idx] = state_dim
    
    def _init_action_map(self, rx_idx: int, tx_idx: int, L: int) -> List[Tuple[int, ...]]:

        self._ensure_pair_capacity(rx_idx, tx_idx)
        
        existing = self.action_maps[rx_idx][tx_idx]

        if not existing or len(existing[0]) != L:
            self.action_maps[rx_idx][tx_idx] = _enumerate_beam_sets(
                self.N1, self.O1, self.N2, self.O2, L
            )

        return self.action_maps[rx_idx][tx_idx]

    def _decode_action(self, rx_idx: int, tx_idx: int, action_idx: int) -> Optional[Tuple[int, ...]]:
        if action_idx is None:
            return None
        if rx_idx >= len(self.action_maps) or tx_idx >= len(self.action_maps[rx_idx]):
            return None
        if 0 <= action_idx < len(self.action_maps[rx_idx][tx_idx]):
            return self.action_maps[rx_idx][tx_idx][action_idx]

        return None

    def _build_state(self, user_beam_sets: List[List[int]], mcs_values: np.ndarray) -> np.ndarray:
        state_parts: List[np.ndarray] = []
        for user_idx, beam_set in enumerate(user_beam_sets):
            beam_arr = np.asarray(beam_set, dtype=np.float32)
            mcs_val = (
                np.array([mcs_values[user_idx]], dtype=np.float32)
                if user_idx < len(mcs_values)
                else np.array([0.0], dtype=np.float32)
            )
            state_parts.append(np.concatenate([beam_arr, mcs_val]))
        if not state_parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(state_parts)

    def _decode_action_vector(self, action_idx: int, user_count: int) -> List[int]:
        digits: List[int] = []
        remaining = int(action_idx)
        for _ in range(user_count):
            digits.append(remaining % 4)
            remaining //= 4
        return digits

    def _build_candidate_indices(self, current_idx: int, action_map: List[Tuple[int, ...]]) -> List[int]:
        candidates = [current_idx]
        for idx in range(len(action_map)):
            if idx == current_idx:
                continue
            candidates.append(idx)
            if len(candidates) == 4:
                break
        while len(candidates) < 4:
            candidates.append(current_idx)
        return candidates

    def prepare_next_actions(
        self,
        pmi_feedback_bits,
        mcs_indices: Optional[np.ndarray] = None,
        node_wise_acks: Optional[np.ndarray] = None,
        user_count: Optional[int] = None,
    ) -> Optional[List[List[Optional[np.ndarray]]]]:
        """Update all agents with the newest feedback and return predicted beams per Rx–Tx pair."""

        # Pull out raw W1 beam indices (per user, per TX) from the PMI feedback.
        w1_structures = _extract_w1_from_feedback(pmi_feedback_bits)
        if len(w1_structures) == 0:
            return None
        
        self.step_counter += 1

        # Decide how many users to target while still keeping the input window bounded.
        total_users = len(w1_structures)
        selected_user_count = user_count if user_count is not None else 2
        selected_user_count = max(1, min(int(selected_user_count), total_users))

        mcs_array = np.asarray(mcs_indices, dtype=np.float32).flatten() if mcs_indices is not None else None
        ack_array = np.asarray(node_wise_acks, dtype=np.float32).flatten() if node_wise_acks is not None else None

        overrides: List[List[Optional[np.ndarray]]] = []

        # Build action maps per TX so we can map each W1 beam set to its index.
        num_tx = max(
            (len(raw_w1) if isinstance(raw_w1, (list, tuple)) else 1) for raw_w1 in w1_structures
        )
        tx_action_maps: List[List[Tuple[int, ...]]] = []
        tx_action_lookup: List[dict] = []
        tx_beam_lengths: List[int] = []

        for tx_idx in range(num_tx):
            self.N1 = 4 if tx_idx == 0 else 2
            self.num_beams = (self.O1 * self.N1) * (self.O2 * self.N2)
            L = self.N1
            for raw_w1 in w1_structures:
                tx_entries = raw_w1 if isinstance(raw_w1, (list, tuple)) else [raw_w1]
                if tx_idx < len(tx_entries):
                    beams = _flatten_w1_indices(tx_entries[tx_idx])
                    if beams.size > 0:
                        L = len(beams)
                        break
            action_map = _enumerate_beam_sets(self.N1, self.O1, self.N2, self.O2, L)
            tx_action_maps.append(action_map)
            tx_action_lookup.append({beam: idx for idx, beam in enumerate(action_map)})
            tx_beam_lengths.append(L)

        # Convert each user's raw W1 beam set into an index within the action map.
        user_beam_sets: List[List[int]] = []
        for raw_w1 in w1_structures:
            tx_entries = raw_w1 if isinstance(raw_w1, (list, tuple)) else [raw_w1]

            beam_indices: List[int] = []
            for tx_idx in range(num_tx):
                beams = (
                    _flatten_w1_indices(tx_entries[tx_idx]).astype(int)
                    if tx_idx < len(tx_entries)
                    else np.array([], dtype=int)
                )

                L = tx_beam_lengths[tx_idx]
                action_lookup = tx_action_lookup[tx_idx]
                if beams.size >= L and L > 0:
                    beam_tuple = tuple(sorted(beams[:L]))
                    beam_indices.append(action_lookup.get(beam_tuple, 0))
                else:
                    beam_indices.append(0)
            user_beam_sets.append(beam_indices)

        # Measure collisions (same W1 set index used by multiple users) per TX.
        tx_collision_counts = []
        for tx_idx in range(num_tx):
            tx_indices = [beam_sets[tx_idx] for beam_sets in user_beam_sets]
            collisions = len(tx_indices) - len(set(tx_indices))
            tx_collision_counts.append(collisions)

        worst_tx_idx = int(np.argmax(tx_collision_counts)) if tx_collision_counts else 0

        # Score users with ACK * MCS, and pick the worst-performing subset.
        user_scores = ack_array[:total_users] * mcs_array[:total_users]
        worst_user_indices = list(np.argsort(user_scores)[:selected_user_count])

        # State uses indices of W1 beam sets (not raw beam IDs) plus each user's MCS.
        selected_beam_sets = [user_beam_sets[idx] for idx in worst_user_indices]
        selected_mcs = mcs_array[worst_user_indices] if len(mcs_array) > 0 else np.zeros(selected_user_count)
        state = self._build_state(selected_beam_sets, selected_mcs)

        self.max_actions = 4 ** selected_user_count
        self._ensure_pair_capacity(0, worst_tx_idx)
        self._maybe_init_agent(0, worst_tx_idx, state.shape[0])

        agent = self.agents[0][worst_tx_idx]
        assert agent is not None

        prev_state = self.prev_states[0][worst_tx_idx]
        prev_action = self.prev_actions[0][worst_tx_idx]
        episode_len = getattr(agent, "memory_counter", 0)
        if not self.evaluation_only and prev_state is not None and prev_action is not None:
            reward = float(np.sum(user_scores))
            agent.store_transition(prev_state, prev_action, reward, state)
            self.log_reward(0, worst_tx_idx, reward)
            agent.activate_target_net(state)

            episode_len = getattr(agent, "memory_counter", 0)
            min_samples = getattr(
                agent,
                "training_start_threshold",
                getattr(agent, "training_batch_size", getattr(agent, "nForgetPoints", 1)),
            )
            can_train = episode_len >= int(min_samples)
            if can_train:
                agent.learn_new(episode_len, max(episode_len - 1, 0), method="double")


        if not self.evaluation_only:
            epsilon_total_steps = self.epsilon_total_steps if self.epsilon_total_steps is not None else 400
            agent.update_epsilon(episode_len, epsilon_total_steps)

        predicted_idx = None
        predicted_idx = agent.choose_action(state)
        action_vector = self._decode_action_vector(predicted_idx, selected_user_count)

        worst_action_map = tx_action_maps[worst_tx_idx]
        overrides = [[None for _ in range(num_tx)] for _ in range(total_users)]

        for idx, user_idx in enumerate(worst_user_indices):
            current_idx = user_beam_sets[user_idx][worst_tx_idx]
            candidates = self._build_candidate_indices(current_idx, worst_action_map)
            chosen_idx = candidates[action_vector[idx]]
            beam_tuple = worst_action_map[chosen_idx] if worst_action_map else None
            overrides[user_idx][worst_tx_idx] = (
                _tuple_to_list(beam_tuple) if beam_tuple is not None else None
            )

        self.log_action(self.step_counter, 0, worst_tx_idx, predicted_idx)
        self.prev_states[0][worst_tx_idx] = state
        self.prev_actions[0][worst_tx_idx] = predicted_idx

        return overrides

    def pred(self, h_freq_csi_history, rc_config, ns3cfg, num_tx_streams):
        
        # Predicting using the CSI method to immitate
        if self.imitation_method == "two_mode":

            h_freq_csi = predict_all_links(h_freq_csi_history, rc_config, ns3cfg, max_workers=8)
        elif self.imitation_method == "weiner_filter":

            weiner_filter_predictor = weiner_filter_pred(method="using_one_link_MIMO")
            h_freq_csi = np.asarray(weiner_filter_predictor.predict(h_freq_csi_history, K=rc_config.history_len-1))

        # Quantizing the predicted CSI
        type_II_PMI_quantizer = quantized_CSI_feedback(method='5G', 
                                                            codebook_selection_method=None,
                                                            num_tx_streams=num_tx_streams,
                                                            architecture='dMIMO_phase2_type_II_CB2',
                                                            rbs_per_subband=4,
                                                            snrdb=10)
        _, PMI_feedback_bits = type_II_PMI_quantizer(
            h_freq_csi,
            return_feedback_bits=True,
        )

        return PMI_feedback_bits

    def extract_w1_override(self, pmi_feedback_bits):
        """Return the w1_beam_indices structure from PMI feedback bits."""

        if pmi_feedback_bits is None:
            return None

        overrides = []
        pmi_entries = pmi_feedback_bits if isinstance(pmi_feedback_bits, list) else [pmi_feedback_bits]
        for rx_entry in pmi_entries:
            if isinstance(rx_entry, dict):
                overrides.append(rx_entry.get("w1_beam_indices"))
            elif isinstance(rx_entry, (list, tuple)):
                tx_list = []
                for tx_entry in rx_entry:
                    if isinstance(tx_entry, dict):
                        tx_list.append(tx_entry.get("w1_beam_indices"))
                    else:
                        tx_list.append(None)
                overrides.append(tx_list)
            else:
                overrides.append(None)

        return overrides if overrides else None

    def save_all(self, base_path, imitation_info: Optional[str] = None) -> None:
        """Persist all agents and associated metadata to disk.

        Args:
            base_path: Directory where model files will be written.
            imitation_info: Optional description of imitation-learning settings
                used during training.
        """

        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)

        agent_files: List[List[Optional[str]]] = []
        for rx_idx, row in enumerate(self.agents):
            rx_files: List[Optional[str]] = []
            for tx_idx, agent in enumerate(row):
                if agent is None:
                    rx_files.append(None)
                    continue

                agent_path = base / f"agent_{rx_idx}_{tx_idx}.pkl"
                agent.save(agent_path)
                rx_files.append(agent_path.name)
            agent_files.append(rx_files)

        metadata = {
            "action_maps": self.action_maps,
            "state_dims": self.state_dims,
            "prev_states": self.prev_states,
            "prev_actions": self.prev_actions,
            "num_beams": self.num_beams,
            "N1": self.N1,
            "N2": self.N2,
            "O1": self.O1,
            "O2": self.O2,
            "max_actions": self.max_actions,
            "memory_size": self.memory_size,
            "input_window_size": self.input_window_size,
            "output_window_size": self.output_window_size,
            "agent_files": agent_files,
            "agent_seeds": self.agent_seeds,
        }

        if imitation_info:
            metadata["imitation_info"] = imitation_info

        with open(base / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_all(self, base_path) -> None:
        """Restore agents and metadata previously persisted with :meth:`save_all`."""

        base = Path(base_path)
        meta_path = base / "metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No checkpoint metadata found at {meta_path}")

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.action_maps = metadata.get("action_maps", [])
        self.state_dims = metadata.get("state_dims", [])
        self.prev_states = metadata.get("prev_states", [])
        self.prev_actions = metadata.get("prev_actions", [])
        self.num_beams = metadata.get("num_beams", self.num_beams)
        self.N1 = metadata.get("N1", self.N1)
        self.N2 = metadata.get("N2", self.N2)
        self.O1 = metadata.get("O1", self.O1)
        self.O2 = metadata.get("O2", self.O2)
        self.max_actions = metadata.get("max_actions", self.max_actions)
        self.memory_size = metadata.get("memory_size", self.memory_size)
        self.input_window_size = metadata.get("input_window_size", self.input_window_size)
        self.output_window_size = metadata.get("output_window_size", self.output_window_size)
        self.agent_seeds = metadata.get("agent_seeds", [])

        agent_files: List[List[Optional[str]]] = metadata.get("agent_files", [])
        self.agents = []
        for rx_idx, tx_row in enumerate(agent_files):
            agent_row: List[Optional[DeepWESNQNetwork]] = []
            for tx_idx, filename in enumerate(tx_row):
                if filename is None:
                    agent_row.append(None)
                    continue

                agent_path = base / filename
                agent_row.append(DeepWESNQNetwork.load(agent_path))
            self.agents.append(agent_row)

        # Ensure companion arrays have matching dimensions
        for rx_idx, tx_row in enumerate(self.action_maps):
            for tx_idx, _ in enumerate(tx_row):
                self._ensure_pair_capacity(rx_idx, tx_idx)

        # Backfill any missing seeds for restored agents
        for rx_idx, tx_row in enumerate(self.agents):
            if len(self.agent_seeds) <= rx_idx:
                self.agent_seeds.append([])

            while len(self.agent_seeds[rx_idx]) < len(tx_row):
                self.agent_seeds[rx_idx].append(None)

            for tx_idx, agent in enumerate(tx_row):
                if self.agent_seeds[rx_idx][tx_idx] is None and agent is not None:
                    self.agent_seeds[rx_idx][tx_idx] = getattr(agent, "random_seed", None)
