import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import itertools
import pickle

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
    """Lightweight manager to run a single DEQN agent for PMI beam predictions."""

    def __init__(
        self,
        max_actions: int = 2,
        memory_size: Optional[int] = None,
        input_window_size: int = 3,
        output_window_size: int = 3,
        drops_per_batch: int = 1,
        num_batches_in_replay_buffer: int = 3,
        train_every_n_steps: int = 1,
        epsilon_total_steps: Optional[int] = None,
        random_seed: Optional[int] = None,
        imitation_method: Optional[str] = "none",
        worst_tx_count: int = 1,
    ):
        
        self.O2 = 1
        self.N2 = 1
        self.O1 = 4

        # Default values to ensure checkpoint loading works before any actions are prepared
        self.N1 = 1
        self.num_beams = (self.O1 * self.N1) * (self.O2 * self.N2)
        
        # Test goal: converge quickly on trivial deterministic rewards.
        self.max_actions = 2
        self.drops_per_batch = max(1, int(drops_per_batch))
        self.num_batches_in_replay_buffer = max(1, int(num_batches_in_replay_buffer))
        self.train_every_n_steps = max(1, int(train_every_n_steps))
        self.batch_size_transitions = self.drops_per_batch * self.train_every_n_steps
        computed_memory_size = self.batch_size_transitions * self.num_batches_in_replay_buffer
        self.memory_size = (
            int(computed_memory_size)
            if memory_size is None
            else max(int(memory_size), int(computed_memory_size))
        )
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.epsilon_total_steps = epsilon_total_steps
        
        self.imitation_method = imitation_method
        self.worst_tx_count = max(1, int(worst_tx_count))

        self.seed_sequence = np.random.SeedSequence(random_seed) if random_seed is not None else None
        self.agent_seed: Optional[int] = None

        self.agent: Optional[DeepWESNQNetwork] = None
        self.action_maps: List[List[Tuple[int, ...]]] = []
        self.prev_state: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.state_dim: Optional[int] = None
        self.reward_log: List[float] = []
        self.action_log: List[int] = []
        self.step_counter: int = 0
        self.drop_counter: int = 0
        self.last_trained_drop: int = 0
        self.last_trained_step: int = 0
        self.total_transitions: int = 0
        self.evaluation_only: bool = False
        self.trivial_state: bool = True

    def set_epsilon_total_steps(self, total_steps: Optional[int]) -> None:
        """Update the epsilon decay horizon for the agent."""

        self.epsilon_total_steps = total_steps

    def set_evaluation_mode(self, evaluation_only: bool) -> None:
        """Enable or disable training during beam selection."""

        self.evaluation_only = evaluation_only

    def reset_episode(self):
        """Clear per-episode state without discarding learned experience."""

        self.prev_state = None
        self.prev_action = None
        
        self.reward_log.clear()
        self.action_log.clear()
        self.step_counter = 0
        self.drop_counter += 1

    def log_reward(self, reward: float) -> None:
        """Record a reward emitted by the DEQN agent.

        Args:
            reward: Reward value produced for the state/action pair.
        """

        self.reward_log.append(float(reward))

    def get_reward_log(self) -> List[float]:
        """Return a copy of the reward log accumulated so far."""

        return list(self.reward_log)
    
    def log_action(
        self,
        action_idx: Optional[int],
    ) -> None:
        """Record the chosen action index for a given step and agent pair."""

        self.action_log.append(int(action_idx) if action_idx is not None else -1)

    def get_action_log(self) -> List[int]:
        """Return a copy of the action log accumulated so far."""

        return list(self.action_log)

    def _next_seed(self) -> int:
        if self.seed_sequence is not None:
            return int(self.seed_sequence.spawn(1)[0].entropy % 10000)

        return int(np.random.SeedSequence().entropy % 10000)

    def _maybe_init_agent(self, state_dim: int):
        if self.agent is None:
            if self.agent_seed is None:
                self.agent_seed = self._next_seed()

            self.agent = DeepWESNQNetwork(
                self.max_actions,
                state_dim,
                self.input_window_size,
                self.output_window_size,
                self.memory_size,
                training_batch_size=self.batch_size_transitions,
                training_start_threshold=self.batch_size_transitions,
                n_layers=1,
                nInternalUnits=64,
                spectral_radius=0.3,
                random_seed=self.agent_seed,
            )
            self.state_dim = state_dim


    def _init_action_map(self, tx_idx: int, L: int) -> List[Tuple[int, ...]]:
        while len(self.action_maps) <= tx_idx:
            self.action_maps.append([])

        existing = self.action_maps[tx_idx]

        if not existing or len(existing[0]) != L:
            self.action_maps[tx_idx] = _enumerate_beam_sets(
                self.N1, self.O1, self.N2, self.O2, L
            )

        return self.action_maps[tx_idx]

    def _decode_action(self, tx_idx: int, action_idx: int) -> Optional[Tuple[int, ...]]:
        if action_idx is None:
            return None
        if tx_idx >= len(self.action_maps):
            return None
        if 0 <= action_idx < len(self.action_maps[tx_idx]):
            return self.action_maps[tx_idx][action_idx]

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

    def _decode_action_vector(self, action_idx: int, digit_count: int) -> List[int]:
        digits: List[int] = []
        remaining = int(action_idx)
        for _ in range(digit_count):
            digits.append(remaining % 2)
            remaining //= 2
        return digits

    def _build_candidate_indices(self, current_idx: int, action_map: List[Tuple[int, ...]]) -> List[int]:
        candidates = [current_idx]
        for idx in range(len(action_map)):
            if idx == current_idx:
                continue
            candidates.append(idx)
            if len(candidates) == 2:
                break
        while len(candidates) < 2:
            candidates.append(current_idx)
        return candidates

    def prepare_next_actions(
        self,
        pmi_feedback_bits,
        mcs_indices: Optional[np.ndarray] = None,
        node_wise_acks: Optional[np.ndarray] = None,
        user_count: Optional[int] = None,
        throughput_debug: Optional[float] = None,
        drop_idx_debug: Optional[int] = None,
    ) -> Optional[List[List[Optional[np.ndarray]]]]:
        """Update the agent with the newest feedback and return predicted beams per Rx–Tx pair."""

        # Pull out raw W1 beam indices (per user, per TX) from the PMI feedback.
        w1_structures = _extract_w1_from_feedback(pmi_feedback_bits)
        if len(w1_structures) == 0:
            return None
        
        self.step_counter += 1

        # Decide how many users to target while still keeping the input window bounded.
        total_users = len(w1_structures)
        selected_user_count = 1

        mcs_indices = mcs_indices + 1
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
            action_map = self._init_action_map(tx_idx, L)
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

        if tx_collision_counts:
            worst_tx_count = min(self.worst_tx_count, num_tx)
            worst_tx_indices = list(np.argsort(tx_collision_counts)[-worst_tx_count:])
        else:
            worst_tx_indices = [0]
        worst_tx_indices = worst_tx_indices[:1]

        # Score users with ACK * MCS, and pick the worst-performing subset.
        user_scores = ack_array[:total_users] * mcs_array[:total_users]
        worst_user_indices = list(np.argsort(user_scores)[:selected_user_count])

        # State uses indices of W1 beam sets (not raw beam IDs) plus each user's MCS.
        selected_beam_sets = [user_beam_sets[idx] for idx in worst_user_indices]
        selected_mcs = mcs_array[worst_user_indices] if len(mcs_array) > 0 else np.zeros(selected_user_count)
        state = self._build_state(selected_beam_sets, selected_mcs)
        if self.trivial_state:
            prev_action_val = 0 if self.prev_action is None else int(self.prev_action)
            state = np.array([float(prev_action_val)], dtype=np.float32)

        action_digit_count = 1
        self.max_actions = 2
        self._maybe_init_agent(state.shape[0])

        agent = self.agent
        assert agent is not None

        prev_state = self.prev_state
        prev_action = self.prev_action
        episode_len = getattr(agent, "memory_counter", 0)
        if not self.evaluation_only and prev_state is not None and prev_action is not None:
            if self.trivial_state:
                reward = 1.0 if prev_action == 0 else 0.0
            else:
                reward = 1.0 if prev_action == 0 else -1.0

            agent.store_transition(prev_state, prev_action, reward, state)
            self.total_transitions += 1
            self.log_reward(reward)
            agent.activate_target_net(state)

            episode_len = min(self.total_transitions, self.memory_size)
            min_samples = getattr(
                agent,
                "training_start_threshold",
                getattr(agent, "training_batch_size", getattr(agent, "nForgetPoints", 1)),
            )
            can_train = episode_len >= int(min_samples)
            should_train = (
                self.train_every_n_steps > 0
                and (self.step_counter % self.train_every_n_steps == 0)
                and self.last_trained_step != self.step_counter
            )
            if can_train and should_train:
                agent.learn_new(episode_len, max(episode_len - 1, 0), method="double")
                self.last_trained_step = self.step_counter

        if not self.evaluation_only:
            epsilon_total_steps = self.epsilon_total_steps if self.epsilon_total_steps is not None else 400
            agent.update_epsilon(episode_len, epsilon_total_steps)

        predicted_idx = agent.choose_action(state)
        action_vector = self._decode_action_vector(predicted_idx, action_digit_count)

        overrides = [[None for _ in range(num_tx)] for _ in range(total_users)]

        for tx_pos, tx_idx in enumerate(worst_tx_indices):
            worst_action_map = tx_action_maps[tx_idx]
            for user_pos, user_idx in enumerate(worst_user_indices):
                current_idx = user_beam_sets[user_idx][tx_idx]
                candidates = self._build_candidate_indices(current_idx, worst_action_map)
                action_idx = action_vector[tx_pos * selected_user_count + user_pos]
                chosen_idx = candidates[action_idx]
                beam_tuple = worst_action_map[chosen_idx] if worst_action_map else None
                overrides[user_idx][tx_idx] = (
                    _tuple_to_list(beam_tuple) if beam_tuple is not None else None
                )

        self.log_action(predicted_idx)
        self.prev_state = state
        self.prev_action = predicted_idx

        return overrides

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
        """Persist the agent and associated metadata to disk.

        Args:
            base_path: Directory where model files will be written.
            imitation_info: Optional description of imitation-learning settings
                used during training.
        """

        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)

        agent_file: Optional[str] = None
        if self.agent is not None:
            agent_path = base / "agent.pkl"
            self.agent.save(agent_path)
            agent_file = agent_path.name


        metadata = {
            "action_maps": self.action_maps,
            "state_dim": self.state_dim,
            "prev_state": self.prev_state,
            "prev_action": self.prev_action,
            "num_beams": self.num_beams,
            "N1": self.N1,
            "N2": self.N2,
            "O1": self.O1,
            "O2": self.O2,
            "max_actions": self.max_actions,
            "memory_size": self.memory_size,
            "input_window_size": self.input_window_size,
            "output_window_size": self.output_window_size,
            "agent_file": agent_file,
            "agent_seed": self.agent_seed,
        }

        if imitation_info:
            metadata["imitation_info"] = imitation_info

        with open(base / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_all(self, base_path) -> None:
        """Restore the agent and metadata previously persisted with :meth:`save_all`."""

        base = Path(base_path)
        meta_path = base / "metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No checkpoint metadata found at {meta_path}")

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.action_maps = metadata.get("action_maps", [])
        self.state_dim = metadata.get("state_dim", None)
        self.prev_state = metadata.get("prev_state", None)
        self.prev_action = metadata.get("prev_action", None)
        self.num_beams = metadata.get("num_beams", self.num_beams)
        self.N1 = metadata.get("N1", self.N1)
        self.N2 = metadata.get("N2", self.N2)
        self.O1 = metadata.get("O1", self.O1)
        self.O2 = metadata.get("O2", self.O2)
        self.max_actions = metadata.get("max_actions", self.max_actions)
        self.memory_size = metadata.get("memory_size", self.memory_size)
        self.input_window_size = metadata.get("input_window_size", self.input_window_size)
        self.output_window_size = metadata.get("output_window_size", self.output_window_size)
        self.agent_seed = metadata.get("agent_seed", None)

        agent_file: Optional[str] = metadata.get("agent_file")
        if agent_file is not None:
            agent_path = base / agent_file
            self.agent = DeepWESNQNetwork.load(agent_path)
            if self.agent_seed is None:
                self.agent_seed = getattr(self.agent, "random_seed", None)