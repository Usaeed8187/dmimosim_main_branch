import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import itertools
import pickle

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
    """Lightweight manager to run a single DEQN agent for PMI beam predictions."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        input_window_size: int = 3,
        output_window_size: int = 3,
        total_steps: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        
        self.O2 = 1
        self.N2 = 1
        self.O1 = 4

        # Default values to ensure checkpoint loading works before any actions are prepared
        self.N1 = 1
        self.num_beams = (self.O1 * self.N1) * (self.O2 * self.N2)
        
        # Test goal: converge quickly on trivial deterministic rewards.
        self.max_actions = 2
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.epsilon_update_period = batch_size
        self.e_greedy_start = 0.7
        self.e_greedy_end = 0.9
        self.epsilon = self.e_greedy_start
        self.e_increase = (self.e_greedy_end - self.e_greedy_start) / max(1, (total_steps // batch_size) - 1)
        
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
        self.last_trained_step: int = 0
        self.transition_idx: int = 0
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

        self.transition_idx = 0

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
                self.batch_size,
                training_batch_size=self.batch_size,
                training_start_threshold=self.batch_size,
                n_layers=1,
                nInternalUnits=16,
                spectral_radius=0.9,
                random_seed=self.agent_seed,
            )
            self.state_dim = state_dim


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

        prev_action_val = 0 if self.prev_action is None else int(self.prev_action)
        state = np.array([float(prev_action_val)], dtype=np.float32)

        action_digit_count = 1
        self.max_actions = 2
        self._maybe_init_agent(state.shape[0])

        agent = self.agent
        assert agent is not None

        prev_state = self.prev_state
        prev_action = self.prev_action

        if not self.evaluation_only and prev_state is not None and prev_action is not None:
            reward = 1.0 if prev_action == 0 else 0.0

            agent.store_transition(prev_state, prev_action, reward, state)
            self.transition_idx += 1
            self.log_reward(reward)
            agent.activate_target_net(state)

            if (self.transition_idx % agent.training_start_threshold) == 0:
            
                agent.learn_new(self.batch_size, self.transition_idx, method="double")

            if (self.transition_idx % self.epsilon_update_period) == 0:
                self.epsilon = min(self.e_greedy_end, self.epsilon + self.e_increase)
                agent.epsilon = self.epsilon
                print("agent.epsilon = ", agent.epsilon)

        action = agent.choose_action(state)

        self.log_action(action)
        self.prev_state = state
        self.prev_action = action

        return None


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
        self.input_window_size = metadata.get("input_window_size", self.input_window_size)
        self.output_window_size = metadata.get("output_window_size", self.output_window_size)
        self.agent_seed = metadata.get("agent_seed", None)

        agent_file: Optional[str] = metadata.get("agent_file")
        if agent_file is not None:
            agent_path = base / agent_file
            self.agent = DeepWESNQNetwork.load(agent_path)
            if self.agent_seed is None:
                self.agent_seed = getattr(self.agent, "random_seed", None)