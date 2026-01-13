import sys
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


def _ensure_1d_array(arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32).flatten()


def _normalize_mask(mask: Optional[np.ndarray], target_len: int) -> np.ndarray:
    arr = _ensure_1d_array(mask)
    if arr.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if arr.size < target_len:
        padding = np.zeros(target_len - arr.size, dtype=np.float32)
        return np.concatenate([arr, padding])
    return arr[:target_len]


def _enumerate_tx_combinations(num_txues: int, num_selected: int) -> List[Tuple[int, ...]]:
    if num_txues <= 0 or num_selected <= 0:
        return []
    num_selected = min(num_selected, num_txues)
    return list(itertools.combinations(range(num_txues), num_selected))


def _safe_mean(values: Optional[np.ndarray]) -> float:
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=np.float32).flatten()
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


class RLTxUESelector:
    """Lightweight manager to run a single DEQN agent for Tx UE selection."""

    def __init__(
        self,
        max_actions: int = 128,
        memory_size: Optional[int] = None,
        input_window_size: int = 3,
        output_window_size: int = 3,
        drops_per_batch: int = 1,
        num_batches_in_replay_buffer: int = 3,
        steps_per_drop: int = 15,
        epsilon_total_steps: Optional[int] = None,
        random_seed: Optional[int] = None,
        imitation_method: Optional[str] = "none",
    ):
        self.max_actions = max_actions
        self.drops_per_batch = max(1, int(drops_per_batch))
        self.num_batches_in_replay_buffer = max(1, int(num_batches_in_replay_buffer))
        self.steps_per_drop = max(1, int(steps_per_drop))
        self.batch_size_transitions = self.drops_per_batch * self.steps_per_drop
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

        self.seed_sequence = np.random.SeedSequence(random_seed) if random_seed is not None else None
        self.agent_seed: Optional[int] = None

        self.agent: Optional[DeepWESNQNetwork] = None
        self.action_map: List[Tuple[int, ...]] = []
        self.prev_state: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.state_dim: Optional[int] = None
        self.reward_log: List[float] = []
        self.action_log: List[int] = []
        self.step_counter: int = 0
        self.drop_counter: int = 0
        self.last_trained_drop: int = 0
        self.total_transitions: int = 0
        self.evaluation_only: bool = False

    def set_epsilon_total_steps(self, total_steps: Optional[int]) -> None:
        """Update the epsilon decay horizon for the agent."""

        self.epsilon_total_steps = total_steps

    def set_evaluation_mode(self, evaluation_only: bool) -> None:
        """Enable or disable training during Tx UE selection."""

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
        self.reward_log.append(float(reward))

    def get_reward_log(self) -> List[float]:
        return list(self.reward_log)

    def log_action(self, action_idx: Optional[int]) -> None:
        self.action_log.append(int(action_idx) if action_idx is not None else -1)

    def get_action_log(self) -> List[int]:
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

    def _init_action_map(self, num_txues: int, num_selected: int) -> List[Tuple[int, ...]]:
        candidate_map = _enumerate_tx_combinations(num_txues, num_selected)
        if candidate_map:
            self.action_map = candidate_map
        return self.action_map

    def _build_state(
        self,
        last_tx_mask: Optional[np.ndarray],
        num_txues: int,
        mcs_indices: Optional[np.ndarray],
        node_wise_acks: Optional[np.ndarray],
        throughput_debug: Optional[float],
    ) -> np.ndarray:
        mask = _normalize_mask(last_tx_mask, num_txues)
        mean_mcs = _safe_mean(mcs_indices)
        mean_ack = _safe_mean(node_wise_acks)
        throughput = float(throughput_debug) if throughput_debug is not None else 0.0
        return np.concatenate([mask, np.array([mean_mcs, mean_ack, throughput], dtype=np.float32)])

    def prepare_tx_ue_selection(
        self,
        num_txues: int,
        num_txue_sel: int,
        last_tx_mask: Optional[np.ndarray] = None,
        mcs_indices: Optional[np.ndarray] = None,
        node_wise_acks: Optional[np.ndarray] = None,
        throughput_debug: Optional[float] = None,
        drop_idx_debug: Optional[int] = None,
    ) -> Optional[List[int]]:
        if num_txues <= 0:
            return None

        self.step_counter += 1

        num_txue_sel = min(num_txue_sel, num_txues)
        action_map = self._init_action_map(num_txues, num_txue_sel)
        if not action_map:
            return None

        state = self._build_state(last_tx_mask, num_txues, mcs_indices, node_wise_acks, throughput_debug)

        self.max_actions = max(len(action_map), 1)
        self._maybe_init_agent(state.shape[0])
        agent = self.agent
        assert agent is not None

        prev_state = self.prev_state
        prev_action = self.prev_action
        episode_len = getattr(agent, "memory_counter", 0)
        if not self.evaluation_only and prev_state is not None and prev_action is not None:
            reward_value = throughput_debug
            if reward_value is None:
                reward_value = _safe_mean(node_wise_acks)
            reward = float(reward_value) if reward_value is not None else 0.0

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
                self.drop_counter > 0
                and self.drops_per_batch > 0
                and (self.drop_counter % self.drops_per_batch == 0)
                and self.last_trained_drop != self.drop_counter
            )
            if can_train and should_train:
                agent.learn_new(episode_len, max(episode_len - 1, 0), method="double")
                self.last_trained_drop = self.drop_counter

        if not self.evaluation_only:
            epsilon_total_steps = self.epsilon_total_steps if self.epsilon_total_steps is not None else 400
            agent.update_epsilon(episode_len, epsilon_total_steps)

        predicted_idx = agent.choose_action(state)
        selected = list(action_map[predicted_idx]) if predicted_idx is not None else None

        self.log_action(predicted_idx)
        self.prev_state = state
        self.prev_action = predicted_idx

        return selected

    def save_all(self, base_path, imitation_info: Optional[str] = None) -> None:
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)

        agent_file: Optional[str] = None
        if self.agent is not None:
            agent_path = base / "agent.pkl"
            self.agent.save(agent_path)
            agent_file = agent_path.name

        metadata = {
            "action_map": self.action_map,
            "state_dim": self.state_dim,
            "prev_state": self.prev_state,
            "prev_action": self.prev_action,
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
        base = Path(base_path)
        meta_path = base / "metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No checkpoint metadata found at {meta_path}")

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.action_map = metadata.get("action_map", [])
        self.state_dim = metadata.get("state_dim", None)
        self.prev_state = metadata.get("prev_state", None)
        self.prev_action = metadata.get("prev_action", None)
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