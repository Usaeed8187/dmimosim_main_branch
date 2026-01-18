import sys
from pathlib import Path
from typing import List, Optional, Tuple
import pickle

import numpy as np
import tensorflow as tf

# Make the ICML_DEQN_clean folder importable
REPO_ROOT = Path(__file__).resolve().parents[2]
DEQN_PATH = REPO_ROOT / "ICML_DEQN_clean"
if str(DEQN_PATH) not in sys.path:
    sys.path.append(str(DEQN_PATH))

from ICML_DEQN_clean.DQN_RC_new_WESN import DeepWESNQNetwork  # noqa: E402


def _scale_csi_by_snr(h_freq_csi, snr_dB_arr):
    h_scaled = tf.convert_to_tensor(h_freq_csi)
    if h_scaled.shape.rank is not None and h_scaled.shape.rank > 4:
        squeeze_axes = [idx for idx, dim in enumerate(h_scaled.shape) if dim == 1]
        if squeeze_axes:
            h_scaled = tf.squeeze(h_scaled, axis=squeeze_axes)
    snr_lin = tf.cast(10.0 ** (tf.convert_to_tensor(snr_dB_arr) / 10.0), h_scaled.dtype)
    snr_lin = tf.reshape(snr_lin, [-1, 1, 1, 1])
    return h_scaled * snr_lin


def _proxy_mutual_information(h_scaled, tx_ant_indices) -> float:
    h_sel = tf.gather(h_scaled, tx_ant_indices, axis=1)
    h_sel = tf.transpose(h_sel, [2, 3, 0, 1])
    gram = tf.matmul(h_sel, h_sel, adjoint_b=True)
    eye = tf.eye(gram.shape[-1], batch_shape=gram.shape[:-2], dtype=gram.dtype)
    mi = tf.linalg.logdet(eye + gram)
    mi = tf.math.real(mi) / tf.math.log(tf.cast(2.0, mi.dtype))
    return float(tf.reduce_mean(mi).numpy())


def _proxy_mi_for_mask(h_scaled, gnb_indices, tx_ue_indices, mask) -> float:
    tx_ant_indices = gnb_indices[:]
    for ue_idx, active in enumerate(mask):
        if active > 0:
            tx_ant_indices.extend(tx_ue_indices[ue_idx])
    return _proxy_mutual_information(h_scaled, tx_ant_indices)


class RLTxSelector:
    """DEQN-based transmitter selector using mutual-information proxy deltas."""

    def __init__(
        self,
        batch_size: Optional[int] = 8,
        input_window_size: int = 3,
        output_window_size: int = 3,
        total_steps: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.epsilon_update_period = batch_size
        self.e_greedy_start = 0.7
        self.e_greedy_end = 0.9
        self.epsilon = self.e_greedy_start
        if total_steps is None or batch_size is None:
            self.e_increase = 0.0
        else:
            self.e_increase = (self.e_greedy_end - self.e_greedy_start) / max(
                1, (total_steps // batch_size) - 1
            )

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size

        self.seed_sequence = np.random.SeedSequence(random_seed) if random_seed is not None else None
        self.agent_seed: Optional[int] = None

        self.agent: Optional[DeepWESNQNetwork] = None
        self.action_maps: dict[Tuple[int, int], List[Optional[Tuple[int, int]]]] = {}
        self.prev_state: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.state_dim: Optional[int] = None
        self.reward_log: List[float] = []
        self.action_log: List[int] = []
        self.transition_idx: int = 0
        self.evaluation_only: bool = False
        self.max_actions: Optional[int] = None
        self.current_mask: Optional[np.ndarray] = None

    def set_evaluation_mode(self, evaluation_only: bool) -> None:
        """Enable or disable training during transmitter selection."""

        self.evaluation_only = evaluation_only

    def reset_episode(self) -> None:
        """Reset episode-specific state."""

        self.prev_state = None
        self.prev_action = None
        self.current_mask = None
        self.reward_log.clear()
        self.action_log.clear()
        self.transition_idx = 0

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

    def _maybe_init_agent(self, state_dim: int, max_actions: int) -> None:
        if self.agent is None or self.max_actions != max_actions:
            self.max_actions = max_actions
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
            self.agent.epsilon = self.epsilon
            self.state_dim = state_dim

    def _build_action_map(
        self, num_txue: int, current_mask: np.ndarray
    ) -> List[Optional[Tuple[int, int]]]:
        mask_key = tuple(int(v) for v in current_mask.tolist())
        key = (num_txue, mask_key)
        if key in self.action_maps:
            return self.action_maps[key]

        actions: List[Optional[Tuple[int, int]]] = [None]
        selected_indices = [idx for idx, active in enumerate(current_mask) if active > 0]
        unselected_indices = [idx for idx, active in enumerate(current_mask) if active <= 0]
        for drop_idx in selected_indices:
            for add_idx in unselected_indices:
                actions.append((drop_idx, add_idx))
        self.action_maps[key] = actions
        return actions

    def _compute_reward(
        self,
        mcs_indices: Optional[np.ndarray],
        node_wise_acks: Optional[np.ndarray],
    ) -> Optional[float]:
        if mcs_indices is None or node_wise_acks is None:
            return None
        mcs_array = np.asarray(mcs_indices, dtype=np.float32).flatten()
        ack_array = np.asarray(node_wise_acks, dtype=np.float32).flatten()
        if mcs_array.size == 0 or ack_array.size == 0:
            return None
        return float(np.sum(mcs_array * ack_array))

    def _build_state(
        self,
        h_scaled,
        gnb_indices,
        tx_ue_indices,
        current_mask,
    ) -> np.ndarray:
        base_mi = _proxy_mi_for_mask(h_scaled, gnb_indices, tx_ue_indices, current_mask)
        deltas: List[float] = []
        for ue_idx, active in enumerate(current_mask):
            trial_mask = np.array(current_mask, dtype=np.float32)
            if active > 0:
                trial_mask[ue_idx] = 0
            else:
                trial_mask[ue_idx] = 1
            new_mi = _proxy_mi_for_mask(h_scaled, gnb_indices, tx_ue_indices, trial_mask)
            deltas.append(new_mi - base_mi)
        return np.asarray(deltas, dtype=np.float32)

    def select_tx_ue_mask(
        self,
        h_freq_csi,
        snr_dB_arr,
        num_txue: int,
        num_txue_sel: int,
        gnb_tx_ant: int,
        tx_ue_ant: int,
        mcs_indices: Optional[np.ndarray] = None,
        node_wise_acks: Optional[np.ndarray] = None,
        base_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if num_txue_sel <= 0 or num_txue <= 0:
            return None
        if h_freq_csi is None or snr_dB_arr is None:
            return base_mask

        if self.current_mask is None:
            if base_mask is not None:
                self.current_mask = np.array(base_mask, dtype=np.float32)
            else:
                init_mask = np.zeros(num_txue, dtype=np.float32)
                init_mask[: min(num_txue_sel, num_txue)] = 1
                self.current_mask = init_mask

        current_mask = np.array(self.current_mask, dtype=np.float32)
        if current_mask.size != num_txue:
            current_mask = np.zeros(num_txue, dtype=np.float32)
            current_mask[: min(num_txue_sel, num_txue)] = 1

        h_scaled = _scale_csi_by_snr(h_freq_csi, snr_dB_arr)
        gnb_indices = list(range(gnb_tx_ant))
        tx_ue_indices = [
            list(range(gnb_tx_ant + ue_idx * tx_ue_ant, gnb_tx_ant + (ue_idx + 1) * tx_ue_ant))
            for ue_idx in range(num_txue)
        ]

        state = self._build_state(h_scaled, gnb_indices, tx_ue_indices, current_mask)
        print("state = ", state)
        actions = self._build_action_map(num_txue, current_mask)
        self._maybe_init_agent(state.shape[0], len(actions))

        agent = self.agent
        assert agent is not None

        predicted_idx = 0
        if not self.evaluation_only and self.prev_state is not None and self.prev_action is not None:
            reward = self._compute_reward(mcs_indices, node_wise_acks)
            if reward is not None:
                print("reward = ", reward)
                agent.activate_target_net(state)
                agent.store_transition(self.prev_state, self.prev_action, reward, state)
                self.log_reward(reward)

                if (
                    (self.transition_idx + 1) >= agent.training_start_threshold
                    and ((self.transition_idx + 1) % self.batch_size) == 0
                ):
                    agent.learn_new(self.batch_size, self.transition_idx, method="double")

                if ((self.transition_idx + 1) % self.epsilon_update_period) == 0:
                    self.epsilon = min(self.e_greedy_end, self.epsilon + self.e_increase)
                    self.agent.epsilon = self.epsilon

                predicted_idx = agent.choose_action(state)
                print("predicted_idx: ",predicted_idx)
                self.transition_idx += 1

        action = actions[predicted_idx] if 0 <= predicted_idx < len(actions) else None
        next_mask = np.array(current_mask, dtype=np.float32)
        if action is not None:
            drop_idx, add_idx = action
            if next_mask[drop_idx] > 0 and next_mask[add_idx] == 0:
                next_mask[drop_idx] = 0
                next_mask[add_idx] = 1

        self.log_action(predicted_idx)
        self.prev_state = state
        self.prev_action = predicted_idx
        self.current_mask = next_mask
        return next_mask

    def save_all(self, base_path) -> None:
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
            "max_actions": self.max_actions,
            "input_window_size": self.input_window_size,
            "output_window_size": self.output_window_size,
            "agent_file": agent_file,
            "agent_seed": self.agent_seed,
            "current_mask": self.current_mask,
        }

        with open(base / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_all(self, base_path) -> None:
        base = Path(base_path)
        meta_path = base / "metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No checkpoint metadata found at {meta_path}")

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.action_maps = metadata.get("action_maps", {})
        self.state_dim = metadata.get("state_dim", None)
        self.prev_state = metadata.get("prev_state", None)
        self.prev_action = metadata.get("prev_action", None)
        self.max_actions = metadata.get("max_actions", None)
        self.input_window_size = metadata.get("input_window_size", self.input_window_size)
        self.output_window_size = metadata.get("output_window_size", self.output_window_size)
        self.agent_seed = metadata.get("agent_seed", None)
        self.current_mask = metadata.get("current_mask", None)

        agent_file: Optional[str] = metadata.get("agent_file")
        if agent_file is not None:
            agent_path = base / agent_file
            self.agent = DeepWESNQNetwork.load(agent_path)
            if self.agent_seed is None:
                self.agent_seed = getattr(self.agent, "random_seed", None)