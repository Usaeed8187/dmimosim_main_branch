from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buffer: List[Transition] = []
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        state = np.stack([b.state for b in batch], axis=0)
        action = np.stack([b.action for b in batch], axis=0)
        reward = np.array([b.reward for b in batch], dtype=np.float32)[:, None]
        next_state = np.stack([b.next_state for b in batch], axis=0)
        done = np.array([b.done for b in batch], dtype=np.float32)[:, None]
        return state, action, reward, next_state, done


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.act_limit * self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: Optional[torch.device] = None,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 200_000,
        batch_size: int = 256,
        start_learning: int = 2_000,
        noise_std: float = 0.1,
        seed: int = 0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.start_learning = start_learning
        self.noise_std = noise_std

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.actor_t = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_t = Critic(obs_dim, act_dim).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay = ReplayBuffer(buffer_size, seed=seed)

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        obs_t = _to_torch(obs[None, :], self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if explore:
            action = action + np.random.normal(0.0, self.noise_std, size=action.shape).astype(np.float32)
        return np.clip(action, -self.actor.act_limit, self.actor.act_limit)

    def _polyak_update(self):
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

    def store_and_maybe_train(self, transition: Transition, evaluation_only: bool = False):
        self.replay.push(
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.done,
        )

        if evaluation_only or len(self.replay) < self.start_learning:
            return

        state, action, reward, next_state, done = self.replay.sample(self.batch_size)
        s_t = _to_torch(state, self.device)
        a_t = _to_torch(action, self.device)
        r_t = _to_torch(reward, self.device)
        s2_t = _to_torch(next_state, self.device)
        d_t = _to_torch(done, self.device)

        with torch.no_grad():
            next_act = self.actor_t(s2_t)
            q_target = self.critic_t(s2_t, next_act)
            y = r_t + self.gamma * (1.0 - d_t) * q_target

        q_val = self.critic(s_t, a_t)
        loss_c = (q_val - y).pow(2).mean()
        self.opt_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()

        loss_a = -self.critic(s_t, self.actor(s_t)).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_a.step()

        self._polyak_update()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_t": self.actor_t.state_dict(),
                "critic_t": self.critic_t.state_dict(),
                "replay": self.replay,
            },
            path,
        )

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload.get("actor", {}))
        self.critic.load_state_dict(payload.get("critic", {}))
        self.actor_t.load_state_dict(payload.get("actor_t", {}))
        self.critic_t.load_state_dict(payload.get("critic_t", {}))
        replay = payload.get("replay")
        if isinstance(replay, ReplayBuffer):
            self.replay = replay


class DDPGChannelPredictor:
    """Per-receiver DDPG agents for continuous channel prediction."""

    def __init__(
        self,
        num_receivers: int,
        num_subbands: int,
        rbs_per_subband: int = 4,
        act_limit: float = 1.0,
        device: Optional[torch.device] = None,
        evaluation_only: bool = False,
        seed: int = 0,
    ):
        self.num_receivers = num_receivers
        self.num_subbands = num_subbands
        self.rbs_per_subband = rbs_per_subband
        self.act_limit = act_limit
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_only = evaluation_only
        self.seed = seed

        self.obs_dim: Optional[int] = None
        self.act_dim: Optional[int] = None
        self.num_rx_antennas: Optional[int] = None
        self.num_tx_nodes: Optional[int] = None
        self.num_tx_antennas: Optional[int] = None

        self.agents: List[DDPGAgent] = []

        self.prev_obs: Dict[int, np.ndarray] = {}
        self.prev_action: Dict[int, np.ndarray] = {}
        self.predicted_subbands: Dict[int, np.ndarray] = {}
        self.reward_log: List[Tuple[int, float]] = []

    def set_evaluation_mode(self, evaluation_only: bool) -> None:
        self.evaluation_only = evaluation_only

    def _ensure_agents_initialized(self, channel: tf.Tensor) -> None:
        # channel shape: [batch, rx_node, rx_ant, tx_node, tx_ant, ofdm_sym, fft]
        _, _, rx_ant, tx_node, tx_ant, _, _ = channel.shape
        if (
            self.num_rx_antennas == rx_ant
            and self.num_tx_nodes == tx_node
            and self.num_tx_antennas == tx_ant
            and self.agents
        ):
            return

        self.num_rx_antennas = int(rx_ant)
        self.num_tx_nodes = int(tx_node)
        self.num_tx_antennas = int(tx_ant)

        feature_dim = self.num_rx_antennas * self.num_tx_nodes * self.num_tx_antennas * self.num_subbands
        self.obs_dim = 2 * feature_dim + 1
        self.act_dim = 2 * feature_dim

        self.agents = [
            DDPGAgent(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                act_limit=self.act_limit,
                device=self.device,
                seed=self.seed + idx,
            )
            for idx in range(self.num_receivers)
        ]

        self.prev_obs.clear()
        self.prev_action.clear()
        self.predicted_subbands.clear()

    def _complex_to_obs(self, complex_vec: np.ndarray, sinr_db: float) -> np.ndarray:
        real = np.real(complex_vec)
        imag = np.imag(complex_vec)
        feat = np.concatenate([real, imag], axis=0).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-6
        feat = feat / norm
        return np.concatenate([feat, np.array([sinr_db], dtype=np.float32)], axis=0)

    def _extract_subbands(self, channel: tf.Tensor) -> np.ndarray:
        # channel shape: [batch, rx_node, rx_ant, tx_node, tx_ant, ofdm_sym, fft]
        self._ensure_agents_initialized(channel)

        mean_over_time = tf.reduce_mean(channel, axis=[0, 5])  # [rx, rx_ant, tx_node, tx_ant, fft]
        mean_t = tf.transpose(mean_over_time, perm=[4, 0, 1, 2, 3])  # [fft, rx, rx_ant, tx_node, tx_ant]
        sb_ids = tf.range(tf.shape(mean_t)[0], dtype=tf.int32) // int(self.rbs_per_subband)
        sb = tf.math.segment_mean(mean_t, sb_ids)  # [subband, rx, rx_ant, tx_node, tx_ant]
        return tf.transpose(sb, perm=[1, 2, 3, 4, 0]).numpy()


    def predict_channels(self, channel_history: np.ndarray, sinr_db: Optional[np.ndarray] = None) -> tf.Tensor:
        if channel_history is None:
            raise ValueError("channel_history is required for DDPG prediction")
        if sinr_db is None:
            sinr_db = np.zeros(self.num_receivers, dtype=np.float32)

        latest_channel = tf.convert_to_tensor(channel_history[-1])
        subbands = self._extract_subbands(latest_channel)
        predicted_channel = latest_channel.numpy()

        for rx_idx in range(min(self.num_receivers, subbands.shape[0])):
            obs = self._complex_to_obs(subbands[rx_idx].reshape(-1), float(sinr_db[rx_idx]))
            action = self.agents[rx_idx].act(obs, explore=not self.evaluation_only)
            real, imag = np.split(action, 2)
            pred_sb = (real + 1j * imag).reshape(subbands[rx_idx].shape)
            self.predicted_subbands[rx_idx] = pred_sb
            self.prev_obs[rx_idx] = obs
            self.prev_action[rx_idx] = action

            scaling = np.repeat(pred_sb, self.rbs_per_subband, axis=-1)[
                ..., : predicted_channel.shape[-1]
            ].astype(np.complex64)
            scaling = scaling.reshape((1, 1) + scaling.shape[:3] + (1, scaling.shape[-1]))

            predicted_channel[:, rx_idx : rx_idx + 1, ...] = predicted_channel[
                :, rx_idx : rx_idx + 1, ...

            ] * scaling

        return tf.convert_to_tensor(predicted_channel)

    def update_with_feedback(
        self,
        true_channel: np.ndarray,
        node_bler: Optional[np.ndarray],
        sinr_db: Optional[np.ndarray],
    ) -> np.ndarray:
        if true_channel is None:
            return np.array(self.reward_log, dtype=np.float32)

        sinr_db = np.zeros(self.num_receivers, dtype=np.float32) if sinr_db is None else sinr_db
        true_subbands = self._extract_subbands(tf.convert_to_tensor(true_channel))

        for rx_idx in range(min(self.num_receivers, true_subbands.shape[0])):
            if rx_idx not in self.prev_obs or rx_idx not in self.prev_action:
                continue

            bler_val = 1.0
            if node_bler is not None and node_bler.size > rx_idx:
                bler_val = float(np.mean(node_bler[rx_idx]))

            target = true_subbands[rx_idx].reshape(-1)

            pred = self.predicted_subbands.get(rx_idx, np.zeros_like(target))
            pred = pred.reshape(-1)

            err = np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-6)
            match_bonus = 1.0 - np.tanh(err)
            reward = (1.0 - bler_val) + match_bonus

            next_obs = self._complex_to_obs(target, float(sinr_db[rx_idx]))
            transition = Transition(
                state=self.prev_obs[rx_idx],
                action=self.prev_action[rx_idx],
                reward=float(reward),
                next_state=next_obs,
                done=0.0,
            )
            self.agents[rx_idx].store_and_maybe_train(transition, evaluation_only=self.evaluation_only)
            self.reward_log.append((rx_idx, float(reward)))
            self.prev_obs[rx_idx] = next_obs

        return np.array(self.reward_log, dtype=np.float32)

    def get_reward_log(self) -> List[Tuple[int, float]]:
        return list(self.reward_log)

    def reset_episode(self) -> None:
        self.prev_obs.clear()
        self.prev_action.clear()
        self.predicted_subbands.clear()
        self.reward_log.clear()

    def save_all(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for idx, agent in enumerate(self.agents):
            agent.save(checkpoint_dir / f"rx_{idx}.pt")

    def load_all(self, checkpoint_dir: Path) -> None:
        for idx, agent in enumerate(self.agents):
            agent.load(checkpoint_dir / f"rx_{idx}.pt")


def default_ddpg_predictor(
    num_receivers: int,
    fft_size: int,
    rbs_per_subband: int = 4,
    evaluation_only: bool = False,
) -> DDPGChannelPredictor:
    num_subbands = int(np.ceil(float(fft_size) / float(rbs_per_subband)))
    return DDPGChannelPredictor(
        num_receivers=num_receivers,
        num_subbands=num_subbands,
        rbs_per_subband=rbs_per_subband,
        evaluation_only=evaluation_only,
    )