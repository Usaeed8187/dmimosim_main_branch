from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer

def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _sparse_initializer(seed: int, density: float = 0.2, spectral_radius: float = 1.0) -> WeightInitializer:
    input_weight = CompositeInitializer().with_seed(seed).uniform()
    reservoir_weight = (
        CompositeInitializer()
        .with_seed(seed)
        .uniform()
        .sparse(density)
        .spectral_normalize()
        .scale(spectral_radius)
    )
    return WeightInitializer(weight_ih_init=input_weight, weight_hh_init=reservoir_weight)


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
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        *,
        groups: int = 4,
        num_layers: Tuple[int, ...] = (2, 2),
        hidden_size: int = 64,
        density: float = 0.2,
        spectral_radius: float = 1.0,
        leaky: float = 1.0,
        regularization: float = 0.5,
        washout: int = 10,
        seed: int = 0,
    ):
        super().__init__()
        activation = self_normalizing_default(leaky_rate=leaky, spectral_radius=spectral_radius)
        initializer = _sparse_initializer(seed=seed, density=density, spectral_radius=spectral_radius)
        self.esn = GroupedDeepESN(
            input_size=obs_dim,
            output_dim=act_dim,
            groups=groups,
            num_layers=num_layers,
            hidden_size=hidden_size,
            initializer=initializer,
            regularization=regularization,
            activation=activation,
            washout=washout,
            bias=True,
        )
        hidden_dim = groups * hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )
        self.act_limit = act_limit

    def forward(self, x):
        features = self.esn(x)
        return self.act_limit * self.head(features)


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        groups: int = 4,
        num_layers: Tuple[int, ...] = (2, 2),
        hidden_size: int = 64,
        density: float = 0.2,
        spectral_radius: float = 1.0,
        leaky: float = 1.0,
        regularization: float = 0.5,
        washout: int = 10,
        seed: int = 0,
    ):

        super().__init__()
        activation = self_normalizing_default(leaky_rate=leaky, spectral_radius=spectral_radius)
        initializer = _sparse_initializer(seed=seed, density=density, spectral_radius=spectral_radius)
        self.esn = GroupedDeepESN(
            input_size=obs_dim,
            output_dim=act_dim,
            groups=groups,
            num_layers=num_layers,
            hidden_size=hidden_size,
            initializer=initializer,
            regularization=regularization,
            activation=activation,
            washout=washout,
            bias=True,
        )
        hidden_dim = groups * hidden_size

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        features = self.esn(s)
        return self.net(torch.cat([features, a], dim=-1))


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

        self.actor = Actor(obs_dim, act_dim, act_limit, seed=seed).to(self.device)
        self.actor_t = Actor(obs_dim, act_dim, act_limit, seed=seed).to(self.device)
        self.critic = Critic(obs_dim, act_dim, seed=seed).to(self.device)
        self.critic_t = Critic(obs_dim, act_dim, seed=seed).to(self.device)

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
    """Per-receiver and per-transmitter DDPG agents for continuous channel prediction."""

    def __init__(
        self,
        num_receivers: int,
        num_subbands: int,
        scs_per_subband: int = 4*12,
        act_limit: float = 1.0,
        device: Optional[torch.device] = None,
        evaluation_only: bool = False,
        seed: int = 0,
    ):
        self.num_receivers = num_receivers
        self.num_subbands = num_subbands
        self.scs_per_subband = scs_per_subband
        self.act_limit = act_limit
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_only = evaluation_only
        self.seed = seed

        self.obs_dim: Optional[int] = None
        self.act_dim: Optional[int] = None
        self.num_rx_antennas: Optional[int] = None
        self.num_tx_nodes: Optional[int] = None
        self.num_tx_antennas: Optional[int] = None
        self.tx_ant_counts: List[int] = []

        self.agents: List[List[DDPGAgent]] = []

        self.prev_obs: Dict[Tuple[int, int], np.ndarray] = {}
        self.prev_action: Dict[Tuple[int, int], np.ndarray] = {}
        self.predicted_subbands: Dict[Tuple[int, int], np.ndarray] = {}

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

        # First transmitter has four antennas, all others have two, but cap at the provided tx_ant dimension.
        self.tx_ant_counts = [
            min(self.num_tx_antennas, 4 if tx_idx == 0 else 2) for tx_idx in range(self.num_tx_nodes)

        ]

        self.agents = []
        for rx_idx in range(self.num_receivers):
            tx_agents: List[DDPGAgent] = []
            for tx_idx, tx_ant_count in enumerate(self.tx_ant_counts):
                feature_dim = self.num_rx_antennas * tx_ant_count * self.num_subbands
                obs_dim = 2 * feature_dim + 1
                act_dim = 2 * feature_dim
                tx_agents.append(
                    DDPGAgent(
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        act_limit=self.act_limit,
                        device=self.device,
                        seed=self.seed + (rx_idx * self.num_tx_nodes + tx_idx),
                    )
                )
            self.agents.append(tx_agents)

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
        sb_ids = tf.range(tf.shape(mean_t)[0], dtype=tf.int32) // int(self.scs_per_subband)
        sb = tf.math.segment_mean(mean_t, sb_ids)  # [subband, rx, rx_ant, tx_node, tx_ant]
        return tf.transpose(sb, perm=[1, 2, 3, 4, 0]).numpy()


    def predict_channels(self, channel_history: np.ndarray, sinr_db: Optional[np.ndarray] = None) -> tf.Tensor:
        if channel_history is None:
            raise ValueError("channel_history is required for DDPG prediction")
        if sinr_db is None:
            sinr_db = np.zeros(self.num_receivers, dtype=np.float32)

        latest_channel = tf.convert_to_tensor(channel_history[-1,...])
        subbands = np.squeeze(self._extract_subbands(latest_channel))
        predicted_channel = latest_channel.numpy()

        for rx_idx in range(self.num_receivers):

            rx_ants = np.arange(rx_idx*2,(rx_idx+1)*2)

            for tx_idx, tx_ant_count in enumerate(self.tx_ant_counts):

                if tx_idx == 0:
                    tx_ants = np.arange(0,4)
                else:
                    tx_ants = np.arange((tx_idx-1)*2+4,tx_idx*2+4)

                curr_subbands = subbands[rx_ants,...][:, tx_ants, ...]

                obs = self._complex_to_obs(curr_subbands.reshape(-1), float(sinr_db[rx_idx]))
                action = self.agents[rx_idx][tx_idx].act(obs, explore=not self.evaluation_only)
                real, imag = np.split(action, 2)
                pred_sb = (real + 1j * imag).reshape(curr_subbands.shape)

                key = (rx_idx, tx_idx)
                self.predicted_subbands[key] = pred_sb
                self.prev_obs[key] = obs
                self.prev_action[key] = action

                scaling = np.repeat(pred_sb, self.scs_per_subband, axis=-1)[
                    ..., : predicted_channel.shape[-1]
                ].astype(np.complex64)
                scaling = scaling.reshape(
                    (1, 1, scaling.shape[0], 1, scaling.shape[1], 1, scaling.shape[2])
                )

                predicted_channel[:, rx_idx : rx_idx + 1, :, tx_idx : tx_idx + 1, ...] = (
                    predicted_channel[:, rx_idx : rx_idx + 1, :, tx_idx : tx_idx + 1, ...]
                    * scaling
                )

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
            for tx_idx, tx_ant_count in enumerate(self.tx_ant_counts):
                key = (rx_idx, tx_idx)
                if key not in self.prev_obs or key not in self.prev_action:
                    continue

                bler_val = 1.0
                if node_bler is not None and node_bler.size > rx_idx:
                    bler_val = float(np.mean(node_bler[rx_idx]))

                target = true_subbands[rx_idx, :, tx_idx, :tx_ant_count, :]
                target_vec = target.reshape(-1)

                pred = self.predicted_subbands.get(key, np.zeros_like(target_vec))
                pred_vec = pred.reshape(-1)

                err = np.linalg.norm(pred_vec - target_vec) / (np.linalg.norm(target_vec) + 1e-6)
                match_bonus = 1.0 - np.tanh(err)
                reward = (1.0 - bler_val) + match_bonus

                next_obs = self._complex_to_obs(target_vec, float(sinr_db[rx_idx]))
                transition = Transition(
                    state=self.prev_obs[key],
                    action=self.prev_action[key],
                    reward=float(reward),
                    next_state=next_obs,
                    done=0.0,
                )
                self.agents[rx_idx][tx_idx].store_and_maybe_train(
                    transition, evaluation_only=self.evaluation_only
                )
                self.reward_log.append((rx_idx, float(reward)))
                self.prev_obs[key] = next_obs

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
        for rx_idx, tx_agents in enumerate(self.agents):
            for tx_idx, agent in enumerate(tx_agents):
                agent.save(checkpoint_dir / f"rx_{rx_idx}_tx_{tx_idx}.pt")

    def load_all(self, checkpoint_dir: Path) -> None:
        for rx_idx, tx_agents in enumerate(self.agents):
            for tx_idx, agent in enumerate(tx_agents):
                agent.load(checkpoint_dir / f"rx_{rx_idx}_tx_{tx_idx}.pt")


def default_ddpg_predictor(
    num_receivers: int,
    fft_size: int,
    scs_per_subband: int = 4*12,
    evaluation_only: bool = False,
) -> DDPGChannelPredictor:
    num_subbands = int(np.ceil(float(fft_size) / float(scs_per_subband)))
    return DDPGChannelPredictor(
        num_receivers=num_receivers,
        num_subbands=num_subbands,
        scs_per_subband=scs_per_subband,
        evaluation_only=evaluation_only,
    )