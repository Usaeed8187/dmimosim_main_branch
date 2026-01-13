"""Standalone trivial DEQN test (reward=1 for action 0, reward=0 for action 1).

This follows the training cadence used in
ICML_DEQN_clean/main_train_DEQN_modified3_WESN.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
icml_root = repo_root / "ICML_DEQN_clean"
if str(icml_root) not in sys.path:
    sys.path.insert(0, str(icml_root))

import numpy as np
from ICML_DEQN_clean.DQN_RC_new_WESN import DeepWESNQNetwork


def run_trivial_deqn(
    batch_size: int,
    total_episodes: int,
    random_seed: int,
    n_internal_units: int,
    spectral_radius: float,
    output_path: Path,
) -> None:
    rng = np.random.default_rng(random_seed)

    dim_actions = 2
    dim_states = 1
    input_window_length = 4
    output_window_length = 4
    learning_method = "double"

    e_greedy_start = 0.7
    e_greedy_end = 0.9
    e_increase = (e_greedy_end - e_greedy_start) / max(1, (total_episodes // batch_size) - 1)
    epsilon = e_greedy_start
    epsilon_update_period = batch_size

    agent = DeepWESNQNetwork(
        dim_actions,
        dim_states,
        input_window_length,
        output_window_length,
        memory_size=batch_size,
        n_layers=1,
        nInternalUnits=n_internal_units,
        reward_decay=0.9,
        e_greedy=e_greedy_start,
        lr=0.01,
        random_seed=random_seed,
        spectral_radius=spectral_radius,
        training_batch_size=batch_size,
        training_start_threshold=batch_size,
    )

    rewards = np.zeros(total_episodes, dtype=np.float32)
    actions = np.zeros(total_episodes, dtype=np.int64)

    prev_action = 0
    observation = np.array([float(prev_action)], dtype=np.float32)

    for step in range(total_episodes):
        action = agent.choose_action(observation)
        reward = 1.0 if action == 0 else 0.0
        rewards[step] = reward
        actions[step] = action

        next_observation = np.array([float(action)], dtype=np.float32)

        agent.activate_target_net(next_observation)
        agent.store_transition(observation, action, reward, next_observation)

        if (step + 1) % batch_size == 0:
            agent.learn_new(batch_size, step, method=learning_method)
            agent.epsilon = epsilon

        if (step + 1) % epsilon_update_period == 0:
            epsilon = min(e_greedy_end, epsilon + e_increase)

        observation = next_observation

    window_len = 20
    kernel = np.ones(window_len, dtype=float) / float(window_len)
    smoothed_rewards = np.convolve(np.asarray(rewards, dtype=float), kernel, mode="valid")

    plt.figure()
    plt.plot(smoothed_rewards)
    plt.savefig('b')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        rewards=rewards,
        actions=actions,
        batch_size=batch_size,
        total_episodes=total_episodes,
        random_seed=random_seed,
        n_internal_units=n_internal_units,
        spectral_radius=spectral_radius,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trivial DEQN convergence test.")
    parser.add_argument("--batch-size", type=int, default=3, help="Training batch size.")
    parser.add_argument("--total-episodes", type=int, default=50, help="Total steps.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed.")
    parser.add_argument("--n-internal-units", type=int, default=16, help="WESN internal units.")
    parser.add_argument("--spectral-radius", type=float, default=0.9, help="WESN spectral radius.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/trivial_deqn/rewards_actions.npz"),
        help="Output .npz path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_trivial_deqn(
        batch_size=args.batch_size,
        total_episodes=args.total_episodes,
        random_seed=args.seed,
        n_internal_units=args.n_internal_units,
        spectral_radius=args.spectral_radius,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()